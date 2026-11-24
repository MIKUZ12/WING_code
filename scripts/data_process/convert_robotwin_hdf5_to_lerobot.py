#!/usr/bin/env python3
"""Convert RoboTwin 2.0 aloha-agilex HDF5 episodes into a WING LeRobot v2.1 dataset.

RoboTwin 2.0 (HF TianxingChen/RoboTwin2.0) ships each task/embodiment/split as a zip
whose ``data/episode*.hdf5`` are the rendered episodes. Each hdf5 contains:

    joint_action/vector          (T, 14) float64  [L 6 joints + L grip + R 6 joints + R grip]
    observation/head_camera/rgb  (T,) jpg-encoded bytes, decoded -> 240x320x3
    observation/left_camera/rgb  (T,) jpg-encoded bytes
    observation/right_camera/rgb (T,) jpg-encoded bytes
    (front_camera / endpose / pointcloud also present -- unused)

WING's robot pipeline wants 14-dim joint+gripper action/state and cameras
cam_high / cam_left_wrist / cam_right_wrist (concat_multi_camera="robotwin"), so we map:

    joint_action/vector -> action AND observation.state
        (the sim is position-controlled; the per-step joint vector is both the commanded
         target and the achieved state, so we use it for both -- the WING dataset then
         slices it into a current-frame proprio + a future action chunk)
    head_camera  -> cam_high
    left_camera  -> cam_left_wrist
    right_camera -> cam_right_wrist

The output follows the LeRobot schema consumed by the WING dataloader and
normalizer.

NOTE: adjust_bottle is a right-arm task, so the left 7 dims are constant across an
episode. z-score normalization stays finite (std_reg=1e-8 + clamp(-5,5)); we keep all
14 dims so RoboTwin eval drives both arms (the model learns the left arm stays put).

Usage:
    python scripts/data_process/convert_robotwin_hdf5_to_lerobot.py \
        --src ./data/robotwin2.0/_raw/aloha-agilex_randomized_500/data \
        --dst ./data/robotwin2.0/adjust_bottle_aloha_randomized \
        --task-name adjust_bottle \
        --instruction "Use the right arm to pick the plastic bottle up and keep it head-up." \
        --fps 30 --overwrite
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from pathlib import Path


JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

# (hdf5 camera group, lerobot camera key, lerobot feature key)
ROBOTWIN_CAMERA_SPECS = (
    ("head_camera", "cam_high", "observation.images.cam_high"),
    ("left_camera", "cam_left_wrist", "observation.images.cam_left_wrist"),
    ("right_camera", "cam_right_wrist", "observation.images.cam_right_wrist"),
)

CAM_HEIGHT = 240
CAM_WIDTH = 320


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    project_root = _project_root()
    src_root = project_root / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _require_runtime_deps():
    missing: list[str] = []
    modules: dict[str, object] = {}
    import_names = {
        "h5py": "h5py",
        "numpy": "numpy",
        "PIL.Image": "Pillow",
        "datasets": "datasets",
        "torch": "torch",
        "av": "av",
    }
    for module_name, package_name in import_names.items():
        try:
            if module_name == "PIL.Image":
                from PIL import Image  # type: ignore

                modules["Image"] = Image
            else:
                modules[module_name] = __import__(module_name)
        except Exception:
            missing.append(package_name)

    if missing:
        pkg_text = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing Python dependencies required for conversion: "
            f"{pkg_text}. Activate the WING training environment first."
        )

    _add_repo_paths()
    try:
        from wing.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "Failed to import WING LeRobot dataset helpers from the repository. "
            "Run this script from the repo with the WING environment activated."
        ) from exc

    modules["LeRobotDataset"] = LeRobotDataset
    return modules


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode(\d+)\.hdf5$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _find_episode_files(src: Path) -> list[Path]:
    return sorted(src.glob("episode*.hdf5"), key=_episode_sort_key)


def _parse_keep_dims(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"--keep-dims slice must be 'start:stop', got {spec!r}")
        start, stop = int(parts[0]), int(parts[1])
        return list(range(start, stop))
    return [int(x) for x in text.split(",") if x.strip()]


def _build_features(kept_indices: list[int] | None) -> dict[str, dict]:
    indices = list(range(len(JOINT_NAMES))) if kept_indices is None else kept_indices
    kept_names = [JOINT_NAMES[i] for i in indices]
    dim = len(indices)

    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": (dim,), "names": kept_names},
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": kept_names},
    }
    for _, _, feature_key in ROBOTWIN_CAMERA_SPECS:
        features[feature_key] = {
            "dtype": "video",
            "shape": (3, CAM_HEIGHT, CAM_WIDTH),
            "names": ["channels", "height", "width"],
        }
    return features


def _decode_rgb(image_module, np, rgb_ds, frame_idx: int):
    """RoboTwin stores each frame as variable-length jpg bytes -> decode to HxWx3 uint8."""
    raw = rgb_ds[frame_idx]
    buf = io.BytesIO(bytes(raw))
    arr = np.asarray(image_module.open(buf).convert("RGB"), dtype=np.uint8)
    return arr


def _save_jpeg_frame(image_module, frame, output_path: Path, jpeg_quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_module.fromarray(frame).save(output_path, format="JPEG", quality=jpeg_quality)


def _prepare_episode_buffer(dataset, episode_index: int):
    episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)
    episode_buffer["size"] = 0
    return episode_buffer


def _convert_episode(
    h5py,
    np,
    image_module,
    dataset,
    episode_path: Path,
    instruction: str,
    jpeg_quality: int,
    max_frames: int | None,
    kept_indices: list[int] | None,
) -> int:
    with h5py.File(episode_path, "r") as h5f:
        if "joint_action/vector" not in h5f:
            raise ValueError(f"{episode_path}: missing joint_action/vector")
        vec_ds = h5f["joint_action/vector"]
        cam_datasets = {
            feature_key: h5f[f"observation/{cam_group}/rgb"]
            for cam_group, _, feature_key in ROBOTWIN_CAMERA_SPECS
        }

        seq_len = int(vec_ds.shape[0])
        if tuple(vec_ds.shape[1:]) != (14,):
            raise ValueError(
                f"{episode_path}: expected joint_action/vector [T, 14], got {vec_ds.shape}"
            )
        for cam_group, _, feature_key in ROBOTWIN_CAMERA_SPECS:
            cam_len = int(cam_datasets[feature_key].shape[0])
            if cam_len != seq_len:
                raise ValueError(
                    f"{episode_path}: {cam_group} has {cam_len} frames, expected {seq_len}"
                )

        if max_frames is not None:
            seq_len = min(seq_len, int(max_frames))

        dataset.episode_buffer = _prepare_episode_buffer(dataset, dataset.meta.total_episodes)
        episode_buffer = dataset.episode_buffer
        per_frame_task = [instruction, instruction, instruction, instruction]

        for frame_idx in range(seq_len):
            episode_buffer["frame_index"].append(frame_idx)
            episode_buffer["timestamp"].append(np.float32(frame_idx / dataset.fps))
            episode_buffer["task"].append(per_frame_task)

            joint_row = np.asarray(vec_ds[frame_idx], dtype=np.float32)
            if kept_indices is not None:
                joint_row = joint_row[kept_indices]
            # Position-controlled sim: the per-step joint vector is both action and state.
            episode_buffer["action"].append(joint_row)
            episode_buffer["observation.state"].append(joint_row.copy())

            for _, _, feature_key in ROBOTWIN_CAMERA_SPECS:
                frame = _decode_rgb(image_module, np, cam_datasets[feature_key], frame_idx)
                image_path = dataset._get_image_file_path(
                    episode_index=episode_buffer["episode_index"],
                    image_key=feature_key,
                    frame_index=frame_idx,
                )
                _save_jpeg_frame(image_module, frame, image_path, jpeg_quality)
                episode_buffer[feature_key].append(str(image_path))

            episode_buffer["size"] += 1

    dataset.save_episode(raw_file_name=episode_path.name)
    return seq_len


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin 2.0 HDF5 episodes into a LeRobot v2.1 dataset for WING."
    )
    parser.add_argument("--src", required=True, help="Dir containing RoboTwin episode*.hdf5 (the unpacked <task>/data dir).")
    parser.add_argument("--dst", required=True, help="Destination LeRobot dataset directory.")
    parser.add_argument("--task-name", required=True, help="Human-readable dataset/task label for logs.")
    parser.add_argument(
        "--instruction",
        default="Use the right arm to pick the plastic bottle up and keep it head-up.",
        help="Instruction written for every episode (single fixed prompt -> 1 text-embed cache entry).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS metadata. Sim fps does not affect WING frame-window sampling; defaults to 30.")
    parser.add_argument("--robot-type", default="aloha", help="Robot type written into meta/info.json.")
    parser.add_argument(
        "--video-codec",
        default="h264",
        choices=["h264", "hevc", "libsvtav1", "h264_nvenc"],
        help="Codec used when encoding LeRobot mp4s. Defaults to h264.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for temporary frame dumps before mp4 encoding.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Convert only the first N episodes (smoke test).")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode index to convert (for sharded parallel conversion).")
    parser.add_argument("--episode-stop", type=int, default=None, help="Stop episode index, exclusive (for sharded parallel conversion).")
    parser.add_argument("--max-frames", type=int, default=None, help="Truncate each episode to the first N frames (smoke test).")
    parser.add_argument("--overwrite", action="store_true", help="Delete the destination directory first if it exists.")
    parser.add_argument(
        "--keep-dims",
        default=None,
        help=(
            "Restrict saved action/state to a subset of joint indices. Slice 'start:stop' "
            "(e.g. '7:14' = right arm only) or comma indices. Default: keep all 14."
        ),
    )
    return parser


def main() -> int:
    args = _make_parser().parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    episode_files = _find_episode_files(src)
    episode_files = episode_files[args.episode_start : args.episode_stop]
    if args.limit_episodes is not None:
        episode_files = episode_files[: args.limit_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found under: {src}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Pass --overwrite to recreate it.")
        shutil.rmtree(dst)

    runtime = _require_runtime_deps()
    h5py = runtime["h5py"]
    np = runtime["numpy"]
    image_module = runtime["Image"]
    LeRobotDataset = runtime["LeRobotDataset"]

    kept_indices = _parse_keep_dims(args.keep_dims)
    if kept_indices is not None:
        for idx in kept_indices:
            if idx < 0 or idx >= len(JOINT_NAMES):
                raise ValueError(f"--keep-dims index {idx} out of range [0, {len(JOINT_NAMES)}).")
        print(f"[convert] keep_dims={kept_indices} ({','.join(JOINT_NAMES[i] for i in kept_indices)})")
    else:
        print(f"[convert] keep_dims=all {len(JOINT_NAMES)}")

    print(f"[convert] task_name={args.task_name}")
    print(f"[convert] src={src}")
    print(f"[convert] dst={dst}")
    print(f"[convert] episodes={len(episode_files)}")
    print(f"[convert] instruction={args.instruction!r}")
    print(f"[convert] video_codec={args.video_codec}")

    dataset = LeRobotDataset.create(
        repo_id=dst.name,
        root=dst,
        fps=args.fps,
        robot_type=args.robot_type,
        features=_build_features(kept_indices),
        use_videos=True,
        video_codec=args.video_codec,
        is_compute_episode_stats_image=False,
    )

    total_frames = 0
    for episode_idx, episode_path in enumerate(episode_files, start=1):
        frame_count = _convert_episode(
            h5py=h5py,
            np=np,
            image_module=image_module,
            dataset=dataset,
            episode_path=episode_path,
            instruction=args.instruction,
            jpeg_quality=args.jpeg_quality,
            max_frames=args.max_frames,
            kept_indices=kept_indices,
        )
        total_frames += frame_count
        print(f"[convert] episode={episode_idx}/{len(episode_files)} name={episode_path.name} frames={frame_count}")

    print(
        f"[done] wrote dataset to {dst} | total_episodes={dataset.meta.total_episodes} "
        f"| total_frames={total_frames} | total_tasks={dataset.meta.total_tasks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
