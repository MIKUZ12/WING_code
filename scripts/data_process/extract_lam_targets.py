#!/usr/bin/env python3
"""Generate generic per-episode latent action targets from one camera stream.

The source dataset remains immutable.  This command writes a sidecar cache:
``dataset_XX/episode_XXXXXX.pt`` contains ``latent_target[L,H,D]`` and
``latent_target_mask[L,H]``.  Camera key, temporal offsets, dimensions, and
identity metadata are explicit command-line/config inputs; no checkpoint hash
or LIBERO path is assumed by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from wing.datasets.spectral_targets import SPECTRAL_CACHE_SCHEMA_VERSION  # noqa: E402
from wing.latent_action import (  # noqa: E402
    load_rgb_student_checkpoint,
    preprocess_rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", required=True, help="LeRobot root; repeat for multiple datasets in training order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--student",
        type=Path,
        required=True,
        help="RGB student trained-state checkpoint; use --lam when its recorded architecture path is stale.",
    )
    parser.add_argument(
        "--lam",
        type=Path,
        default=None,
        help="Local LAM architecture checkpoint override for students with a stale recorded path.",
    )
    parser.add_argument("--camera-key", required=True, help="LeRobot video feature, e.g. observation.images.image")
    parser.add_argument("--frame-offsets", default="0,4,8,12,16,20,24,28,32", help="strictly increasing offsets beginning at 0")
    parser.add_argument("--expected-fps", type=float, default=None, help="optional dataset/video fps contract")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None, help="bounded smoke extraction; cache remains incomplete")
    parser.add_argument("--identity-json", type=Path, default=None, help="optional user-defined JSON identity stored verbatim in manifest")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: os.fspath(value) if isinstance(value, Path) else _raise_json_type(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _raise_json_type(value: Any) -> None:
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No episode metadata in {path}")
    return rows


def _dataset_manifest(root: Path, dataset_index: int, camera_key: str, expected_fps: float | None) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(camera_key)
    if feature is None or feature.get("dtype") != "video":
        raise ValueError(f"{camera_key!r} is not a video feature in {info_path}")
    fps = float(info["fps"])
    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"Dataset fps mismatch at {root}: expected {expected_fps}, got {fps}")
    video_path_template = info.get("video_path")
    if not isinstance(video_path_template, str) or "{video_key}" not in video_path_template:
        raise ValueError(f"LeRobot info.json must provide a video_path template with {{video_key}}: {info_path}")
    return {
        "dataset_index": dataset_index,
        "root": os.path.realpath(root),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "fps": fps,
        "chunks_size": int(info["chunks_size"]),
        "camera_key": camera_key,
        "camera_shape_hwc": feature.get("shape"),
        "video_path_template": video_path_template,
    }


def _decode_native_rgb(path: Path, length: int, expected_fps: float | None) -> np.ndarray:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if expected_fps is not None and float(stream.average_rate) != expected_fps:
            raise ValueError(f"Video fps mismatch in {path}: expected {expected_fps}, got {stream.average_rate}")
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != length:
        raise ValueError(f"Decoded frame count mismatch in {path}: expected {length}, got {len(frames)}")
    return np.stack(frames)


@torch.inference_mode()
def _encode_episode(model: torch.nn.Module, frames: np.ndarray, offsets: torch.Tensor, device: torch.device, sample_batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_batch_size <= 0:
        raise ValueError("sample-batch-size must be positive")
    prepared = torch.stack([preprocess_rgb_frame(frame) for frame in frames])
    length, horizon = prepared.shape[0], len(offsets)
    valid = torch.arange(length).unsqueeze(1) + offsets.unsqueeze(0) < length
    outputs: list[torch.Tensor] = []
    for start in range(0, length, sample_batch_size):
        stop = min(start + sample_batch_size, length)
        starts = torch.arange(start, stop).unsqueeze(1)
        indices = (starts + offsets.unsqueeze(0)).clamp_max(length - 1)
        clips = prepared[indices]
        pairs = torch.stack((clips[:, :-1], clips[:, 1:]), dim=2).reshape(-1, 2, *prepared.shape[1:])
        values = model(pairs.to(device, non_blocking=True)).detach().float().cpu()
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError("RGB student produced invalid target values")
        outputs.append(values.reshape(stop - start, horizon - 1, values.shape[-1]))
    return torch.cat(outputs), valid[:, 1:]


def main() -> None:
    args = parse_args()
    offsets = tuple(int(item) for item in args.frame_offsets.split(",") if item.strip())
    if len(offsets) < 2 or offsets[0] != 0 or any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("frame-offsets must contain at least [0, positive...] in strictly increasing order")
    roots = [Path(value).resolve() for value in args.datasets]
    if len(set(roots)) != len(roots):
        raise ValueError("dataset roots must not repeat")
    identity = None if args.identity_json is None else json.loads(args.identity_json.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, student_metadata = load_rgb_student_checkpoint(
        args.student,
        device=device,
        lam_checkpoint=args.lam,
    )
    model.eval().requires_grad_(False)
    # Normalize persistent metadata before both comparison and serialization;
    # legacy checkpoints commonly store pathlib.Path values.
    student_metadata = json.loads(
        json.dumps(
            student_metadata,
            default=lambda value: os.fspath(value) if isinstance(value, Path) else _raise_json_type(value),
        )
    )
    # Checkpoint metadata is the persistent public contract.  The in-memory
    # module currently exposes `latent_dim`, but label generation must not rely
    # on that implementation detail.
    latent_dim = int(student_metadata.get("latent_dim", model.head.out_features))
    datasets = [_dataset_manifest(root, index, args.camera_key, args.expected_fps) for index, root in enumerate(roots)]
    manifest = {
        "schema_version": SPECTRAL_CACHE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "datasets": datasets,
        "target": {"key": "latent_target", "mask_key": "latent_target_mask", "horizon": len(offsets) - 1, "dimension": latent_dim},
        "temporal": {"frame_offsets": list(offsets), "padding": "clamp_episode_end_and_mask_invalid_pairs"},
        "student": {"checkpoint": os.path.realpath(args.student), "metadata": student_metadata},
        "identity": identity,
    }
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "datasets", "target", "temporal", "student", "identity"):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Existing cache contract differs at {key}: {manifest_path}")
    else:
        _atomic_json_save(manifest, manifest_path)

    completed = 0
    for dataset_index, root in enumerate(roots):
        for row in _load_episodes(root):
            if args.max_episodes is not None and completed >= args.max_episodes:
                manifest["completion"] = {"status": "partial", "episode_shards": completed}
                _atomic_json_save(manifest, manifest_path)
                print(json.dumps(manifest["completion"], sort_keys=True))
                return
            episode_index, episode_length = int(row["episode_index"]), int(row["length"])
            shard = output / f"dataset_{dataset_index:02d}" / f"episode_{episode_index:06d}.pt"
            if shard.exists() and not args.overwrite:
                completed += 1
                continue
            video_path = root / datasets[dataset_index]["video_path_template"].format(
                episode_chunk=episode_index // int(datasets[dataset_index]["chunks_size"]),
                video_key=args.camera_key,
                episode_index=episode_index,
            )
            target, mask = _encode_episode(model, _decode_native_rgb(video_path, episode_length, args.expected_fps), torch.tensor(offsets), device, args.sample_batch_size)
            _atomic_torch_save({
                "schema_version": SPECTRAL_CACHE_SCHEMA_VERSION,
                "latent_target": target,
                "latent_target_mask": mask,
                "source": {"dataset_root": os.path.realpath(root), "episode_index": episode_index, "episode_length": episode_length, "camera_key": args.camera_key, "video_path": os.path.realpath(video_path)},
            }, shard)
            completed += 1
            print(json.dumps({"status": "episode_complete", "dataset_index": dataset_index, "episode_index": episode_index, "completed": completed}, sort_keys=True), flush=True)
    expected = sum(item["total_episodes"] for item in datasets)
    if completed != expected:
        raise RuntimeError(f"Extraction produced {completed} episodes; metadata requires {expected}")
    manifest["completion"] = {"status": "complete", "episode_shards": completed}
    _atomic_json_save(manifest, manifest_path)
    print(json.dumps(manifest["completion"], sort_keys=True))


if __name__ == "__main__":
    main()
