#!/usr/bin/env python3
"""Extract each unique fixed-gap latent action RGB-student pair once per episode.

The source datasets remain immutable.  Each output episode stores
``latent_pair_target[L,D]`` and ``latent_pair_valid[L]``.  For uniform offsets
``[0,g,...,Hg]``, H-step window ``t`` is a view over pair rows
``t + [0,g,...,(H-1)g]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from wing.datasets.spectral_targets import (  # noqa: E402
    SPECTRAL_CACHE_SCHEMA_VERSION,
    assemble_compact_latent_targets,
)
from wing.latent_action import (  # noqa: E402
    load_rgb_student_checkpoint,
    preprocess_rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--lam", type=Path, default=None)
    parser.add_argument("--camera-key", required=True)
    parser.add_argument("--pair-gap", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--expected-fps", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Bounded per-worker smoke extraction; omit for a full shard.",
    )
    parser.add_argument(
        "--max-source-groups", type=int, default=None,
        help="Bounded per-worker source groups for concurrency probes.",
    )
    parser.add_argument("--identity-json", type=Path, default=None)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--require-existing-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=os.fspath) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No episode metadata in {path}")
    return rows


def source_group_owners(group_keys: list[str], num_shards: int) -> list[int]:
    """Assign every repeated source-video key to one deterministic worker."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    ordinals: dict[str, int] = {}
    owners: list[int] = []
    for key in group_keys:
        if key not in ordinals:
            ordinals[key] = len(ordinals)
        owners.append(ordinals[key] % num_shards)
    return owners


@dataclass(frozen=True)
class EpisodeJob:
    dataset_index: int
    dataset_root: Path
    episode_index: int
    episode_length: int
    frame_start: int
    shard: Path
    video_path: Path
    source_group: str


def _dataset_manifest(
    root: Path, dataset_index: int, camera_key: str, expected_fps: float | None
) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(camera_key)
    if feature is None or feature.get("dtype") != "video":
        raise ValueError(f"{camera_key!r} is not a video feature in {info_path}")
    fps = float(info["fps"])
    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"Dataset fps mismatch at {root}: expected {expected_fps}, got {fps}")
    template = info.get("video_path")
    if not isinstance(template, str) or "{video_key}" not in template:
        raise ValueError(f"Invalid video_path template in {info_path}")
    return {
        "dataset_index": dataset_index,
        "root": os.path.realpath(root),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "fps": fps,
        "chunks_size": int(info["chunks_size"]),
        "camera_key": camera_key,
        "camera_shape": feature.get("shape"),
        "video_path_template": template,
    }


def _student_manifest_metadata(student: Path, lam: Path | None) -> tuple[dict[str, Any], int]:
    payload = torch.load(student, map_location="cpu", weights_only=False, mmap=True)
    metadata = dict(payload["metadata"])
    metadata["checkpoint_format_version"] = int(payload.get("format_version", 0))
    resolved_lam = Path(lam) if lam is not None else Path(metadata["lam_checkpoint"])
    if not resolved_lam.is_file():
        raise FileNotFoundError(resolved_lam)
    metadata["resolved_lam_checkpoint"] = str(resolved_lam.resolve())
    metadata = json.loads(json.dumps(metadata, default=os.fspath))
    return metadata, int(metadata.get("latent_dim", 32))


def _flush_group_pairs(
    model: torch.nn.Module,
    pending: list[torch.Tensor],
    pending_starts: list[int],
    consumers: dict[int, list[tuple[int, int]]],
    targets: list[torch.Tensor],
    device: torch.device,
) -> int:
    if not pending:
        return 0
    values = model(torch.stack(pending).to(device, non_blocking=True)).detach().float().cpu()
    if values.ndim != 2 or len(values) != len(pending_starts) or not bool(torch.isfinite(values).all()):
        raise ValueError("RGB student produced invalid target values")
    for value, pair_start in zip(values, pending_starts):
        for episode_position, local_index in consumers[pair_start]:
            targets[episode_position][local_index] = value
    count = len(pending)
    pending.clear()
    pending_starts.clear()
    return count


@torch.inference_mode()
def encode_source_group(
    model: torch.nn.Module,
    video_path: Path,
    *,
    episodes: list[EpisodeJob],
    pair_gap: int,
    expected_fps: float | None,
    device: torch.device,
    pair_batch_size: int,
    latent_dim: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    """Open one source once and share unique absolute pairs across its episodes."""
    import av

    if not episodes:
        raise ValueError("episodes must not be empty")
    if pair_gap <= 0 or pair_batch_size <= 0:
        raise ValueError("pair_gap and pair_batch_size must be positive")
    if any(job.video_path != video_path or job.frame_start < 0 for job in episodes):
        raise ValueError("source-group episodes must share one video and nonnegative starts")
    targets = [
        torch.zeros(job.episode_length, latent_dim, dtype=torch.float32)
        for job in episodes
    ]
    masks = [
        torch.zeros(job.episode_length, dtype=torch.bool)
        for job in episodes
    ]
    consumers: dict[int, list[tuple[int, int]]] = {}
    assignments = 0
    for episode_position, job in enumerate(episodes):
        valid_count = max(job.episode_length - pair_gap, 0)
        masks[episode_position][:valid_count] = True
        assignments += valid_count
        for local_index in range(valid_count):
            consumers.setdefault(job.frame_start + local_index, []).append(
                (episode_position, local_index)
            )
    if not consumers:
        return list(zip(targets, masks)), {
            "source_frames_decoded": 0,
            "unique_pairs": 0,
            "episode_pair_assignments": 0,
        }
    needed_starts = set(consumers)
    final_endpoint = max(needed_starts) + pair_gap
    prepared_sources: dict[int, torch.Tensor] = {}
    pending: list[torch.Tensor] = []
    pending_starts: list[int] = []
    source_decoded = 0
    inferred = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if expected_fps is not None and float(stream.average_rate) != expected_fps:
            raise ValueError(
                f"Video fps mismatch in {video_path}: expected {expected_fps}, got {stream.average_rate}"
            )
        for frame in container.decode(video=0):
            source_index = source_decoded
            source_decoded += 1
            if source_index > final_endpoint:
                break
            pair_start = source_index - pair_gap
            needs_preprocess = source_index in needed_starts or pair_start in needed_starts
            if not needs_preprocess:
                continue
            prepared = preprocess_rgb_frame(frame.to_ndarray(format="rgb24"))
            if source_index in needed_starts:
                prepared_sources[source_index] = prepared
            if pair_start in needed_starts:
                source = prepared_sources.pop(pair_start, None)
                if source is None:
                    raise RuntimeError(f"Missing prepared source frame {pair_start}")
                pending.append(torch.stack((source, prepared)))
                pending_starts.append(pair_start)
                if len(pending) == pair_batch_size:
                    inferred += _flush_group_pairs(
                        model, pending, pending_starts, consumers, targets, device
                    )
    inferred += _flush_group_pairs(
        model, pending, pending_starts, consumers, targets, device
    )
    if inferred != len(needed_starts):
        raise ValueError(
            f"Source video ended before all pairs were encoded: "
            f"expected={len(needed_starts)}, got={inferred}, path={video_path}"
        )
    return list(zip(targets, masks)), {
        "source_frames_decoded": source_decoded,
        "unique_pairs": inferred,
        "episode_pair_assignments": assignments,
    }


def _valid_existing_shard(
    path: Path,
    *,
    dataset_root: Path,
    episode_index: int,
    episode_length: int,
    frame_start: int,
    video_path: Path,
    pair_gap: int,
    latent_dim: int,
) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = payload["source"]
        pair_valid = payload["latent_pair_valid"]
        valid_count = max(episode_length - pair_gap, 0)
        return (
            tuple(payload["latent_pair_target"].shape) == (episode_length, latent_dim)
            and tuple(pair_valid.shape) == (episode_length,)
            and pair_valid.dtype == torch.bool
            and bool(pair_valid[:valid_count].all())
            and not bool(pair_valid[valid_count:].any())
            and bool(torch.isfinite(payload["latent_pair_target"]).all())
            and os.path.realpath(source["dataset_root"]) == os.path.realpath(dataset_root)
            and int(source["episode_index"]) == episode_index
            and int(source["episode_length"]) == episode_length
            and int(source.get("frame_start", 0)) == frame_start
            and os.path.realpath(source["video_path"]) == os.path.realpath(video_path)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.pair_gap <= 0 or args.horizon <= 0:
        raise ValueError("pair-gap and horizon must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must lie in [0,num-shards)")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    if args.max_source_groups is not None and args.max_source_groups <= 0:
        raise ValueError("max-source-groups must be positive")
    roots = [Path(value).resolve() for value in args.datasets]
    if len(set(roots)) != len(roots):
        raise ValueError("dataset roots must not repeat")
    identity = (
        None
        if args.identity_json is None
        else json.loads(args.identity_json.read_text(encoding="utf-8"))
    )
    student_metadata, latent_dim = _student_manifest_metadata(args.student, args.lam)
    datasets = [
        _dataset_manifest(root, index, args.camera_key, args.expected_fps)
        for index, root in enumerate(roots)
    ]
    offsets = [index * args.pair_gap for index in range(args.horizon + 1)]
    manifest = {
        "schema_version": SPECTRAL_CACHE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "datasets": datasets,
        "target": {
            "storage": "compact_fixed_gap_pairs",
            "key": "latent_pair_target",
            "mask_key": "latent_pair_valid",
            "dimension": latent_dim,
            "horizon": args.horizon,
        },
        "temporal": {
            "pair_gap_frames": args.pair_gap,
            "frame_offsets": offsets,
            "assembly": "target[t,j]=latent_pair_target[t+frame_offsets[j]]",
        },
        "student": {
            "checkpoint": os.path.realpath(args.student),
            "metadata": student_metadata,
        },
        "identity": identity,
    }
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if args.require_existing_manifest and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Worker requires an initialized compact manifest: {manifest_path}"
        )
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version", "datasets", "target", "temporal", "student", "identity"
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(
                    f"Existing compact cache contract differs at {key}: {manifest_path}"
                )
    else:
        _atomic_json_save(manifest, manifest_path)
    if args.initialize_only:
        print(json.dumps({"status": "initialized", "manifest": str(manifest_path)}))
        return

    device = torch.device(args.device)
    model, loaded_metadata = load_rgb_student_checkpoint(
        args.student, device=device, lam_checkpoint=args.lam
    )
    model.eval().requires_grad_(False)
    loaded_metadata = json.loads(json.dumps(loaded_metadata, default=os.fspath))
    if loaded_metadata != student_metadata:
        raise ValueError("Student metadata changed between manifest initialization and model load")

    all_jobs: list[EpisodeJob] = []
    for dataset_index, root in enumerate(roots):
        for row in _load_episodes(root):
            episode_index, episode_length = int(row["episode_index"]), int(row["length"])
            frame_start = int(row.get("frame_start", 0))
            shard = (
                output
                / f"dataset_{dataset_index:02d}"
                / f"episode_{episode_index:06d}.pt"
            )
            if "source_video" in row:
                video_path = Path(row["source_video"]).resolve()
                source_group = f"{dataset_index}:{os.path.realpath(video_path)}"
            else:
                video_path = root / datasets[dataset_index]["video_path_template"].format(
                    episode_chunk=episode_index // int(datasets[dataset_index]["chunks_size"]),
                    video_key=args.camera_key,
                    episode_index=episode_index,
                )
                source_group = f"{dataset_index}:{os.path.realpath(video_path)}"
            all_jobs.append(
                EpisodeJob(
                    dataset_index=dataset_index,
                    dataset_root=root,
                    episode_index=episode_index,
                    episode_length=episode_length,
                    frame_start=frame_start,
                    shard=shard,
                    video_path=video_path,
                    source_group=source_group,
                )
            )
    owners = source_group_owners(
        [job.source_group for job in all_jobs], args.num_shards
    )
    jobs = [
        job for job, owner in zip(all_jobs, owners) if owner == args.shard_index
    ]
    if args.max_source_groups is not None:
        selected_groups = list(dict.fromkeys(job.source_group for job in jobs))[
            : args.max_source_groups
        ]
        selected = set(selected_groups)
        jobs = [job for job in jobs if job.source_group in selected]
    if args.max_episodes is not None:
        jobs = jobs[: args.max_episodes]

    grouped_jobs: dict[str, list[EpisodeJob]] = {}
    for job in jobs:
        grouped_jobs.setdefault(job.source_group, []).append(job)

    completed = skipped = unique_pairs_done = source_groups_decoded = 0
    started = time.monotonic()
    for source_group, group_jobs in grouped_jobs.items():
        pending_jobs: list[EpisodeJob] = []
        for job in group_jobs:
            if job.shard.exists() and not args.overwrite:
                if not _valid_existing_shard(
                    job.shard,
                    dataset_root=job.dataset_root,
                    episode_index=job.episode_index,
                    episode_length=job.episode_length,
                    frame_start=job.frame_start,
                    video_path=job.video_path,
                    pair_gap=args.pair_gap,
                    latent_dim=latent_dim,
                ):
                    raise ValueError(f"Existing compact shard is invalid: {job.shard}")
                skipped += 1
            else:
                pending_jobs.append(job)
        if not pending_jobs:
            continue
        video_paths = {job.video_path for job in pending_jobs}
        if len(video_paths) != 1:
            raise AssertionError("One source group resolved to multiple video paths")
        episode_outputs, group_metrics = encode_source_group(
            model,
            next(iter(video_paths)),
            episodes=pending_jobs,
            pair_gap=args.pair_gap,
            expected_fps=args.expected_fps,
            device=device,
            pair_batch_size=args.pair_batch_size,
            latent_dim=latent_dim,
        )
        source_groups_decoded += 1
        unique_pairs_done += int(group_metrics["unique_pairs"])
        for job, (target, valid) in zip(pending_jobs, episode_outputs):
            assembled, assembled_mask = assemble_compact_latent_targets(
                target, valid, frame_offsets=offsets, pair_gap=args.pair_gap
            )
            if tuple(assembled.shape) != (
                job.episode_length, args.horizon, latent_dim
            ):
                raise AssertionError("Compact H-step assembly shape is inconsistent")
            _atomic_torch_save(
                {
                    "schema_version": SPECTRAL_CACHE_SCHEMA_VERSION,
                    "latent_pair_target": target,
                    "latent_pair_valid": valid,
                    "source": {
                        "dataset_root": os.path.realpath(job.dataset_root),
                        "dataset_index": job.dataset_index,
                        "episode_index": job.episode_index,
                        "episode_length": job.episode_length,
                        "frame_start": job.frame_start,
                        "camera_key": args.camera_key,
                        "video_path": os.path.realpath(job.video_path),
                        "source_group": source_group,
                    },
                },
                job.shard,
            )
            completed += 1
        elapsed = max(time.monotonic() - started, 1.0e-6)
        print(
            json.dumps(
                {
                    "status": "source_group_complete",
                    "source_group": source_group,
                    "source_group_episodes": len(pending_jobs),
                    **group_metrics,
                    "worker_completed": completed,
                    "worker_source_groups_decoded": source_groups_decoded,
                    "unique_pairs_per_second": unique_pairs_done / elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    progress = {
        "status": "worker_complete",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_episodes": len(jobs),
        "new_episode_shards": completed,
        "skipped_episode_shards": skipped,
        "source_groups_decoded": source_groups_decoded,
        "unique_pairs_encoded": unique_pairs_done,
        "bounded": args.max_episodes is not None or args.max_source_groups is not None,
    }
    _atomic_json_save(
        progress,
        output
        / "workers"
        / f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}.json",
    )
    print(json.dumps(progress, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
