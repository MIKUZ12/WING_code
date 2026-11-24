#!/usr/bin/env python3
"""Validate every expected compact episode shard, then mark the cache complete."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "extract_lam_pairs_finalize",
    REPO_ROOT / "scripts/data_process/extract_lam_pairs.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_EXTRACTOR = importlib.util.module_from_spec(_SPEC)
import sys
sys.modules[_SPEC.name] = _EXTRACTOR
_SPEC.loader.exec_module(_EXTRACTOR)


def _atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    args = parser.parse_args()
    if args.expected_workers <= 0:
        raise ValueError("expected-workers must be positive")
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latent_dim = int(manifest["target"]["dimension"])
    pair_gap = int(manifest["temporal"]["pair_gap_frames"])
    for shard_index in range(args.expected_workers):
        progress_path = (
            args.cache_root / "workers"
            / f"shard_{shard_index:05d}_of_{args.expected_workers:05d}.json"
        )
        if not progress_path.is_file():
            raise FileNotFoundError(f"Missing unbounded worker progress: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("status") != "worker_complete"
            or int(progress.get("num_shards", -1)) != args.expected_workers
            or int(progress.get("shard_index", -1)) != shard_index
            or bool(progress.get("bounded", True))
        ):
            raise ValueError(f"Invalid formal worker progress: {progress_path}")
    checked = 0
    for dataset in manifest["datasets"]:
        dataset_index = int(dataset["dataset_index"])
        root = Path(dataset["root"])
        rows = _EXTRACTOR._load_episodes(root)
        for row in rows:
            episode_index = int(row["episode_index"])
            episode_length = int(row["length"])
            frame_start = int(row.get("frame_start", 0))
            if "source_video" in row:
                video_path = Path(row["source_video"]).resolve()
            else:
                video_path = root / dataset["video_path_template"].format(
                    episode_chunk=episode_index // int(dataset["chunks_size"]),
                    video_key=dataset["camera_key"],
                    episode_index=episode_index,
                )
            shard = (
                args.cache_root / f"dataset_{dataset_index:02d}"
                / f"episode_{episode_index:06d}.pt"
            )
            if not shard.is_file() or not _EXTRACTOR._valid_existing_shard(
                shard,
                dataset_root=root,
                episode_index=episode_index,
                episode_length=episode_length,
                frame_start=frame_start,
                video_path=video_path,
                pair_gap=pair_gap,
                latent_dim=latent_dim,
            ):
                raise ValueError(f"Missing or invalid compact shard: {shard}")
            checked += 1
    expected = sum(int(item["total_episodes"]) for item in manifest["datasets"])
    if checked != expected:
        raise RuntimeError(f"Validated {checked} shards, expected {expected}")
    manifest["completion"] = {
        "status": "complete",
        "episode_shards": checked,
        "worker_shards": args.expected_workers,
    }
    _atomic_json_save(manifest, manifest_path)
    print(json.dumps(manifest["completion"], sort_keys=True))


if __name__ == "__main__":
    main()
