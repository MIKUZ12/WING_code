#!/usr/bin/env python3
"""Build a WING-LAM RGB-student index from pair lists and teacher targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action.rgb_student_data import (  # noqa: E402
    PairKey,
    read_pair_list,
    source_video_id,
    validate_target_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="DATASET::PAIR_LIST::TARGET_NPZ (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_rows: list[dict[str, object]] = []
    pair_ids: set[str] = set()
    keys: set[tuple[str, PairKey]] = set()
    for spec in args.source:
        parts = spec.split("::")
        if len(parts) != 3 or not all(parts):
            raise ValueError("source must be DATASET::PAIR_LIST::TARGET_NPZ")
        dataset, pair_list_raw, target_raw = parts
        pair_list = Path(pair_list_raw).resolve()
        target = Path(target_raw).resolve()
        paths = read_pair_list(pair_list)
        validate_target_archive(target, pair_list=pair_list)
        with np.load(target, allow_pickle=False) as data:
            length = len(data["z_int"])
            if length != len(paths):
                raise ValueError(f"target/list length mismatch: {length} != {len(paths)}")
            videos = data["video"].astype(str)
            starts = data["frame_start"].astype(int)
            ends = data["frame_end"].astype(int)
            gaps = data["gap"].astype(int)
            splits = data["split"].astype(str)
        for target_index, values in enumerate(
            zip(paths, videos, starts, ends, gaps, splits, strict=True)
        ):
            pair_cache, video, start, end, gap, split = values
            key = PairKey.create(video, start, end, gap)
            scoped_key = (dataset, key)
            if scoped_key in keys:
                raise ValueError(f"duplicate pair key in dataset {dataset}: {key}")
            keys.add(scoped_key)
            pair_id = f"{dataset}-{key.pair_id}"
            if pair_id in pair_ids:
                raise ValueError(f"pair id collision: {pair_id}")
            pair_ids.add(pair_id)
            output_rows.append(
                {
                    "pair_id": pair_id,
                    "dataset": dataset,
                    "source_video_id": source_video_id(pair_cache, key),
                    "split": split,
                    "video": key.video,
                    "frame_start": key.frame_start,
                    "frame_end": key.frame_end,
                    "gap": key.gap,
                    "pair_cache": str(pair_cache),
                    "target_npz": str(target),
                    "target_index": target_index,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(args.output), "pairs": len(output_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
