#!/usr/bin/env python3
"""Label manifest-defined RGB frame pairs with a frozen WING-LAM RGB student.

The input JSONL must contain ``video``, ``frame_start``, ``frame_end``, and
``gap``.  No AllTracker, EgoHOS, affine, track, or mask input is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action import (  # noqa: E402
    RGBStudentPairKey,
    load_rgb_student_checkpoint,
    preprocess_rgb_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lam-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("input manifest is empty")
    keys = [
        RGBStudentPairKey.create(
            row["video"], row["frame_start"], row["frame_end"], row["gap"]
        )
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("input manifest contains duplicate frame pairs")

    device = torch.device(args.device)
    model, metadata = load_rgb_student_checkpoint(
        args.checkpoint,
        device=device,
        lam_checkpoint=args.lam_checkpoint,
    )
    model.eval().requires_grad_(False)
    latent = np.full((len(keys), 32), np.nan, dtype=np.float32)
    pending: list[torch.Tensor] = []
    pending_indices: list[int] = []

    def flush() -> None:
        if not pending:
            return
        videos = torch.stack(pending).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            values = model(videos).float().cpu().numpy()
        latent[np.asarray(pending_indices)] = values
        pending.clear()
        pending_indices.clear()

    by_video: dict[str, list[int]] = defaultdict(list)
    for row_index, key in enumerate(keys):
        by_video[key.video].append(row_index)
    for video, row_indices in by_video.items():
        by_end: dict[int, list[int]] = defaultdict(list)
        references: Counter[int] = Counter()
        for row_index in row_indices:
            key = keys[row_index]
            by_end[key.frame_end].append(row_index)
            references[key.frame_start] += 1
            references[key.frame_end] += 1
        selected_frames: dict[int, torch.Tensor] = {}
        maximum = max(keys[index].frame_end for index in row_indices)
        decoded_through = -1
        with av.open(video) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index > maximum:
                    break
                decoded_through = frame_index
                if references[frame_index] > 0:
                    selected_frames[frame_index] = preprocess_rgb_frame(
                        frame.to_ndarray(format="rgb24")
                    )
                for row_index in by_end.get(frame_index, ()):
                    key = keys[row_index]
                    pending.append(
                        torch.stack(
                            [selected_frames[key.frame_start], selected_frames[key.frame_end]]
                        )
                    )
                    pending_indices.append(row_index)
                    references[key.frame_start] -= 1
                    references[key.frame_end] -= 1
                    if references[key.frame_start] == 0:
                        selected_frames.pop(key.frame_start, None)
                    if references[key.frame_end] == 0:
                        selected_frames.pop(key.frame_end, None)
                    if len(pending) == args.batch_size:
                        flush()
        if decoded_through < maximum:
            raise IndexError(f"frame {maximum} was not decoded from {video}")
    flush()
    if latent.shape != (len(keys), 32) or not np.isfinite(latent).all():
        raise RuntimeError(f"invalid RGB student output shape {latent.shape}")
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "student_checkpoint": str(args.checkpoint.resolve()),
        "teacher_checkpoint": metadata["init_wing_lam_checkpoint"],
        "forward_inputs": ["two RGB frames"],
        "normalization": "raw student z_int; compute train-only statistics downstream",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                z_int=latent,
                video=np.asarray([key.video for key in keys]),
                frame_start=np.asarray([key.frame_start for key in keys], dtype=np.int64),
                frame_end=np.asarray([key.frame_end for key in keys], dtype=np.int64),
                gap=np.asarray([key.gap for key in keys], dtype=np.int64),
                metadata_json=np.asarray(json.dumps(provenance, sort_keys=True)),
            )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(args.output), "pairs": len(keys)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

