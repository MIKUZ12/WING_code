#!/usr/bin/env python3
"""Export frozen trajectory-assisted WING-LAM ``z_int`` student targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action import (  # noqa: E402
    WINGLAM,
    decode_rgb_video,
)
from wing.latent_action.lam_dataset import WINGLAMPairsDataset  # noqa: E402
from wing.latent_action.source_lam import load_lam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-list", type=Path, required=True)
    parser.add_argument("--wing-lam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (args.pair_list, args.wing_lam_checkpoint, args.lam_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    payload = torch.load(
        args.wing_lam_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    metadata = payload["metadata"]
    teacher_step = int(payload.get("step", 0))
    lam = load_lam(str(args.lam_checkpoint), device="cpu")
    model = WINGLAM(
        lam,
        camera_latent_dim=int(metadata["camera_latent_dim"]),
        interaction_latent_dim=int(metadata["interaction_latent_dim"]),
        global_model=str(metadata["global_model"]),
    )
    if model.interaction_latent_dim != 32:
        raise ValueError("RGB student requires a 32-D teacher interaction latent")
    model.load_state_dict(payload["model"], strict=True)
    del payload, lam
    model = model.to(device).eval().requires_grad_(False)

    # Same pixels as WINGLAMPairsDataset, but resize short clips in batches.
    import wing.latent_action.lam_dataset as dataset_module

    dataset_module._decode_video = decode_rgb_video
    dataset = WINGLAMPairsDataset(
        [args.pair_list],
        augment_camera=False,
        max_cached_videos=64,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    latents: list[np.ndarray] = []
    available: list[np.ndarray] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                factors = model.encode_factors(
                    videos=batch["videos"],
                    points0=batch["points0"],
                    points1=batch["points1"],
                    canonical_residual=batch["canonical_residual"],
                    valid=batch["valid"],
                    background=batch["background"],
                    interaction=batch["interaction"],
                    visibility=batch["visibility"],
                    confidence=batch["confidence"],
                    sample_latents=False,
                )
            latents.append(factors["mu_interaction"].float().cpu().numpy())
            available.append(factors["interaction_available"].cpu().numpy())
            if batch_index == 1 or batch_index % 250 == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "batches": len(loader),
                            "pairs_per_second": (
                                min(batch_index * args.batch_size, len(dataset))
                                / max(elapsed, 1e-12)
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    rows = []
    for path in dataset.records:
        with np.load(path, allow_pickle=False) as archive:
            rows.append(json.loads(str(archive["metadata_json"].item())))
    z_int = np.concatenate(latents).astype(np.float32)
    interaction_available = np.concatenate(available).astype(bool)
    if z_int.shape != (len(rows), 32) or not np.isfinite(z_int).all():
        raise RuntimeError(f"teacher produced invalid z_int shape {z_int.shape}")
    provenance = {
        "pair_list": str(args.pair_list.resolve()),
        "wing_lam_checkpoint": str(args.wing_lam_checkpoint.resolve()),
        "lam_checkpoint": str(args.lam_checkpoint.resolve()),
        "teacher_step": teacher_step,
        "student_forward_dependency": "RGB frame pair only; teacher geometry is offline supervision",
    }
    arrays = {
        "z_int": z_int,
        "interaction_available": interaction_available,
        "video": np.asarray([str(Path(row["video"]).resolve()) for row in rows]),
        "frame_start": np.asarray([int(row["frame_start"]) for row in rows], dtype=np.int64),
        "frame_end": np.asarray([int(row["frame_end"]) for row in rows], dtype=np.int64),
        "gap": np.asarray([int(row["gap"]) for row in rows], dtype=np.int64),
        "split": np.full(len(rows), args.split),
        "metadata_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": len(rows),
                "interaction_available": int(interaction_available.sum()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
