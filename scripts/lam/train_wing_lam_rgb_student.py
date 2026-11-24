#!/usr/bin/env python3
"""Distill a frozen WING-LAM teacher into a two-frame RGB-only student."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action import (  # noqa: E402
    RGBStudentDataset,
    build_rgb_student,
)
from wing.latent_action.lam_dataset import (  # noqa: E402
    sample_camera_augmentation,
    warp_images,
)


class WindowShuffleSampler(Sampler[int]):
    """Shuffle windows while keeping all pairs from one short clip contiguous."""

    def __init__(self, rows: list[dict[str, object]], generator: torch.Generator) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["video"])].append(index)
        self.groups = list(grouped.values())
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for group_index in torch.randperm(len(self.groups), generator=self.generator).tolist():
            group = self.groups[group_index]
            for pair_index in torch.randperm(len(group), generator=self.generator).tolist():
                yield group[pair_index]

    def __len__(self) -> int:
        return sum(map(len, self.groups))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--init-wing-lam-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cached-videos", type=int, default=128)
    parser.add_argument("--last-backbone-blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--camera-invariance-weight", type=float, default=1.0)
    parser.add_argument("--camera-translation-px", type=float, default=2.0)
    parser.add_argument("--camera-rotation-degrees", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    return parser.parse_args()


def resume_seed(seed: int, completed_step: int) -> int:
    if completed_step < 0:
        raise ValueError("completed step must be non-negative")
    return (seed + completed_step * 1_000_003) % (2**63 - 1)


def target_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape}/{target.shape}")
    return torch.nn.functional.mse_loss(prediction, target)


def main() -> int:
    args = parse_args()
    if min(args.steps, args.batch_size, args.max_cached_videos, args.log_every) <= 0:
        raise ValueError("steps, batch-size, cache size, and log interval must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be non-negative")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0 or min(
        args.camera_invariance_weight,
        args.camera_translation_px,
        args.camera_rotation_degrees,
    ) < 0:
        raise ValueError("regularization and camera-intervention settings must be non-negative")
    if args.resume is None and args.init_wing_lam_checkpoint is None:
        raise ValueError("a fresh run requires --init-wing-lam-checkpoint")
    if args.resume is not None and args.init_wing_lam_checkpoint is not None:
        raise ValueError("--resume and --init-wing-lam-checkpoint are mutually exclusive")
    for path in (args.index, args.lam_checkpoint, args.resume or args.init_wing_lam_checkpoint):
        if path is None or not path.is_file():
            raise FileNotFoundError(path)

    resume_payload = None
    start_step = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        previous = resume_payload["metadata"]
        previous_history = resume_payload.get("history", ())
        start_step = int(
            resume_payload.get(
                "step",
                previous_history[-1]["step"] if previous_history else 0,
            )
        )
        if start_step >= args.steps:
            raise ValueError(f"resume step {start_step} is not below requested final step {args.steps}")
        expected = {
            "last_backbone_blocks": args.last_backbone_blocks,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "weight_decay": args.weight_decay,
            "camera_invariance_weight": args.camera_invariance_weight,
            "camera_translation_px": args.camera_translation_px,
            "camera_rotation_degrees": args.camera_rotation_degrees,
            "seed": args.seed,
        }
        mismatches = {
            key: (previous.get(key), value)
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")

    sampling_seed = resume_seed(args.seed, start_step)
    random.seed(sampling_seed)
    np.random.seed(sampling_seed % (2**32))
    torch.manual_seed(sampling_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = RGBStudentDataset(args.index, max_cached_videos=args.max_cached_videos)
    generator = torch.Generator().manual_seed(sampling_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=WindowShuffleSampler(dataset.rows, generator),
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    if resume_payload is None:
        teacher_checkpoint = args.init_wing_lam_checkpoint.resolve()
        model = build_rgb_student(
            lam_checkpoint=args.lam_checkpoint,
            teacher_checkpoint=teacher_checkpoint,
        )
    else:
        previous = resume_payload["metadata"]
        recorded_teacher = previous.get("init_wing_lam_checkpoint")
        if not recorded_teacher:
            raise ValueError("Resume checkpoint is missing its teacher checkpoint metadata.")
        teacher_checkpoint = Path(recorded_teacher)
        model = build_rgb_student(lam_checkpoint=args.lam_checkpoint)
        model.load_state_dict(resume_payload["model"], strict=True)
    model.backbone.set_trainable(last_blocks=args.last_backbone_blocks)
    model.to(device)
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": args.learning_rate},
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        del resume_payload

    history: list[dict[str, float | int]] = []
    metadata = {
        "student_forward_inputs": ["videos[B,2,H,W,3]"],
        "latent_dim": 32,
        "index": str(args.index.resolve()),
        "lam_checkpoint": str(args.lam_checkpoint.resolve()),
        "init_wing_lam_checkpoint": str(teacher_checkpoint),
        "last_backbone_blocks": args.last_backbone_blocks,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "weight_decay": args.weight_decay,
        "camera_invariance_weight": args.camera_invariance_weight,
        "camera_translation_px": args.camera_translation_px,
        "camera_rotation_degrees": args.camera_rotation_degrees,
        "seed": args.seed,
        "sampling_seed": sampling_seed,
        "resume_sampling_policy": "fresh deterministic stream keyed by seed and completed step",
    }

    def save_checkpoint(path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metadata": metadata,
                    "history": history,
                },
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    last_log_time = started
    last_log_step = start_step
    for step in range(start_step + 1, args.steps + 1):
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            cpu_batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            videos = batch["videos"]
            height, width = videos.shape[2:4]
            matrices = torch.stack(
                [
                    sample_camera_augmentation(
                        translation_pixels=args.camera_translation_px,
                        image_hw=(height, width),
                        rotation_degrees=args.camera_rotation_degrees,
                        scale_fraction=0.0,
                    )
                    for _ in range(len(videos))
                ]
            ).to(device=device, dtype=videos.dtype)
            augmented = videos.clone()
            augmented[:, 1] = warp_images(
                augmented[:, 1], matrices, padding_mode="border"
            )
            natural, intervention = model(torch.cat([videos, augmented], dim=0)).chunk(2)
            target_loss = 0.5 * (
                target_mse(natural, batch["target"])
                + target_mse(intervention, batch["target"])
            )
            invariance_loss = target_mse(natural, intervention)
            loss = target_loss + args.camera_invariance_weight * invariance_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite student loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "target_loss": float(target_loss.detach()),
            "invariance_loss": float(invariance_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        history.append(row)
        if step == start_step + 1 or step % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.monotonic()
            elapsed_steps = step - last_log_step
            logged = row | {
                "elapsed_seconds": now - started,
                "steps_per_second": elapsed_steps / max(now - last_log_time, 1e-12),
                "examples_per_second": (
                    elapsed_steps * args.batch_size / max(now - last_log_time, 1e-12)
                ),
            }
            print(json.dumps(logged, sort_keys=True), flush=True)
            last_log_time, last_log_step = now, step
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            intermediate = args.output.with_name(
                f"{args.output.stem}_step{step:08d}{args.output.suffix}"
            )
            save_checkpoint(intermediate, step)
    save_checkpoint(args.output, args.steps)
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
