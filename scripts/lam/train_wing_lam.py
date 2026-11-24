#!/usr/bin/env python3
"""Train the WING-LAM trajectory objective from immutable pair caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action import (  # noqa: E402
    WINGLAM,
    WINGLAMLossWeights,
    lam_loss,
    load_pair_supervision,
)
from wing.latent_action.lam_dataset import (  # noqa: E402
    WINGLAMPairsDataset,
    discover_pair_records,
)
from wing.latent_action.source_lam import load_lam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-cache", type=Path, action="append", required=True)
    parser.add_argument("--validation-pair-cache", type=Path, action="append")
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-wing-lam",
        type=Path,
        help="Strict-load model weights from a WING-LAM checkpoint; optimizer/step are reset.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-cached-videos",
        type=int,
        default=128,
        help="Maximum decoded videos retained in the in-memory LRU cache.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="Required when --last-backbone-blocks is positive; should be lower than head LR.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--camera-latent-dim", type=int, default=16)
    parser.add_argument("--interaction-latent-dim", type=int, default=32)
    parser.add_argument("--global-model", choices=("affine", "homography"), default="affine")
    parser.add_argument("--last-backbone-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-seed", type=int, default=20260717)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--deterministic-latents", action="store_true")
    parser.add_argument("--camera-translation-px", type=float)
    parser.add_argument("--camera-rotation-degrees", type=float, default=4.0)
    parser.add_argument("--camera-scale-fraction", type=float, default=0.04)
    parser.add_argument(
        "--camera-padding-mode",
        choices=("zeros", "border", "reflection"),
        default="zeros",
    )
    parser.add_argument(
        "--camera-augmentation-mode",
        choices=("addition", "replacement"),
        default="addition",
    )
    parser.add_argument("--replacement-min-valid-fraction", type=float, default=0.5)
    parser.add_argument(
        "--training-view-mode",
        choices=("addition_pair", "empirical_pair", "natural_empirical_triplet"),
    )
    parser.add_argument("--camera-motion-bank", type=Path)
    parser.add_argument(
        "--motion-sampling-power",
        type=float,
        default=0.0,
        help="0 is uniform; positive values mildly oversample large interaction residuals.",
    )
    parser.add_argument(
        "--motion-normalize-by-pair-cache",
        action="store_true",
        help="Normalize motion weights within each pair cache so cache-level mass follows pair count.",
    )
    parser.add_argument(
        "--validation-cache-name",
        action="append",
        help="Optional name for each --validation-pair-cache, in the same order.",
    )
    parser.add_argument(
        "--validation-samples-per-cache",
        type=int,
        default=0,
        help="If positive, evaluate a frozen evenly spaced subset from every validation cache.",
    )
    parser.add_argument("--track-weight", type=float, default=1.0)
    parser.add_argument("--global-motion-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--cross-compose-weight", type=float, default=1.0)
    parser.add_argument("--interaction-invariance-weight", type=float, default=0.1)
    parser.add_argument("--kl-camera-weight", type=float, default=1e-6)
    parser.add_argument("--kl-interaction-weight", type=float, default=1e-6)
    parser.add_argument("--interaction-variance-weight", type=float, default=0.0)
    parser.add_argument("--interaction-covariance-weight", type=float, default=0.0)
    parser.add_argument("--interaction-std-target", type=float, default=0.1)
    parser.add_argument("--active-residual-alpha", type=float, default=0.0)
    parser.add_argument("--active-residual-epsilon", type=float, default=1e-3)
    parser.add_argument("--active-residual-threshold-scale", type=float, default=2.0)
    parser.add_argument("--active-residual-cap-quantile", type=float, default=0.95)
    return parser.parse_args()


def normalize_weights_by_group(weights: np.ndarray, group_sizes: list[int]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    if sum(group_sizes) != len(weights) or any(size <= 0 for size in group_sizes):
        raise ValueError("group sizes must be positive and cover every weight")
    offset = 0
    for size in group_sizes:
        group = weights[offset : offset + size]
        total = float(group.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("each sampling group must have positive finite weight")
        weights[offset : offset + size] = group * (size / total)
        offset += size
    return weights


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, num=count, dtype=np.int64).tolist()


def resume_sampling_seed(seed: int, completed_step: int) -> int:
    """Return a reproducible, non-repeating sampling stream for a resumed run."""
    if completed_step < 0:
        raise ValueError("completed step must be non-negative")
    return (seed + completed_step * 1_000_003) % (2**63 - 1)


def save_checkpoint(
    path: Path,
    *,
    model: WINGLAM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "camera_latent_dim": model.camera_latent_dim,
            "interaction_latent_dim": model.interaction_latent_dim,
            "global_model": model.global_model,
            "canonical_residual": "inv(W)(p1)-p0",
            "composition": "W(p0 + M_int * r_int)",
            "lam_checkpoint": str(args.lam_checkpoint.resolve()),
            "last_backbone_blocks": args.last_backbone_blocks,
            "learning_rates": {
                "heads": args.learning_rate,
                "backbone": args.resolved_backbone_learning_rate,
            },
            "initialization": {
                "wing_lam_checkpoint": args.initialization_wing_lam_checkpoint,
            },
            "loss_weights": {
                "track": args.track_weight,
                "global_motion": args.global_motion_weight,
                "residual": args.residual_weight,
                "cross_compose": args.cross_compose_weight,
                "interaction_invariance": args.interaction_invariance_weight,
                "kl_camera": args.kl_camera_weight,
                "kl_interaction": args.kl_interaction_weight,
                "interaction_variance": args.interaction_variance_weight,
                "interaction_covariance": args.interaction_covariance_weight,
                "interaction_std_target": args.interaction_std_target,
                "active_residual_alpha": args.active_residual_alpha,
                "active_residual_epsilon": args.active_residual_epsilon,
                "active_residual_threshold_scale": args.active_residual_threshold_scale,
                "active_residual_cap_quantile": args.active_residual_cap_quantile,
                "active_residual_cap": args.resolved_active_residual_cap,
            },
            "camera_augmentation": {
                "mode": args.camera_augmentation_mode,
                "translation_px": args.camera_translation_px,
                "rotation_degrees": args.camera_rotation_degrees,
                "scale_fraction": args.camera_scale_fraction,
                "padding_mode": args.camera_padding_mode,
                "replacement_min_valid_fraction": args.replacement_min_valid_fraction,
                "training_view_mode": args.training_view_mode,
                "camera_motion_bank": (
                    str(args.camera_motion_bank.resolve()) if args.camera_motion_bank is not None else None
                ),
                "assignment_summary": args.camera_assignment_summary,
            },
            "motion_sampling_power": args.motion_sampling_power,
            "sampling": args.sampling_summary,
            "deterministic_latents": args.deterministic_latents,
            "training_seed": args.seed,
            "sampling_seed": args.resolved_sampling_seed,
            "resume_sampling_policy": "deterministic fresh stream keyed by seed and completed step",
            "pair_caches": [str(path.resolve()) for path in args.pair_cache],
            "validation_pair_caches": (
                [str(path.resolve()) for path in args.validation_pair_cache]
                if args.validation_pair_cache
                else []
            ),
            "validation_cache_names": args.resolved_validation_cache_names,
            "validation_samples_per_cache": args.validation_samples_per_cache,
        },
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def emit_metrics(metrics: dict[str, float | int | str], output: Path | None) -> None:
    line = json.dumps(metrics, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as handle:
            handle.write(line + "\n")


def evaluate(
    *,
    model: WINGLAM,
    loader: DataLoader,
    device: torch.device,
    loss_weights: WINGLAMLossWeights,
    deterministic_latents: bool,
    compute_cross: bool,
    max_batches: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(seed)
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch_size = int(batch["points0"].shape[0])
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                outputs = model(
                    batch,
                    sample_latents=not deterministic_latents,
                    compute_cross=compute_cross,
                )
                loss, components = lam_loss(batch, outputs, loss_weights)
            values = {"loss": float(loss), **{name: float(value) for name, value in components.items()}}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            samples += batch_size
    model.train()
    if samples == 0:
        raise RuntimeError("validation loader produced no samples")
    return {f"val_{name}": value / samples for name, value in totals.items()} | {
        "val_samples": float(samples)
    }


def main() -> int:
    args = parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.log_every <= 0
        or args.max_cached_videos <= 0
        or args.validation_batches <= 0
        or args.validation_samples_per_cache < 0
    ):
        raise ValueError(
            "steps, batch-size, log-every, validation-batches and max-cached-videos must be positive"
        )
    if (args.validation_pair_cache is None) != (args.validate_every == 0):
        raise ValueError(
            "--validation-pair-cache and positive --validate-every must be provided together"
        )
    if args.validation_cache_name is not None:
        if args.validation_pair_cache is None or len(args.validation_cache_name) != len(
            args.validation_pair_cache
        ):
            raise ValueError(
                "--validation-cache-name must be provided once per validation pair cache"
            )
        if len(set(args.validation_cache_name)) != len(args.validation_cache_name):
            raise ValueError("validation cache names must be unique")
        if any(not name.replace("_", "").isalnum() for name in args.validation_cache_name):
            raise ValueError("validation cache names must contain only letters, digits, or underscores")
    args.resolved_validation_cache_names = args.validation_cache_name or (
        [""] if args.validation_pair_cache and len(args.validation_pair_cache) == 1 else [
            f"source_{index}" for index in range(len(args.validation_pair_cache or []))
        ]
    )
    if args.resume is not None and args.init_wing_lam is not None:
        raise ValueError("--resume and --init-wing-lam are mutually exclusive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.last_backbone_blocks > 0:
        if args.backbone_learning_rate is None or args.backbone_learning_rate <= 0:
            raise ValueError(
                "positive --backbone-learning-rate is required when unfreezing backbone blocks"
            )
        if args.backbone_learning_rate >= args.learning_rate:
            raise ValueError("backbone learning rate must be lower than the WING-LAM head learning rate")
        args.resolved_backbone_learning_rate = args.backbone_learning_rate
    else:
        if args.backbone_learning_rate is not None:
            raise ValueError("--backbone-learning-rate requires --last-backbone-blocks > 0")
        args.resolved_backbone_learning_rate = None
    loss_values = (
        args.track_weight,
        args.global_motion_weight,
        args.residual_weight,
        args.cross_compose_weight,
        args.interaction_invariance_weight,
        args.kl_camera_weight,
        args.kl_interaction_weight,
        args.interaction_variance_weight,
        args.interaction_covariance_weight,
    )
    if any(value < 0 for value in loss_values) or not any(value > 0 for value in loss_values):
        raise ValueError("loss weights must be non-negative and at least one must be positive")
    if (
        args.motion_sampling_power < 0
        or args.camera_rotation_degrees < 0
        or args.camera_scale_fraction < 0
        or args.interaction_std_target <= 0
        or not 0.0 <= args.replacement_min_valid_fraction <= 1.0
        or args.active_residual_alpha < 0
        or args.active_residual_epsilon <= 0
        or args.active_residual_threshold_scale < 0
        or not 0.0 < args.active_residual_cap_quantile <= 1.0
        or (args.camera_translation_px is not None and args.camera_translation_px < 0)
    ):
        raise ValueError("augmentation, sampling, and spread parameters must be non-negative")
    if not args.lam_checkpoint.is_file():
        raise FileNotFoundError(args.lam_checkpoint)
    if args.init_wing_lam is not None and not args.init_wing_lam.is_file():
        raise FileNotFoundError(args.init_wing_lam)
    empirical_modes = ("empirical_pair", "natural_empirical_triplet")
    if args.training_view_mode in empirical_modes:
        if args.camera_motion_bank is None or not args.camera_motion_bank.is_file():
            raise FileNotFoundError(args.camera_motion_bank or "--camera-motion-bank is required")
    elif args.camera_motion_bank is not None:
        raise ValueError("--camera-motion-bank is only valid for empirical training-view modes")
    args.initialization_wing_lam_checkpoint = (
        str(args.init_wing_lam.resolve()) if args.init_wing_lam is not None else None
    )
    resume_checkpoint = None
    start_step = 0
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        start_step = int(resume_checkpoint["step"])
        previous_seed = resume_checkpoint["metadata"].get("training_seed")
        if previous_seed is not None and previous_seed != args.seed:
            raise ValueError(
                f"resume training seed {previous_seed} does not match requested {args.seed}"
            )
    args.resolved_sampling_seed = resume_sampling_seed(args.seed, start_step)
    torch.manual_seed(args.resolved_sampling_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = WINGLAMPairsDataset(
        args.pair_cache,
        augment_camera=True,
        max_cached_videos=args.max_cached_videos,
        camera_translation_pixels=args.camera_translation_px,
        camera_rotation_degrees=args.camera_rotation_degrees,
        camera_scale_fraction=args.camera_scale_fraction,
        camera_padding_mode=args.camera_padding_mode,
        camera_augmentation_mode=args.camera_augmentation_mode,
        replacement_min_valid_fraction=args.replacement_min_valid_fraction,
        training_view_mode=args.training_view_mode,
        camera_motion_bank=args.camera_motion_bank,
        view_assignment_seed=args.seed,
    )
    args.camera_assignment_summary = dataset.assignment_summary
    pair_cache_sizes = [len(discover_pair_records([path])) for path in args.pair_cache]
    if sum(pair_cache_sizes) != len(dataset):
        raise AssertionError("pair cache sizes do not match the combined dataset")
    sampler = None
    motion = []
    active_magnitudes = []
    empty_interaction_pairs = 0
    if args.motion_sampling_power > 0 or args.active_residual_alpha > 0:
        for path in dataset.records:
            record = load_pair_supervision(path)
            active = record.valid & record.interaction
            magnitude = np.linalg.norm(record.canonical_residual[active], axis=1)
            if not len(magnitude):
                empty_interaction_pairs += 1
                motion.append(0.0)
                continue
            motion.append(float(magnitude.mean()))
            active_magnitudes.append(magnitude)
    if args.active_residual_alpha > 0:
        if not active_magnitudes:
            raise ValueError("no pair contains active interaction tracks")
        args.resolved_active_residual_cap = max(
            float(np.quantile(np.concatenate(active_magnitudes), args.active_residual_cap_quantile)),
            args.active_residual_epsilon,
        )
    else:
        args.resolved_active_residual_cap = 0.05
    if args.motion_sampling_power > 0:
        motion = np.asarray(motion, dtype=np.float64)
        positive_motion = motion[motion > 0]
        if not len(positive_motion):
            raise ValueError("motion sampling requires at least one pair with active interaction")
        median = max(float(np.median(positive_motion)), 1e-8)
        sample_weights = np.clip(motion / median, 0.25, 4.0) ** args.motion_sampling_power
        if args.motion_normalize_by_pair_cache:
            sample_weights = normalize_weights_by_group(sample_weights, pair_cache_sizes)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.resolved_sampling_seed),
        )
    elif args.motion_normalize_by_pair_cache:
        raise ValueError("--motion-normalize-by-pair-cache requires positive motion sampling power")
    args.sampling_summary = {
        "pair_cache_sizes": pair_cache_sizes,
        "motion_normalize_by_pair_cache": args.motion_normalize_by_pair_cache,
        "pair_cache_weight_mass": (
            [
                float(sample_weights[sum(pair_cache_sizes[:index]) : sum(pair_cache_sizes[: index + 1])].sum())
                for index in range(len(pair_cache_sizes))
            ]
            if args.motion_sampling_power > 0
            else [float(size) for size in pair_cache_sizes]
        ),
    }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(args.resolved_sampling_seed),
    )
    validation_loaders: list[tuple[str, DataLoader]] = []
    if args.validation_pair_cache:
        for name, cache in zip(
            args.resolved_validation_cache_names,
            args.validation_pair_cache,
            strict=True,
        ):
            validation_dataset = WINGLAMPairsDataset(
                [cache],
                augment_camera=True,
                max_cached_videos=args.max_cached_videos,
                camera_translation_pixels=args.camera_translation_px,
                camera_rotation_degrees=args.camera_rotation_degrees,
                camera_scale_fraction=args.camera_scale_fraction,
                camera_padding_mode=args.camera_padding_mode,
                camera_augmentation_mode=args.camera_augmentation_mode,
                replacement_min_valid_fraction=args.replacement_min_valid_fraction,
                training_view_mode=args.training_view_mode,
                camera_motion_bank=args.camera_motion_bank,
                view_assignment_seed=args.validation_seed,
            )
            selected_dataset = (
                Subset(
                    validation_dataset,
                    evenly_spaced_indices(
                        len(validation_dataset), args.validation_samples_per_cache
                    ),
                )
                if args.validation_samples_per_cache > 0
                else validation_dataset
            )
            validation_loaders.append(
                (
                    name,
                    DataLoader(
                        selected_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=device.type == "cuda",
                        drop_last=False,
                    ),
                )
            )
    lam = load_lam(str(args.lam_checkpoint), device="cpu")
    model = WINGLAM(
        lam,
        camera_latent_dim=args.camera_latent_dim,
        interaction_latent_dim=args.interaction_latent_dim,
        global_model=args.global_model,
    )
    del lam
    if args.init_wing_lam is not None:
        initialization = torch.load(
            args.init_wing_lam,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        metadata = initialization["metadata"]
        expected = (model.camera_latent_dim, model.interaction_latent_dim, model.global_model)
        actual = (
            metadata["camera_latent_dim"],
            metadata["interaction_latent_dim"],
            metadata["global_model"],
        )
        if actual != expected:
            raise ValueError(f"initialization architecture {actual} does not match requested {expected}")
        model.load_state_dict(initialization["model"], strict=True)
        del initialization
    model.set_backbone_trainable(last_blocks=args.last_backbone_blocks)
    model.to(device)
    backbone_parameters = [
        parameter for parameter in model.visual_encoder.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not head_parameters:
        raise RuntimeError("WING-LAM has no trainable head parameters")
    parameter_groups = [{"params": head_parameters, "lr": args.learning_rate}]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.resolved_backbone_learning_rate}
        )
    trainable = [*head_parameters, *backbone_parameters]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    if args.resume is not None:
        assert resume_checkpoint is not None
        checkpoint = resume_checkpoint
        metadata = checkpoint["metadata"]
        expected = (model.camera_latent_dim, model.interaction_latent_dim, model.global_model)
        actual = (
            metadata["camera_latent_dim"],
            metadata["interaction_latent_dim"],
            metadata["global_model"],
        )
        if actual != expected:
            raise ValueError(f"resume architecture {actual} does not match requested {expected}")
        if metadata.get("last_backbone_blocks") != args.last_backbone_blocks:
            raise ValueError(
                "resume checkpoint last_backbone_blocks does not match the requested training stage"
            )
        previous_learning_rates = metadata.get("learning_rates")
        requested_learning_rates = {
            "heads": args.learning_rate,
            "backbone": args.resolved_backbone_learning_rate,
        }
        if previous_learning_rates is not None and previous_learning_rates != requested_learning_rates:
            raise ValueError(
                f"resume learning rates {previous_learning_rates} do not match "
                f"requested {requested_learning_rates}"
            )
        previous_initialization = metadata.get("initialization", {})
        args.initialization_wing_lam_checkpoint = previous_initialization.get(
            "wing_lam_checkpoint"
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        del checkpoint, resume_checkpoint

    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    loss_weights = WINGLAMLossWeights(
        track=args.track_weight,
        global_motion=args.global_motion_weight,
        residual=args.residual_weight,
        cross_compose=args.cross_compose_weight,
        interaction_invariance=args.interaction_invariance_weight,
        kl_camera=args.kl_camera_weight,
        kl_interaction=args.kl_interaction_weight,
        interaction_variance=args.interaction_variance_weight,
        interaction_covariance=args.interaction_covariance_weight,
        interaction_std_target=args.interaction_std_target,
        active_residual_alpha=args.active_residual_alpha,
        active_residual_epsilon=args.active_residual_epsilon,
        active_residual_threshold_scale=args.active_residual_threshold_scale,
        active_residual_cap=args.resolved_active_residual_cap,
    )
    model.train()
    iterator = iter(loader)
    if args.metrics_output is not None and args.metrics_output.exists() and start_step == 0:
        raise FileExistsError(
            f"metrics output already exists for a fresh run: {args.metrics_output}"
        )
    emit_metrics(
        {
            "event": "setup",
            "step": start_step,
            "train_pairs": len(dataset),
            "validation_pairs": sum(len(loader.dataset) for _, loader in validation_loaders),
            "validation_sources": json.dumps(
                {name or "combined": len(loader.dataset) for name, loader in validation_loaders},
                sort_keys=True,
            ),
            "sampling_summary": json.dumps(args.sampling_summary, sort_keys=True),
            "sampling_seed": args.resolved_sampling_seed,
            "empty_interaction_pairs": empty_interaction_pairs,
            "empty_interaction_fraction": empty_interaction_pairs / len(dataset),
            "head_learning_rate": args.learning_rate,
            "backbone_learning_rate": args.resolved_backbone_learning_rate or 0.0,
            "trainable_head_parameters": sum(parameter.numel() for parameter in head_parameters),
            "trainable_backbone_parameters": sum(
                parameter.numel() for parameter in backbone_parameters
            ),
        },
        args.metrics_output,
    )
    log_started = time.monotonic()
    log_start_step = start_step
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            outputs = model(
                batch,
                sample_latents=not args.deterministic_latents,
                compute_cross=args.cross_compose_weight > 0,
            )
            loss, components = lam_loss(batch, outputs, loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.monotonic()
            metrics = {name: float(value.detach()) for name, value in components.items()}
            metrics.update(
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "samples_per_second": (
                        (step - log_start_step) * args.batch_size / max(now - log_started, 1e-9)
                    ),
                }
            )
            emit_metrics(metrics, args.metrics_output)
            log_started = now
            log_start_step = step
        if validation_loaders and step % args.validate_every == 0:
            validation_started = time.monotonic()
            source_metrics = []
            for source_index, (name, validation_loader) in enumerate(validation_loaders):
                current = evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    loss_weights=loss_weights,
                    deterministic_latents=args.deterministic_latents,
                    compute_cross=args.cross_compose_weight > 0,
                    max_batches=args.validation_batches,
                    seed=args.validation_seed + source_index,
                )
                source_metrics.append(current)
            if len(source_metrics) == 1 and not validation_loaders[0][0]:
                metrics = source_metrics[0]
            else:
                metrics = {}
                for (name, _), current in zip(validation_loaders, source_metrics, strict=True):
                    for key, value in current.items():
                        suffix = key.removeprefix("val_")
                        metrics[f"val_{name}_{suffix}"] = value
                value_keys = [key for key in source_metrics[0] if key != "val_samples"]
                for key in value_keys:
                    suffix = key.removeprefix("val_")
                    metrics[f"val_combined_{suffix}"] = float(
                        np.mean([current[key] for current in source_metrics])
                    )
                metrics["val_samples"] = float(
                    sum(current["val_samples"] for current in source_metrics)
                )
            metrics.update(
                {
                    "event": "validation",
                    "step": step,
                    "validation_seconds": time.monotonic() - validation_started,
                }
            )
            emit_metrics(metrics, args.metrics_output)
            log_started = time.monotonic()
            log_start_step = step
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            intermediate = args.output.with_name(f"{args.output.stem}_step{step:08d}{args.output.suffix}")
            checkpoint_started = time.monotonic()
            save_checkpoint(
                intermediate,
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
            )
            emit_metrics(
                {
                    "event": "checkpoint",
                    "step": step,
                    "checkpoint_seconds": time.monotonic() - checkpoint_started,
                },
                args.metrics_output,
            )
            log_started = time.monotonic()
            log_start_step = step
    save_checkpoint(
        args.output,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
