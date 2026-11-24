#!/usr/bin/env python3
"""Fit DCT normalization statistics for a generic latent action cache.

The legacy dense robot-cache path keeps its deterministic episode-modulus
split.  Compact robot caches may explicitly select all valid anchors, while
compact Ego caches must use the authoritative Stage-I logical index, whose
source-video-group split prevents train/validation leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wing.datasets.ego_stage1_index import EgoStage1LogicalIndex


def orthonormal_dct(size: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError("size must be positive")
    time = torch.arange(size, dtype=torch.float64)
    freq = torch.arange(size, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(torch.pi / size * (time + 0.5) * freq) * (2.0 / size) ** 0.5
    matrix[0] = size ** -0.5
    return matrix


@dataclass
class _StreamingMoments:
    """Numerically stable parallel-Welford population moments."""

    mean: torch.Tensor
    m2: torch.Tensor
    count: int = 0

    @classmethod
    def zeros(cls, frequencies: int, target_dim: int) -> "_StreamingMoments":
        mean = torch.zeros(frequencies, target_dim, dtype=torch.float64)
        return cls(mean=mean, m2=torch.zeros_like(mean))

    def update(self, values: torch.Tensor) -> int:
        if values.ndim != 3 or values.shape[1:] != self.mean.shape:
            raise ValueError(
                f"Moment input must have shape [N,{self.mean.shape[0]},{self.mean.shape[1]}], "
                f"got {tuple(values.shape)}."
            )
        batch_count = int(values.shape[0])
        if batch_count == 0:
            return 0
        batch_mean = values.mean(0)
        batch_m2 = (values - batch_mean).square().sum(0)
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
        else:
            combined_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean.add_(delta * (batch_count / combined_count))
            self.m2.add_(
                batch_m2
                + delta.square() * (self.count * batch_count / combined_count)
            )
        self.count += batch_count
        return batch_count


def _update_moments(values: torch.Tensor, moments: _StreamingMoments) -> int:
    if values.ndim != 3 or values.shape[1:] != moments.mean.shape:
        raise ValueError(
            f"Moment input must have shape [N,{moments.mean.shape[0]},{moments.mean.shape[1]}], "
            f"got {tuple(values.shape)}."
        )
    return moments.update(values)


def _fit_dense(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    matrix: torch.Tensor,
    moments: _StreamingMoments,
) -> tuple[int, dict[str, Any]]:
    available_indices = sorted(int(item["dataset_index"]) for item in manifest["datasets"])
    dataset_indices = available_indices if args.dataset_index is None else sorted(set(args.dataset_index))
    unknown_indices = sorted(set(dataset_indices) - set(available_indices))
    if unknown_indices:
        raise ValueError(
            f"Unknown dataset indices {unknown_indices}; available indices are {available_indices}"
        )
    paths: list[Path] = []
    for dataset_index in dataset_indices:
        paths.extend(sorted((args.cache_root / f"dataset_{dataset_index:02d}").glob("episode_*.pt")))
    if not paths:
        raise FileNotFoundError("No latent-action episode shards found")

    count = 0
    for path in paths:
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        heldout = episode_index % args.holdout_mod == args.holdout_remainder
        if args.episode_split != "all" and (args.episode_split == "heldout") != heldout:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        target = torch.as_tensor(
            payload[manifest.get("target", {}).get("key", "latent_target")],
            dtype=torch.float32,
        )
        mask = torch.as_tensor(
            payload[manifest.get("target", {}).get("mask_key", "latent_target_mask")],
            dtype=torch.bool,
        )
        if (
            target.ndim != 3
            or target.shape[1:] != (args.horizon, args.target_dim)
            or mask.shape != target.shape[:2]
        ):
            raise ValueError(f"Invalid cache shard contract: {path}")
        valid = mask.all(dim=1)
        if not valid.any():
            continue
        coeff = torch.einsum("kh,nhd->nkd", matrix, target[valid].double())[
            :, : args.low_frequencies
        ]
        count += _update_moments(coeff, moments)
    metadata = {
        "episode_split": args.episode_split,
        "holdout_mod": args.holdout_mod,
        "holdout_remainder": args.holdout_remainder,
        "dataset_indices": dataset_indices,
    }
    return count, metadata


def _validate_compact_shard(
    path: Path,
    *,
    identity: Any,
    target_dim: int,
    pair_gap: int,
    target_key: str = "latent_pair_target",
    mask_key: str = "latent_pair_valid",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing compact latent action episode shard: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    target = torch.as_tensor(payload[target_key], dtype=torch.float64)
    valid = torch.as_tensor(payload[mask_key], dtype=torch.bool)
    if target.shape != (identity.episode_length, target_dim):
        raise ValueError(f"Unexpected latent_pair_target shape in {path}: {tuple(target.shape)}")
    if valid.shape != (identity.episode_length,):
        raise ValueError(f"Unexpected latent_pair_valid shape in {path}: {tuple(valid.shape)}")
    expected_valid = torch.arange(identity.episode_length) + pair_gap < identity.episode_length
    if not torch.equal(valid, expected_valid):
        raise ValueError(f"Compact latent action pair-valid mask violates fixed-gap tail semantics: {path}")
    if not torch.isfinite(target).all():
        raise ValueError(f"Non-finite latent action target in {path}")
    source = payload.get("source", {})
    required = (
        "dataset_root",
        "dataset_index",
        "episode_index",
        "episode_length",
        "frame_start",
        "video_path",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"Compact Stage-I shard lacks source identity {missing}: {path}")
    if (
        os.path.realpath(source["dataset_root"]) != identity.dataset_root
        or int(source["dataset_index"]) != identity.dataset_index
        or int(source["episode_index"]) != identity.episode_index
        or int(source["episode_length"]) != identity.episode_length
        or int(source["frame_start"]) != identity.frame_start
        or os.path.realpath(source["video_path"]) != identity.source_video
    ):
        raise ValueError(f"Compact shard/source-split identity mismatch: {path}")
    return target, valid


def _fit_compact_all(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    matrix: torch.Tensor,
    moments: _StreamingMoments,
    *,
    pair_gap: int,
    cache_offsets: tuple[int, ...],
) -> tuple[int, dict[str, Any]]:
    if args.episode_split != "all":
        raise ValueError(
            "Compact caches without --logical-index-manifest require "
            "--episode-split all; episode-modulus splits are not valid for "
            "source-grouped Ego data."
        )
    datasets = {
        int(item["dataset_index"]): item for item in manifest.get("datasets", [])
    }
    available_indices = sorted(datasets)
    dataset_indices = (
        available_indices
        if args.dataset_index is None
        else sorted(set(args.dataset_index))
    )
    unknown_indices = sorted(set(dataset_indices) - set(available_indices))
    if unknown_indices:
        raise ValueError(
            f"Unknown dataset indices {unknown_indices}; available indices are "
            f"{available_indices}"
        )

    shards: list[tuple[int, Path]] = []
    for dataset_index in dataset_indices:
        shard_dir = args.cache_root / f"dataset_{dataset_index:02d}"
        shards.extend(
            (dataset_index, path) for path in sorted(shard_dir.glob("episode_*.pt"))
        )
    if not shards:
        raise FileNotFoundError("No compact latent action episode shards found")

    gather_offsets = torch.tensor(cache_offsets[:-1], dtype=torch.long)
    window_span = int(cache_offsets[-1])
    count = 0
    for dataset_index, path in shards:
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pair_target = torch.as_tensor(
            payload[manifest.get("target", {}).get("key", "latent_pair_target")],
            dtype=torch.float64,
        )
        pair_valid = torch.as_tensor(
            payload[manifest.get("target", {}).get("mask_key", "latent_pair_valid")],
            dtype=torch.bool,
        )
        source = payload.get("source", {})
        required = (
            "dataset_root",
            "dataset_index",
            "episode_index",
            "episode_length",
            "video_path",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Compact shard lacks source identity {missing}: {path}")
        episode_length = int(source["episode_length"])
        if pair_target.shape != (episode_length, args.target_dim):
            raise ValueError(
                f"Unexpected latent_pair_target shape in {path}: {tuple(pair_target.shape)}"
            )
        if pair_valid.shape != (episode_length,):
            raise ValueError(
                f"Unexpected latent_pair_valid shape in {path}: {tuple(pair_valid.shape)}"
            )
        expected_valid = torch.arange(episode_length) + pair_gap < episode_length
        if not torch.equal(pair_valid, expected_valid):
            raise ValueError(
                f"Compact latent action pair-valid mask violates fixed-gap tail semantics: {path}"
            )
        if not torch.isfinite(pair_target).all():
            raise ValueError(f"Non-finite latent action target in {path}")

        dataset = datasets[dataset_index]
        expected_video = Path(dataset["root"]) / str(dataset["video_path_template"]).format(
            episode_chunk=episode_index // int(dataset["chunks_size"]),
            video_key=dataset["camera_key"],
            episode_index=episode_index,
        )
        if (
            os.path.realpath(source["dataset_root"])
            != os.path.realpath(dataset["root"])
            or int(source["dataset_index"]) != dataset_index
            or int(source["episode_index"]) != episode_index
            or os.path.realpath(source["video_path"])
            != os.path.realpath(expected_video)
        ):
            raise ValueError(f"Compact shard/source identity mismatch: {path}")

        num_starts = episode_length - window_span
        if num_starts <= 0:
            continue
        starts = torch.arange(num_starts, dtype=torch.long)
        gather = starts[:, None] + gather_offsets[None, :]
        if not bool(pair_valid[gather].all()):
            raise ValueError(f"All-anchor selection includes an invalid compact pair: {path}")
        trajectories = pair_target[gather]
        coeff = torch.einsum("kh,nhd->nkd", matrix, trajectories)[
            :, : args.low_frequencies
        ]
        count += _update_moments(coeff, moments)

    metadata = {
        "episode_split": "all",
        "dataset_indices": dataset_indices,
        "frame_offsets": list(cache_offsets),
        "target_frame_offsets": list(cache_offsets[:-1]),
        "selection_policy": "all_valid_compact_anchors",
    }
    return count, metadata


def _fit_compact(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    matrix: torch.Tensor,
    moments: _StreamingMoments,
) -> tuple[int, dict[str, Any]]:
    target_meta = manifest.get("target", {})
    if int(target_meta.get("horizon", -1)) != args.horizon:
        raise ValueError("Compact cache horizon differs from --horizon.")
    if int(target_meta.get("dimension", -1)) != args.target_dim:
        raise ValueError("Compact cache dimension differs from --target-dim.")
    temporal = manifest.get("temporal", {})
    cache_offsets = tuple(int(value) for value in temporal.get("frame_offsets", []))
    pair_gap = int(temporal.get("pair_gap_frames", -1))
    if len(cache_offsets) != args.horizon + 1 or cache_offsets != tuple(
        range(0, pair_gap * (args.horizon + 1), pair_gap)
    ):
        raise ValueError("Compact cache does not declare H+1 uniform fixed-gap frame offsets.")

    if args.logical_index_manifest is None:
        return _fit_compact_all(
            args,
            manifest,
            matrix,
            moments,
            pair_gap=pair_gap,
            cache_offsets=cache_offsets,
        )
    if args.dataset_index is not None:
        raise ValueError("The Stage-I logical index owns compact-cache dataset selection.")

    dataset_roots = [os.path.realpath(item["root"]) for item in manifest.get("datasets", [])]
    logical = EgoStage1LogicalIndex(
        args.logical_index_manifest,
        split=args.logical_split,
        expected_dataset_roots=dataset_roots,
        expected_anchor_stride=24,
        expected_window_span=48,
    )
    if args.logical_split != "train":
        raise ValueError("Normalization statistics must use logical split=train.")
    if pair_gap != 6 or logical.frame_offsets != tuple(range(0, 49, 6)):
        raise ValueError("Ego K4 statistics require gap=6 and frame offsets [0,6,...,48].")
    if logical.frame_offsets != cache_offsets:
        raise ValueError("Logical-index and compact-cache frame offsets differ.")

    count = 0
    position = 0
    while position < len(logical):
        dataset_index = int(logical.dataset_indices[position])
        episode_index = int(logical.episode_indices[position])
        end = position + 1
        while (
            end < len(logical)
            and int(logical.dataset_indices[end]) == dataset_index
            and int(logical.episode_indices[end]) == episode_index
        ):
            end += 1
        identity = logical.episode_identity(dataset_index, episode_index)
        path = (
            args.cache_root
            / f"dataset_{dataset_index:02d}"
            / f"episode_{episode_index:06d}.pt"
        )
        pair_target, pair_valid = _validate_compact_shard(
            path,
            identity=identity,
            target_dim=args.target_dim,
            pair_gap=pair_gap,
            target_key=target_meta.get("key", "latent_pair_target"),
            mask_key=target_meta.get("mask_key", "latent_pair_valid"),
        )
        starts = logical.frame_indices[position:end]
        gather_offsets = torch.tensor(cache_offsets[:-1], dtype=torch.long)
        gather = starts[:, None] + gather_offsets[None, :]
        if int(gather.max()) >= identity.episode_length:
            raise ValueError(f"Logical anchor exceeds compact episode bounds: {path}")
        if not bool(pair_valid[gather].all()):
            raise ValueError(f"Logical anchor selected an invalid compact pair: {path}")
        trajectories = pair_target[gather]
        coeff = torch.einsum("kh,nhd->nkd", matrix, trajectories)[
            :, : args.low_frequencies
        ]
        count += _update_moments(coeff, moments)
        position = end
    if count != len(logical):
        raise RuntimeError(f"Processed {count} compact anchors, expected {len(logical)}.")
    metadata = {
        "selection_manifest": str(logical.manifest_path),
        "logical_split": logical.split,
        "anchor_stride": logical.anchor_stride,
        "window_span": logical.window_span,
        "frame_offsets": list(logical.frame_offsets),
        "target_frame_offsets": list(logical.frame_offsets[:-1]),
        "dataset_indices": list(range(len(dataset_roots))),
    }
    return count, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-index",
        type=int,
        action="append",
        default=None,
        help="Dataset index to include; repeat for multiple datasets. Defaults to all manifest datasets.",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument("--low-frequencies", type=int, default=2)
    parser.add_argument("--episode-split", choices=("all", "train", "heldout"), default="train")
    parser.add_argument("--holdout-mod", type=int, default=10)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument(
        "--logical-index-manifest",
        type=Path,
        default=None,
        help="Authoritative Ego Stage-I source-group split and logical-anchor manifest.",
    )
    parser.add_argument("--logical-split", choices=("train",), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="Allow bounded smoke statistics from a partial cache; never use these for formal training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_root = args.cache_root.expanduser().resolve()
    if args.logical_index_manifest is not None:
        args.logical_index_manifest = args.logical_index_manifest.expanduser().resolve()
    if not 0 < args.low_frequencies <= args.horizon:
        raise ValueError("low-frequencies must be in [1, horizon]")
    if args.holdout_mod <= 1 or not 0 <= args.holdout_remainder < args.holdout_mod:
        raise ValueError("invalid deterministic episode split")
    manifest_path = args.cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing latent-action cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion", {}).get("status") != "complete" and not args.allow_partial_cache:
        raise ValueError(
            "latent-action cache is not marked complete. Finish extraction first, or use "
            "--allow-partial-cache only for a bounded smoke test."
        )
    storage = str(manifest.get("target", {}).get("storage", "dense_episode_windows"))
    if storage not in {"dense_episode_windows", "compact_fixed_gap_pairs"}:
        raise ValueError(f"Unsupported latent action cache storage: {storage!r}.")
    if storage == "dense_episode_windows" and args.logical_index_manifest is not None:
        raise ValueError("--logical-index-manifest is only valid for compact Ego caches.")

    matrix = orthonormal_dct(args.horizon)
    moments = _StreamingMoments.zeros(args.low_frequencies, args.target_dim)
    if storage == "compact_fixed_gap_pairs":
        count, selection_metadata = _fit_compact(args, manifest, matrix, moments)
    else:
        count, selection_metadata = _fit_dense(args, manifest, matrix, moments)
    if count == 0:
        raise RuntimeError("No fully valid target trajectories were selected")
    if count != moments.count:
        raise RuntimeError(f"Moment count mismatch: selected={count}, accumulated={moments.count}.")
    mean = moments.mean
    variance = (moments.m2 / count).clamp_min(1e-12)
    output = {
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "sample_count": count,
        "horizon": args.horizon,
        "target_dim": args.target_dim,
        "low_frequencies": args.low_frequencies,
        "dct": "orthonormal_dct_ii_time_axis",
        "cache_manifest": str(manifest_path),
        "cache_storage": storage,
        "cache_identity": manifest.get("identity"),
        **selection_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"wrote={args.output} valid_samples={count} mean_shape={tuple(mean.shape)}")


if __name__ == "__main__":
    main()
