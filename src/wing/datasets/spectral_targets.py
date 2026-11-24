"""Dataset-side contract for predicted latent-action supervision.

The wrapper deliberately leaves WING's action/video sample untouched.  It
only joins an offline sidecar containing an auxiliary ``[H, D]`` latent action target and
its validity mask.  The sidecar identity is episode based, so a dataset can
keep its native action representation and video layout.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .ego_stage1_index import EgoStage1LogicalIndex, EgoStage1EpisodeIdentity
from .lerobot.vae_latent_cache import VaeLatentCache, SparseSourceVaeLatentCache


SPECTRAL_CACHE_SCHEMA_VERSION = 1


def assemble_compact_latent_targets(
    pair_target: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    frame_offsets: Sequence[int],
    pair_gap: int,
    start_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble trajectories from one cached latent per unique fixed-gap pair."""
    if pair_target.ndim != 2:
        raise ValueError("pair_target must have shape [L,D].")
    if pair_valid.shape != pair_target.shape[:1]:
        raise ValueError("pair_valid must have shape [L].")
    gap = int(pair_gap)
    offsets = tuple(int(value) for value in frame_offsets)
    if gap <= 0:
        raise ValueError("pair_gap must be positive.")
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("frame_offsets must begin at zero and contain at least two values.")
    if any(right - left != gap for left, right in zip(offsets, offsets[1:])):
        raise ValueError("frame_offsets must use the cached pair_gap uniformly.")
    length = pair_target.shape[0]
    if length <= 0:
        raise ValueError("pair_target must contain at least one frame.")
    if start_indices is None:
        starts = torch.arange(length, dtype=torch.long, device=pair_target.device)
    else:
        starts = torch.as_tensor(
            start_indices, dtype=torch.long, device=pair_target.device
        ).reshape(-1)
        if bool(((starts < 0) | (starts >= length)).any()):
            raise IndexError("start_indices must lie within the episode.")
    interval_starts = torch.as_tensor(
        offsets[:-1], dtype=torch.long, device=pair_target.device
    )
    indices = starts[:, None] + interval_starts[None, :]
    in_bounds = indices < length
    safe_indices = indices.clamp_max(length - 1)
    target = pair_target[safe_indices]
    valid = in_bounds & pair_valid.to(
        device=pair_target.device, dtype=torch.bool
    )[safe_indices]
    return target, valid


def _realpath(path: str | Path) -> str:
    return os.path.realpath(os.fspath(path))


@dataclass(frozen=True)
class SpectralSampleKey:
    dataset_index: int
    dataset_root: str
    episode_index: int
    frame_index: int
    episode_length: int

    @property
    def shard_relative_path(self) -> Path:
        return Path(f"dataset_{self.dataset_index:02d}") / f"episode_{self.episode_index:06d}.pt"


@dataclass(frozen=True)
class _DatasetLayout:
    dataset_index: int
    dataset_root: str
    global_start: int
    global_end: int
    episode_ids: tuple[int, ...]
    episode_starts: torch.Tensor
    episode_ends: torch.Tensor


class SpectralSampleResolver:
    """Resolve a BaseLerobotDataset global index without dataset-specific IDs."""

    def __init__(self, layouts: Sequence[_DatasetLayout]) -> None:
        if not layouts:
            raise ValueError("At least one dataset layout is required.")
        self.layouts = tuple(layouts)
        self.total_frames = self.layouts[-1].global_end

    @classmethod
    def from_base_lerobot_dataset(cls, base_dataset: Any) -> "SpectralSampleResolver":
        layouts: list[_DatasetLayout] = []
        global_start = 0
        for dataset_index, dataset in enumerate(base_dataset.multi_dataset._datasets):
            starts = dataset.episode_data_index["from"].detach().cpu().long()
            ends = dataset.episode_data_index["to"].detach().cpu().long()
            episode_ids = tuple(map(int, dataset.episodes)) if dataset.episodes is not None else tuple(range(len(starts)))
            if len(episode_ids) != len(starts):
                raise RuntimeError(f"Episode layout mismatch for {dataset.root}.")
            frames = int(dataset.num_frames)
            layouts.append(_DatasetLayout(
                dataset_index=dataset_index,
                dataset_root=_realpath(dataset.root),
                global_start=global_start,
                global_end=global_start + frames,
                episode_ids=episode_ids,
                episode_starts=starts,
                episode_ends=ends,
            ))
            global_start += frames
        return cls(layouts)

    @property
    def dataset_roots(self) -> list[str]:
        return [layout.dataset_root for layout in self.layouts]

    def resolve(self, global_index: int) -> SpectralSampleKey:
        index = int(global_index)
        if not 0 <= index < self.total_frames:
            raise IndexError(f"Global sample index {index} is outside [0,{self.total_frames}).")
        for layout in self.layouts:
            if index >= layout.global_end:
                continue
            local_index = index - layout.global_start
            position = int(torch.searchsorted(layout.episode_ends, torch.tensor(local_index), right=True).item())
            start = int(layout.episode_starts[position].item())
            end = int(layout.episode_ends[position].item())
            return SpectralSampleKey(
                dataset_index=layout.dataset_index,
                dataset_root=layout.dataset_root,
                episode_index=layout.episode_ids[position],
                frame_index=local_index - start,
                episode_length=end - start,
            )
        raise AssertionError("A valid global sample did not resolve to a dataset.")


class _TrackedDatasetProxy:
    """Records the actual BaseLerobotDataset index after its retry loop."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._state = threading.local()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self._dataset[index]
        if "idx" not in sample:
            raise KeyError("BaseLerobotDataset sample has no authoritative `idx`.")
        self._state.actual_index = int(sample["idx"])
        return sample

    def consume_actual_index(self) -> int:
        if not hasattr(self._state, "actual_index"):
            raise RuntimeError("No upstream sample index was recorded for this request.")
        result = int(self._state.actual_index)
        del self._state.actual_index
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


class SpectralEpisodeCache:
    """Read generic per-episode latent action target sidecars with worker-local LRU."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_dataset_roots: Sequence[str],
        horizon: int,
        target_dim: int,
        max_cached_episodes: int = 32,
        require_complete: bool = True,
        verify_manifest_identity: bool = False,
        expected_identity: Optional[dict[str, Any]] = None,
        allow_path_relocation: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.horizon = int(horizon)
        self.target_dim = int(target_dim)
        self.max_cached_episodes = int(max_cached_episodes)
        self.allow_path_relocation = bool(allow_path_relocation)
        if min(self.horizon, self.target_dim, self.max_cached_episodes) <= 0:
            raise ValueError("horizon, target_dim, and max_cached_episodes must be positive.")
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing latent-action cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.storage = str(self.manifest.get("target", {}).get("storage", "dense_episode_windows"))
        if self.storage not in {"dense_episode_windows", "compact_fixed_gap_pairs"}:
            raise ValueError(f"Unsupported latent action target storage: {self.storage!r}.")
        target = self.manifest.get("target", {})
        default_target_key = (
            "latent_pair_target" if self.storage == "compact_fixed_gap_pairs" else "latent_target"
        )
        default_mask_key = (
            "latent_pair_valid" if self.storage == "compact_fixed_gap_pairs" else "latent_target_mask"
        )
        self.target_key = str(target.get("key", default_target_key))
        self.mask_key = str(target.get("mask_key", default_mask_key))
        if not self.target_key or not self.mask_key:
            raise ValueError("latent action target manifest keys must be non-empty strings.")
        # Older completed caches encode an unused selection block as JSON null.
        # Treat null like an omitted optional block; explicit selection mappings
        # still opt into sparse authoritative-anchor semantics below.
        selection = self.manifest.get("selection") or {}
        if not isinstance(selection, dict):
            raise ValueError("latent action target manifest selection must be a mapping or null.")
        self.pair_selection = str(selection.get("pair_selection", ""))
        self.frame_offsets: tuple[int, ...] = ()
        self.pair_gap = 0
        if self.storage == "compact_fixed_gap_pairs":
            temporal = self.manifest.get("temporal", {})
            self.frame_offsets = tuple(int(value) for value in temporal.get("frame_offsets", []))
            self.pair_gap = int(temporal.get("pair_gap_frames", -1))
            if len(self.frame_offsets) != self.horizon + 1:
                raise ValueError(
                    f"Compact latent action cache needs H+1 frame offsets, got {self.frame_offsets}."
                )
            if self.frame_offsets[0] != 0 or any(
                right - left != self.pair_gap
                for left, right in zip(self.frame_offsets, self.frame_offsets[1:])
            ):
                raise ValueError("Compact latent action cache has inconsistent fixed-gap temporal metadata.")
        self._episodes: OrderedDict[
            Path, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
        ] = OrderedDict()
        self._validate_manifest(expected_dataset_roots, verify_manifest_identity, expected_identity)
        if require_complete:
            completion = self.manifest.get("completion", {})
            if completion.get("status") != "complete":
                raise ValueError("latent-action cache is not marked complete; set require_complete_cache=false only for a bounded smoke run.")

    def _validate_manifest(self, expected_roots: Sequence[str], verify_identity: bool, expected_identity: Optional[dict[str, Any]]) -> None:
        manifest = self.manifest
        if int(manifest.get("schema_version", -1)) != SPECTRAL_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported latent-action cache schema version.")
        target = manifest.get("target", {})
        if int(target.get("horizon", -1)) != self.horizon or int(target.get("dimension", -1)) != self.target_dim:
            raise ValueError(f"latent-action target contract mismatch: cache={target}, requested H={self.horizon}, D={self.target_dim}.")
        roots = [_realpath(item["root"]) for item in manifest.get("datasets", [])]
        expected = [_realpath(item) for item in expected_roots]
        if roots != expected and not self.allow_path_relocation:
            raise ValueError(f"latent-action cache dataset order differs from training data. cache={roots}, expected={expected}.")
        if len(roots) != len(expected):
            raise ValueError(
                f"latent-action cache dataset count differs from training data. cache={len(roots)}, expected={len(expected)}."
            )
        if verify_identity and expected_identity is not None and manifest.get("identity") != expected_identity:
            raise ValueError("latent-action cache identity differs from expected_identity.")

    def _load(
        self,
        key: SpectralSampleKey,
        expected_source: Optional[EgoStage1EpisodeIdentity] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        path = self.cache_root / key.shard_relative_path
        cached = self._episodes.get(path)
        if cached is not None:
            self._episodes.move_to_end(path)
            target, mask, source = cached
            self._validate_source(path, key, source, expected_source)
            return cached
        if not path.is_file():
            raise FileNotFoundError(f"Missing latent-action episode shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if self.storage == "compact_fixed_gap_pairs":
            target = torch.as_tensor(payload[self.target_key], dtype=torch.float32)
            mask = torch.as_tensor(payload[self.mask_key], dtype=torch.bool)
            if target.ndim != 2 or target.shape[1] != self.target_dim:
                raise ValueError(f"Unexpected latent_pair_target shape in {path}: {tuple(target.shape)}")
            if mask.shape != target.shape[:1]:
                raise ValueError(f"Unexpected latent_pair_valid shape in {path}: {tuple(mask.shape)}")
            expected_mask = torch.arange(target.shape[0]) + self.pair_gap < target.shape[0]
            if self.pair_selection == "authoritative_anchor_union":
                invalid_mask = mask & ~expected_mask
            else:
                invalid_mask = mask != expected_mask
            if bool(invalid_mask.any()):
                raise ValueError(f"Compact latent action pair-valid mask violates fixed-gap tail semantics: {path}")
        else:
            target = torch.as_tensor(payload[self.target_key], dtype=torch.float32)
            mask = torch.as_tensor(payload[self.mask_key], dtype=torch.bool)
            if target.ndim != 3 or target.shape[1:] != (self.horizon, self.target_dim):
                raise ValueError(f"Unexpected latent_target shape in {path}: {tuple(target.shape)}")
            if mask.shape != target.shape[:2]:
                raise ValueError(f"Unexpected latent_target_mask shape in {path}: {tuple(mask.shape)}")
        if target.shape[0] != key.episode_length:
            raise ValueError(f"Episode length mismatch in {path}: cache={target.shape[0]}, data={key.episode_length}")
        if not torch.isfinite(target).all():
            raise ValueError(f"Non-finite latent action target in {path}")
        source = payload.get("source", {})
        self._validate_source(path, key, source, expected_source)
        self._episodes[path] = (target, mask, source)
        while len(self._episodes) > self.max_cached_episodes:
            self._episodes.popitem(last=False)
        return target, mask, source

    def _validate_source(
        self,
        path: Path,
        key: SpectralSampleKey,
        source: dict[str, Any],
        expected_source: Optional[EgoStage1EpisodeIdentity],
    ) -> None:
        if source and (
            (
                not self.allow_path_relocation
                and _realpath(source.get("dataset_root", key.dataset_root)) != key.dataset_root
            )
            or int(source.get("dataset_index", key.dataset_index)) != key.dataset_index
            or int(source.get("episode_index", key.episode_index)) != key.episode_index
            or int(source.get("episode_length", key.episode_length)) != key.episode_length
        ):
            raise ValueError(f"latent-action shard identity mismatch: {path}")
        if expected_source is not None:
            required = ("dataset_root", "dataset_index", "episode_index", "episode_length", "frame_start", "video_path")
            missing = [name for name in required if name not in source]
            if missing:
                raise ValueError(f"Compact Stage-I shard lacks source identity {missing}: {path}")
            if (
                (
                    not self.allow_path_relocation
                    and _realpath(source["dataset_root"]) != expected_source.dataset_root
                )
                or int(source["dataset_index"]) != expected_source.dataset_index
                or int(source["episode_index"]) != expected_source.episode_index
                or int(source["episode_length"]) != expected_source.episode_length
                or int(source["frame_start"]) != expected_source.frame_start
                or (
                    not self.allow_path_relocation
                    and _realpath(source["video_path"]) != expected_source.source_video
                )
            ):
                raise ValueError(f"latent-action shard/source-split identity mismatch: {path}")

    def get(self, key: SpectralSampleKey) -> tuple[torch.Tensor, torch.Tensor]:
        """Return episode-dense windows; retained for legacy callers."""
        target, mask, _ = self._load(key)
        if self.storage == "compact_fixed_gap_pairs":
            return assemble_compact_latent_targets(
                target,
                mask,
                frame_offsets=self.frame_offsets,
                pair_gap=self.pair_gap,
            )
        return target, mask

    def get_window(
        self,
        key: SpectralSampleKey,
        *,
        expected_source: Optional[EgoStage1EpisodeIdentity] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target, mask, _ = self._load(key, expected_source)
        if self.storage == "compact_fixed_gap_pairs":
            window, window_mask = assemble_compact_latent_targets(
                target,
                mask,
                frame_offsets=self.frame_offsets,
                pair_gap=self.pair_gap,
                start_indices=torch.tensor([key.frame_index]),
            )
            return window[0], window_mask[0]
        return target[key.frame_index], mask[key.frame_index]


class SpectralTargetDataset(Dataset):
    """Attach latent action targets to ``RobotVideoDataset`` while keeping real actions.

    This is intentionally a wrapper, rather than a replacement action sidecar:
    action flow matching continues to receive the source dataset's real action.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        cache_root: str | Path,
        *,
        horizon: int = 8,
        target_dim: int = 32,
        max_cached_episodes: int = 32,
        require_complete_cache: bool = True,
        selected_dataset_roots: Optional[Sequence[str]] = None,
        episode_split: str = "all",
        episode_holdout_mod: int = 10,
        episode_holdout_remainder: int = 0,
        verify_manifest_identity: bool = False,
        expected_identity: Optional[dict[str, Any]] = None,
        logical_index_manifest: Optional[str | Path] = None,
        logical_split: str = "train",
        anchor_stride: int = 24,
        full_window_span: int = 48,
        require_proprio_unavailable: bool = False,
        vae_latent_cache_dir: Optional[str | Path] = None,
        vae_latent_cache_required: bool = False,
        vae_latent_cache_max_open_shards: int = 8,
        vae_latent_cache_expected_vae_hash: Optional[str] = "e1de6c02cdac79f8b739f4d3698cd216",
        allow_path_relocation: bool = False,
    ) -> None:
        if not hasattr(base_dataset, "lerobot_dataset"):
            raise TypeError("base_dataset must be WING RobotVideoDataset.")
        self.base_dataset = base_dataset
        raw = base_dataset.lerobot_dataset
        if isinstance(raw, _TrackedDatasetProxy):
            self._tracked = raw
            resolver_base = raw._dataset
        else:
            self._tracked = _TrackedDatasetProxy(raw)
            base_dataset.lerobot_dataset = self._tracked
            resolver_base = raw
        self.resolver = SpectralSampleResolver.from_base_lerobot_dataset(resolver_base)
        self.cache = SpectralEpisodeCache(
            cache_root,
            expected_dataset_roots=self.resolver.dataset_roots,
            horizon=horizon,
            target_dim=target_dim,
            max_cached_episodes=max_cached_episodes,
            require_complete=require_complete_cache,
            verify_manifest_identity=verify_manifest_identity,
            expected_identity=expected_identity,
            allow_path_relocation=allow_path_relocation,
        )
        self.horizon, self.target_dim = int(horizon), int(target_dim)
        split = str(episode_split).lower()
        if split not in {"all", "train", "heldout"}:
            raise ValueError("episode_split must be all, train, or heldout.")
        if episode_holdout_mod <= 1 or not 0 <= episode_holdout_remainder < episode_holdout_mod:
            raise ValueError("episode split requires modulus > 1 and a valid remainder.")
        selected = None if selected_dataset_roots is None else {_realpath(root) for root in selected_dataset_roots}
        unknown = set() if selected is None else selected - set(self.resolver.dataset_roots)
        if unknown:
            raise ValueError(f"selected_dataset_roots are absent from base_dataset: {sorted(unknown)}")
        self.logical_index = None
        self.require_proprio_unavailable = bool(require_proprio_unavailable)
        indices: list[int]
        if logical_index_manifest is not None:
            if selected is not None or split != "all":
                raise ValueError(
                    "Stage-I logical index owns dataset/split selection; selected_dataset_roots and episode_split must be unset/all."
                )
            self.logical_index = EgoStage1LogicalIndex(
                logical_index_manifest,
                split=logical_split,
                expected_dataset_roots=self.resolver.dataset_roots,
                expected_anchor_stride=anchor_stride,
                expected_window_span=full_window_span,
                allow_path_relocation=allow_path_relocation,
            )
            if int(getattr(base_dataset, "num_frames", -1)) != full_window_span + 1:
                raise ValueError(
                    "Stage-I RobotVideoDataset must expose 49 raw slots for a 48-frame window span."
                )
            if tuple(getattr(base_dataset, "video_sample_indices", ())) != self.logical_index.frame_offsets:
                raise ValueError(
                    "Stage-I RGB/model endpoint offsets differ from the logical-index contract."
                )
            if int(getattr(self._tracked, "global_sample_stride", -1)) != 1:
                raise ValueError("Stage-I raw window requires global_sample_stride=1.")
            if self.resolver.total_frames != sum(
                int(item["total_frames"]) for item in self.logical_index.manifest["datasets"]
            ):
                raise ValueError(
                    "Stage-I base dataset must expose all episodes in manifest root order (use val_set_proportion=0)."
                )
            manifest_episodes = self.logical_index.episodes
            for layout in self.resolver.layouts:
                active = [
                    manifest_episodes[(layout.dataset_index, episode_id)]
                    for episode_id in layout.episode_ids
                ]
                expected_ids = tuple(
                    item.episode_index
                    for item in sorted(
                        (
                            value for value in manifest_episodes.values()
                            if value.dataset_index == layout.dataset_index
                        ),
                        key=lambda value: value.global_start,
                    )
                )
                active_lengths = tuple(
                    int(end) - int(start)
                    for start, end in zip(
                        layout.episode_starts.tolist(), layout.episode_ends.tolist()
                    )
                )
                if layout.episode_ids != expected_ids or active_lengths != tuple(
                    item.episode_length for item in active
                ):
                    raise ValueError(
                        "Stage-I base dataset episode order/layout differs from the all-episodes manifest."
                    )
            indices = self.logical_index.source_indices.tolist()
            self.require_proprio_unavailable = True
        else:
            indices = []
            for layout in self.resolver.layouts:
                if selected is not None and layout.dataset_root not in selected:
                    continue
                for episode_id, start, end in zip(layout.episode_ids, layout.episode_starts.tolist(), layout.episode_ends.tolist()):
                    heldout = int(episode_id) % episode_holdout_mod == episode_holdout_remainder
                    if split != "all" and (split == "heldout") != heldout:
                        continue
                    indices.extend(range(layout.global_start + int(start), layout.global_start + int(end)))
        if not indices:
            raise ValueError("latent-action selection produced no samples.")
        self.sample_indices = tuple(indices)
        cache_text = "" if vae_latent_cache_dir is None else str(vae_latent_cache_dir).strip()
        logical_vae_root = cache_text or None
        if vae_latent_cache_required and logical_vae_root is None:
            raise ValueError(
                "vae_latent_cache_required=true but Stage-I logical VAE cache path is empty."
            )
        self.logical_vae_latent_cache = None
        if logical_vae_root is not None:
            if self.logical_index is None:
                raise ValueError("Sparse logical VAE cache requires logical_index_manifest.")
            if getattr(base_dataset, "vae_latent_cache", None) is not None:
                raise ValueError(
                    "Configure Stage-I VAE cache on SpectralTargetDataset, not inner RobotVideoDataset."
                )
            base_signature = base_dataset.vae_latent_cache_signature
            logical_cache = VaeLatentCache(
                logical_vae_root,
                expected_dataset_signature=self.logical_index.vae_dataset_signature(base_signature),
                expected_num_samples=len(self.logical_index),
                expected_vae_hash=vae_latent_cache_expected_vae_hash,
                require_complete=True,
                max_open_shards=vae_latent_cache_max_open_shards,
                expected_logical_metadata=self.logical_index.vae_logical_metadata,
                expected_logical_indices=self.logical_index.vae_identity_tensors,
                allow_path_relocation=allow_path_relocation,
            )
            base_dataset.vae_latent_cache = SparseSourceVaeLatentCache(logical_cache)
            base_dataset.lerobot_dataset._set_return_images(False)
            processor = base_dataset.lerobot_dataset.processor
            if processor is None:
                raise ValueError("Stage-I logical VAE cache requires a WING processor.")
            if bool(getattr(processor, "image_augmentation", False)):
                raise ValueError(
                    "Stage-I logical VAE cache cannot be used with image augmentation."
                )
            processor.set_image_processing_enabled(False)
            self.logical_vae_latent_cache = logical_cache

    def __len__(self) -> int:
        return len(self.sample_indices)

    @property
    def vae_latent_cache_signature(self) -> str:
        if self.logical_index is None:
            raise ValueError("Logical VAE signature requires a Stage-I logical index.")
        return self.logical_index.vae_dataset_signature(
            self.base_dataset.vae_latent_cache_signature
        )

    @property
    def vae_logical_metadata(self) -> dict[str, Any]:
        if self.logical_index is None:
            raise ValueError("Logical VAE metadata requires a Stage-I logical index.")
        return self.logical_index.vae_logical_metadata

    @property
    def vae_logical_indices(self) -> dict[str, torch.Tensor]:
        if self.logical_index is None:
            raise ValueError("Logical VAE indices require a Stage-I logical index.")
        return self.logical_index.vae_identity_tensors

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self.sample_indices[int(index)]
        sample = dict(self.base_dataset[base_index])
        actual_index = self._tracked.consume_actual_index()
        if actual_index != base_index:
            raise RuntimeError(f"Upstream data retry changed sample {base_index} to {actual_index}; refusing mismatched latent action target.")
        key = self.resolver.resolve(actual_index)
        expected_source = None
        if self.logical_index is not None:
            expected_source = self.logical_index.validate_resolved_sample(
                int(index),
                source_index=actual_index,
                dataset_index=key.dataset_index,
                episode_index=key.episode_index,
                frame_index=key.frame_index,
                episode_length=key.episode_length,
            )
        target, mask = self.cache.get_window(key, expected_source=expected_source)
        if self.logical_index is not None and not bool(mask.all()):
            raise RuntimeError("Stage-I eligible anchor produced an invalid compact latent action H8 window.")
        if self.logical_index is not None:
            image_is_pad = sample.get("image_is_pad")
            if image_is_pad is None or bool(torch.as_tensor(image_is_pad, dtype=torch.bool).any()):
                raise RuntimeError("Stage-I eligible anchor produced padded/missing RGB endpoints.")
        sample["latent_target"] = target
        sample["latent_target_mask"] = mask
        if self.require_proprio_unavailable:
            proprio_mask = sample.get("proprio_mask")
            if proprio_mask is None:
                raise RuntimeError("Ego Stage-I requires an explicit all-false proprio_mask.")
            if bool(torch.as_tensor(proprio_mask, dtype=torch.bool).any()):
                raise RuntimeError("Ego Stage-I sample unexpectedly contains available proprioception.")
            sample["proprio_available"] = torch.tensor(False, dtype=torch.bool)
        sample["latent_source_index"] = torch.tensor(actual_index, dtype=torch.long)
        sample["latent_source_dataset_index"] = torch.tensor(key.dataset_index, dtype=torch.long)
        sample["latent_source_episode_index"] = torch.tensor(key.episode_index, dtype=torch.long)
        sample["latent_source_frame_index"] = torch.tensor(key.frame_index, dtype=torch.long)
        return sample
