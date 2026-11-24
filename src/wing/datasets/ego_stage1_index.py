"""Authoritative source-group split and logical anchors for Ego Stage-I."""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


EGO_STAGE1_INDEX_SCHEMA_VERSION = 1


def _realpath(path: str | Path) -> str:
    return os.path.realpath(os.fspath(path))


def _canonical_manifest_path(path: str | Path) -> str:
    """Normalize an already-canonical manifest path without touching storage.

    The split builder resolves source-video symlinks once.  Calling realpath
    again for every one of 43k episodes turns dataset startup into ~86k NFS
    lstat calls and can block for minutes under concurrent video workloads.
    """
    value = os.path.normpath(os.fspath(path))
    if not os.path.isabs(value):
        raise ValueError(f"Stage-I manifest path must be absolute: {value!r}.")
    return value


@dataclass(frozen=True)
class EgoStage1EpisodeIdentity:
    dataset_index: int
    dataset_root: str
    episode_index: int
    episode_length: int
    frame_start: int
    source_video: str
    source_group: str
    split: str
    global_start: int


class EgoStage1LogicalIndex:
    """Load a completed eligible-anchor index without redefining its split."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        expected_dataset_roots: Sequence[str],
        expected_anchor_stride: int = 24,
        expected_window_span: int = 48,
        allow_path_relocation: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing Ego Stage-I index manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != EGO_STAGE1_INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported Ego Stage-I index schema version.")
        if self.manifest.get("status") != "complete":
            raise ValueError("Ego Stage-I index manifest is not complete.")

        split = str(split).lower()
        if split not in {"train", "val", "all"}:
            raise ValueError("Ego Stage-I logical split must be train, val, or all.")
        self.split = split
        roots = [_realpath(item["root"]) for item in self.manifest.get("datasets", [])]
        expected = [_realpath(root) for root in expected_dataset_roots]
        self.allow_path_relocation = bool(allow_path_relocation)
        if roots != expected and not self.allow_path_relocation:
            raise ValueError(
                f"Ego Stage-I index dataset order mismatch: manifest={roots}, active={expected}."
            )
        if len(roots) != len(expected):
            raise ValueError(
                f"Ego Stage-I index dataset count mismatch: manifest={len(roots)}, active={len(expected)}."
            )
        # Dataset index/order, episode layout and logical identity tensors remain
        # authoritative after a release bundle is downloaded to another root.
        # Only the storage prefix changes.  Keep both forms so cache validation
        # can distinguish an intentional relocation from a changed sample set.
        self.manifest_dataset_roots = tuple(roots)
        self.dataset_roots = tuple(expected)

        sampling = self.manifest.get("sampling", {})
        self.anchor_stride = int(sampling.get("anchor_stride", -1))
        self.window_span = int(sampling.get("window_span", -1))
        self.frame_offsets = tuple(int(x) for x in sampling.get("frame_offsets", []))
        if self.anchor_stride != int(expected_anchor_stride):
            raise ValueError(
                f"Ego Stage-I anchor stride mismatch: {self.anchor_stride} vs {expected_anchor_stride}."
            )
        if self.window_span != int(expected_window_span):
            raise ValueError(
                f"Ego Stage-I window span mismatch: {self.window_span} vs {expected_window_span}."
            )
        if not self.frame_offsets or self.frame_offsets[0] != 0 or self.frame_offsets[-1] != self.window_span:
            raise ValueError("Ego Stage-I frame offsets do not cover the declared window span.")

        episodes: dict[tuple[int, int], EgoStage1EpisodeIdentity] = {}
        group_splits: dict[str, str] = {}
        for item in self.manifest.get("episodes", []):
            identity = EgoStage1EpisodeIdentity(
                dataset_index=int(item["dataset_index"]),
                dataset_root=self.dataset_roots[int(item["dataset_index"])],
                episode_index=int(item["episode_index"]),
                episode_length=int(item["episode_length"]),
                frame_start=int(item["frame_start"]),
                source_video=_canonical_manifest_path(item["source_video"]),
                source_group=_canonical_manifest_path(item["source_group"]),
                split=str(item["split"]),
                global_start=int(item["global_start"]),
            )
            key = (identity.dataset_index, identity.episode_index)
            if key in episodes:
                raise ValueError(f"Duplicate episode identity in Stage-I manifest: {key}.")
            manifest_dataset_root = _canonical_manifest_path(item["dataset_root"])
            if manifest_dataset_root != self.manifest_dataset_roots[identity.dataset_index]:
                raise ValueError(f"Episode dataset root mismatch in Stage-I manifest: {key}.")
            previous_split = group_splits.setdefault(identity.source_group, identity.split)
            if previous_split != identity.split:
                raise ValueError(
                    f"Source group crosses Stage-I splits: {identity.source_group}."
                )
            episodes[key] = identity
        expected_episodes = int(self.manifest.get("counts", {}).get("all", {}).get("episodes", -1))
        if len(episodes) != expected_episodes:
            raise ValueError(
                f"Stage-I episode count mismatch: manifest={expected_episodes}, records={len(episodes)}."
            )
        self.episodes = episodes

        files = self.manifest.get("logical_index_files", {})
        entry = files.get(split)
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError(f"Stage-I manifest has no logical index for split {split!r}.")
        index_path = self.manifest_path.parent / str(entry["file"])
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing Stage-I logical index payload: {index_path}")
        payload: dict[str, Any] = torch.load(index_path, map_location="cpu", weights_only=True)
        names = ("source_indices", "dataset_indices", "episode_indices", "frame_indices")
        tensors = {
            name: torch.as_tensor(payload[name], dtype=torch.long).reshape(-1) for name in names
        }
        count = int(entry.get("count", -1))
        if any(tensor.numel() != count for tensor in tensors.values()):
            shapes = {name: value.numel() for name, value in tensors.items()}
            raise ValueError(f"Stage-I logical index tensor length mismatch: {shapes}, expected {count}.")
        if count and not bool((tensors["source_indices"][1:] > tensors["source_indices"][:-1]).all()):
            raise ValueError("Stage-I source indices must be strictly increasing and unique.")
        frames = tensors["frame_indices"]
        if count and bool(((frames < 0) | (frames.remainder(self.anchor_stride) != 0)).any()):
            raise ValueError("Stage-I frame indices violate episode-local anchor stride.")
        expected_count = int(self.manifest.get("counts", {}).get(split, {}).get("anchors", -1))
        if count != expected_count:
            raise ValueError(
                f"Stage-I logical anchor count mismatch: payload={count}, manifest={expected_count}."
            )
        cursor = 0
        ordered_episodes = sorted(episodes.values(), key=lambda item: item.global_start)
        for identity in ordered_episodes:
            if split != "all" and identity.split != split:
                continue
            expected_frames = torch.arange(
                0,
                max(identity.episode_length - self.window_span, 0),
                self.anchor_stride,
                dtype=torch.long,
            )
            width = int(expected_frames.numel())
            region = slice(cursor, cursor + width)
            if width and (
                not torch.equal(tensors["frame_indices"][region], expected_frames)
                or not torch.equal(
                    tensors["source_indices"][region], expected_frames + identity.global_start
                )
                or not bool((tensors["dataset_indices"][region] == identity.dataset_index).all())
                or not bool((tensors["episode_indices"][region] == identity.episode_index).all())
            ):
                raise ValueError(
                    f"Stage-I logical anchors disagree with episode identity {(identity.dataset_index, identity.episode_index)}."
                )
            cursor += width
        if cursor != count:
            raise ValueError(
                f"Stage-I logical index coverage mismatch: reconstructed={cursor}, payload={count}."
            )
        self.source_indices = tensors["source_indices"]
        self.dataset_indices = tensors["dataset_indices"]
        self.episode_indices = tensors["episode_indices"]
        self.frame_indices = tensors["frame_indices"]

    @property
    def vae_identity_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "source_indices": self.source_indices,
            "dataset_indices": self.dataset_indices,
            "episode_indices": self.episode_indices,
            "frame_indices": self.frame_indices,
        }

    @property
    def vae_logical_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selection_manifest": str(self.manifest_path),
            "split": self.split,
            "num_samples": len(self),
            "dataset_roots": list(self.dataset_roots),
            "anchor_stride": self.anchor_stride,
            "window_span": self.window_span,
            "frame_offsets": list(self.frame_offsets),
        }

    def vae_dataset_signature(self, base_dataset_signature: str) -> str:
        payload = {
            "schema_version": 1,
            "base_dataset_signature": str(base_dataset_signature),
            "logical_index": self.vae_logical_metadata,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return int(self.source_indices.numel())

    def episode_identity(self, dataset_index: int, episode_index: int) -> EgoStage1EpisodeIdentity:
        key = (int(dataset_index), int(episode_index))
        try:
            return self.episodes[key]
        except KeyError as exc:
            raise KeyError(f"Episode {key} is absent from the Stage-I split manifest.") from exc

    def validate_resolved_sample(
        self,
        logical_index: int,
        *,
        source_index: int,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
        episode_length: int,
    ) -> EgoStage1EpisodeIdentity:
        position = int(logical_index)
        observed = (
            int(source_index), int(dataset_index), int(episode_index), int(frame_index)
        )
        expected = (
            int(self.source_indices[position]),
            int(self.dataset_indices[position]),
            int(self.episode_indices[position]),
            int(self.frame_indices[position]),
        )
        if observed != expected:
            raise RuntimeError(
                f"Stage-I logical/source identity mismatch at {position}: observed={observed}, expected={expected}."
            )
        identity = self.episode_identity(dataset_index, episode_index)
        if identity.episode_length != int(episode_length):
            raise RuntimeError("Stage-I episode length differs from the active dataset.")
        if identity.split != self.split and self.split != "all":
            raise RuntimeError("Stage-I episode belongs to a different source-group split.")
        if int(frame_index) + self.window_span >= int(episode_length):
            raise RuntimeError("Stage-I logical anchor is not a full window.")
        return identity
