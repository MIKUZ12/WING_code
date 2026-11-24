"""Validated, worker-safe BF16 VAE latent cache for RobotVideoDataset."""

from __future__ import annotations

import json
import os
import uuid
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


VAE_LATENT_CACHE_SCHEMA_VERSION = 1


class VaeLatentCache:
    """Read fixed-shape latent records without materializing a whole shard."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_dataset_signature: str,
        expected_num_samples: int,
        expected_vae_hash: str | None = None,
        require_complete: bool = True,
        max_open_shards: int = 8,
        expected_logical_metadata: dict[str, Any] | None = None,
        expected_logical_indices: dict[str, torch.Tensor] | None = None,
        allow_path_relocation: bool = False,
    ) -> None:
        self._fds: OrderedDict[Path, int] = OrderedDict()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing VAE latent manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != VAE_LATENT_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported VAE latent cache schema version.")
        if require_complete and self.manifest.get("status") != "complete":
            raise ValueError("VAE latent cache is not marked complete.")
        if (
            not allow_path_relocation
            and self.manifest.get("dataset_signature") != expected_dataset_signature
        ):
            raise ValueError("VAE latent cache dataset signature differs from the active dataset.")
        if expected_vae_hash is not None:
            actual_vae_hash = self.manifest.get("vae", {}).get("model_hash")
            if actual_vae_hash != expected_vae_hash:
                raise ValueError(
                    f"VAE latent cache model hash mismatch: cache={actual_vae_hash!r}, "
                    f"expected={expected_vae_hash!r}."
                )
        self.num_samples = int(self.manifest.get("num_samples", -1))
        if self.num_samples != int(expected_num_samples):
            raise ValueError(
                f"VAE latent cache length mismatch: cache={self.num_samples}, "
                f"dataset={expected_num_samples}."
            )
        if self.manifest.get("dtype") != "bfloat16":
            raise ValueError(f"VAE latent cache must use bfloat16, got {self.manifest.get('dtype')!r}.")
        self.latent_shape = tuple(int(x) for x in self.manifest.get("latent_shape", []))
        if len(self.latent_shape) != 4 or min(self.latent_shape) <= 0:
            raise ValueError(f"Invalid cached latent shape: {self.latent_shape}.")
        self.record_bytes = int(torch.tensor(self.latent_shape).prod().item()) * 2
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError("VAE latent manifest contains no shards.")
        expected_start = 0
        for shard in self.shards:
            start, end = int(shard["start"]), int(shard["end"])
            if start != expected_start or end <= start:
                raise ValueError(f"Non-contiguous VAE latent shard range: {shard}.")
            path = self.cache_dir / str(shard["file"])
            expected_bytes = (end - start) * self.record_bytes
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"VAE latent shard size mismatch for {path}: expected {expected_bytes} bytes."
                )
            expected_start = end
        if expected_start != self.num_samples:
            raise ValueError(
                f"VAE latent shard coverage ends at {expected_start}, expected {self.num_samples}."
            )
        self._shard_ends = [int(shard["end"]) for shard in self.shards]
        self.logical_indices: dict[str, torch.Tensor] | None = None
        if expected_logical_metadata is not None or expected_logical_indices is not None:
            if expected_logical_metadata is None or expected_logical_indices is None:
                raise ValueError(
                    "Logical VAE cache validation requires both metadata and identity tensors."
                )
            logical = self.manifest.get("logical_index")
            if not isinstance(logical, dict):
                raise ValueError("VAE cache has no logical-index contract.")
            actual_metadata = logical.get("metadata")
            if not isinstance(actual_metadata, dict):
                raise ValueError("VAE cache logical-index metadata must be a mapping.")
            if allow_path_relocation:
                # These fields encode storage locations, not sample identity.
                # The exact logical tensors below still bind every cache record
                # to dataset/episode/frame and prevent silent sample drift.
                ignored = {"selection_manifest", "dataset_roots"}
                actual_metadata = {
                    key: value
                    for key, value in actual_metadata.items()
                    if key not in ignored
                }
                expected_logical_metadata = {
                    key: value
                    for key, value in expected_logical_metadata.items()
                    if key not in ignored
                }
            if actual_metadata != expected_logical_metadata:
                raise ValueError("VAE cache logical-index metadata differs from the active dataset.")
            identity_path = self.cache_dir / str(logical.get("identity_file", ""))
            if not identity_path.is_file():
                raise FileNotFoundError(f"Missing VAE logical identity sidecar: {identity_path}")
            payload = torch.load(identity_path, map_location="cpu", weights_only=True)
            names = ("source_indices", "dataset_indices", "episode_indices", "frame_indices")
            actual = {
                name: torch.as_tensor(payload[name], dtype=torch.long).reshape(-1)
                for name in names
            }
            expected = {
                name: torch.as_tensor(expected_logical_indices[name], dtype=torch.long).reshape(-1)
                for name in names
            }
            for name in names:
                if actual[name].numel() != self.num_samples or not torch.equal(
                    actual[name], expected[name]
                ):
                    raise ValueError(
                        f"VAE logical identity tensor {name!r} differs from the active dataset."
                    )
            self.logical_indices = actual
        self.max_open_shards = max(int(max_open_shards), 1)

    def __len__(self) -> int:
        return self.num_samples

    def _find_shard(self, index: int) -> dict[str, Any]:
        position = bisect_right(self._shard_ends, int(index))
        if position >= len(self.shards):
            raise IndexError(f"VAE latent index {index} is not covered by the manifest.")
        shard = self.shards[position]
        if not int(shard["start"]) <= int(index) < int(shard["end"]):
            raise IndexError(f"VAE latent index {index} is not covered by the manifest.")
        return shard

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(path)
        if fd is not None:
            self._fds.move_to_end(path)
            return fd
        fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        while len(self._fds) > self.max_open_shards:
            _, old_fd = self._fds.popitem(last=False)
            os.close(old_fd)
        return fd

    def __getitem__(self, index: int) -> torch.Tensor:
        index = int(index)
        if not 0 <= index < self.num_samples:
            raise IndexError(f"VAE latent index {index} outside [0,{self.num_samples}).")
        shard = self._find_shard(index)
        local_index = index - int(shard["start"])
        path = self.cache_dir / str(shard["file"])
        raw = os.pread(self._fd(path), self.record_bytes, local_index * self.record_bytes)
        if len(raw) != self.record_bytes:
            raise IOError(f"Short read from VAE latent shard {path} at record {local_index}.")
        # Clone owns the memory after the temporary bytearray is released.
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().reshape(self.latent_shape)

    def close(self) -> None:
        for fd in getattr(self, "_fds", {}).values():
            os.close(fd)
        if hasattr(self, "_fds"):
            self._fds.clear()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fds"] = OrderedDict()
        return state


class SparseSourceVaeLatentCache:
    """Map sparse raw source indices to contiguous logical VAE records."""

    def __init__(self, cache: VaeLatentCache) -> None:
        if cache.logical_indices is None:
            raise ValueError("Sparse source lookup requires a validated logical VAE cache.")
        self.cache = cache
        self.source_indices = cache.logical_indices["source_indices"]
        if self.source_indices.numel() and not bool(
            (self.source_indices[1:] > self.source_indices[:-1]).all()
        ):
            raise ValueError("Logical VAE source indices must be strictly increasing.")
        self.latent_shape = cache.latent_shape

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, raw_source_index: int) -> torch.Tensor:
        source = int(raw_source_index)
        position = int(torch.searchsorted(self.source_indices, torch.tensor(source)).item())
        if position >= len(self.source_indices) or int(self.source_indices[position]) != source:
            raise KeyError(f"Raw source index {source} is absent from the logical VAE cache.")
        return self.cache[position]

    def close(self) -> None:
        self.cache.close()


def write_bfloat16_shard(
    path: str | Path,
    latents: torch.Tensor,
    *,
    temporary_tag: str | None = None,
) -> int:
    """Atomically write contiguous ``[N,C,T,H,W]`` BF16 records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = latents.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    tag = "writer" if temporary_tag is None else str(temporary_tag)
    temporary = path.with_name(
        f".{path.name}.tmp.{tag}.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(payload)
