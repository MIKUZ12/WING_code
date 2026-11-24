"""Worker-safe reader for packed BF16 text-embedding shards."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


PACKED_TEXT_CACHE_SCHEMA_VERSION = 1


class PackedTextEmbeddingCache:
    """Read variable-length BF16 contexts indexed by dataset-local task id."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_task_count: int,
        expected_context_len: int,
        expected_hidden_dim: int = 4096,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        manifest_path = self.cache_root / "packed_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing packed text-cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        schema = self.manifest.get(
            "schema_version", self.manifest.get("schema", -1)
        )
        if int(schema) != PACKED_TEXT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported packed text-cache schema: {schema!r}.")
        if self.manifest.get("status") != "complete":
            raise ValueError("Packed text cache is not marked complete.")
        if self.manifest.get("dtype") != "bfloat16":
            raise ValueError(
                "Packed text cache must use bfloat16, got "
                f"{self.manifest.get('dtype')!r}."
            )

        self.task_count = int(self.manifest.get("task_count", -1))
        self.context_len = int(self.manifest.get("context_len", -1))
        self.hidden_dim = int(self.manifest.get("hidden_dim", -1))
        self.shard_size = int(self.manifest.get("shard_size", -1))
        if self.task_count != int(expected_task_count):
            raise ValueError(
                "Packed text-cache task count mismatch: "
                f"cache={self.task_count}, dataset={expected_task_count}."
            )
        if self.context_len != int(expected_context_len):
            raise ValueError(
                "Packed text-cache context length mismatch: "
                f"cache={self.context_len}, requested={expected_context_len}."
            )
        if self.hidden_dim != int(expected_hidden_dim):
            raise ValueError(
                "Packed text-cache hidden dimension mismatch: "
                f"cache={self.hidden_dim}, expected={expected_hidden_dim}."
            )
        if min(self.task_count, self.context_len, self.hidden_dim, self.shard_size) <= 0:
            raise ValueError("Packed text-cache dimensions and shard_size must be positive.")

        self.shard_root = self.cache_root / "packed_shards"
        self.num_shards = math.ceil(self.task_count / self.shard_size)
        self._shards: list[dict[str, Any]] = []
        self._validate_shards()

        # File descriptors are process-local and opened only on first access in
        # that process. This avoids inheriting live descriptors into DataLoader
        # workers when Linux uses fork.
        self._runtime_pid: int | None = None
        self._context_fds: dict[int, int] = {}
        self._offset_fds: dict[int, int] = {}

    def _validate_shards(self) -> None:
        for shard_index in range(self.num_shards):
            prefix = self.shard_root / f"shard_{shard_index:06d}"
            context_path = Path(f"{prefix}.contexts.bf16")
            offsets_path = Path(f"{prefix}.offsets.u64")
            complete_path = Path(f"{prefix}.complete.json")
            for path in (context_path, offsets_path, complete_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing packed text-cache shard file: {path}")

            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if completion.get("status") != "complete":
                raise ValueError(f"Packed text-cache shard is incomplete: {complete_path}")

            start = shard_index * self.shard_size
            end = min(start + self.shard_size, self.task_count)
            rows = end - start
            expected_offset_bytes = (rows + 1) * 8
            if offsets_path.stat().st_size != expected_offset_bytes:
                raise ValueError(
                    f"Packed offsets size mismatch for {offsets_path}: "
                    f"expected {expected_offset_bytes} bytes."
                )

            offsets = np.memmap(offsets_path, mode="r", dtype="<u8", shape=(rows + 1,))
            try:
                if int(offsets[0]) != 0:
                    raise ValueError(f"Packed offsets must start at zero: {offsets_path}")
                lengths = np.diff(offsets)
                if bool((lengths > self.context_len).any()):
                    raise ValueError(
                        f"Packed context length exceeds {self.context_len}: {offsets_path}"
                    )
                token_count = int(offsets[-1])
            finally:
                del offsets
            expected_context_bytes = token_count * self.hidden_dim * 2
            if context_path.stat().st_size != expected_context_bytes:
                raise ValueError(
                    f"Packed context size mismatch for {context_path}: "
                    f"expected {expected_context_bytes} bytes."
                )

            optional_contract = {
                "task_start": start,
                "task_end": end,
                "row_count": rows,
                "token_count": token_count,
            }
            for key, expected in optional_contract.items():
                if key in completion and int(completion[key]) != expected:
                    raise ValueError(
                        f"Packed shard completion field {key!r} mismatch in "
                        f"{complete_path}: cache={completion[key]!r}, expected={expected}."
                    )
            self._shards.append(
                {
                    "context_path": context_path,
                    "offsets_path": offsets_path,
                    "rows": rows,
                }
            )

    def _close_runtime(self) -> None:
        for fd in (*self._context_fds.values(), *self._offset_fds.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._context_fds.clear()
        self._offset_fds.clear()
        self._runtime_pid = None

    def _ensure_runtime(self) -> None:
        pid = os.getpid()
        if self._runtime_pid != pid:
            self._close_runtime()
            self._runtime_pid = pid

    @staticmethod
    def _open_readonly(path: Path) -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        return os.open(path, flags)

    def _fds_for_shard(self, shard_index: int) -> tuple[int, int]:
        self._ensure_runtime()
        context_fd = self._context_fds.get(shard_index)
        offsets_fd = self._offset_fds.get(shard_index)
        shard = self._shards[shard_index]
        if context_fd is None:
            context_fd = self._open_readonly(shard["context_path"])
            self._context_fds[shard_index] = context_fd
        if offsets_fd is None:
            offsets_fd = self._open_readonly(shard["offsets_path"])
            self._offset_fds[shard_index] = offsets_fd
        return context_fd, offsets_fd

    @staticmethod
    def _pread_exact(fd: int, size: int, offset: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = os.pread(fd, remaining, offset + size - remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def __len__(self) -> int:
        return self.task_count

    def __getitem__(self, task_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        task_index = int(task_index)
        if task_index < 0 or task_index >= self.task_count:
            raise IndexError(
                f"Packed text-cache task index {task_index} is outside "
                f"[0,{self.task_count})."
            )
        shard_index, local_row = divmod(task_index, self.shard_size)
        context_fd, offsets_fd = self._fds_for_shard(shard_index)

        raw_offsets = self._pread_exact(offsets_fd, 16, local_row * 8)
        if len(raw_offsets) != 16:
            raise OSError(
                f"Short packed-offset read for task {task_index}: got {len(raw_offsets)} bytes."
            )
        start_token, end_token = struct.unpack("<QQ", raw_offsets)
        valid_tokens = end_token - start_token
        if valid_tokens > self.context_len:
            raise ValueError(
                f"Packed context for task {task_index} has {valid_tokens} tokens, "
                f"expected at most {self.context_len}."
            )

        context = torch.zeros(
            (self.context_len, self.hidden_dim), dtype=torch.bfloat16
        )
        if valid_tokens:
            target = context[:valid_tokens].view(torch.uint8).numpy().reshape(-1)
            byte_offset = start_token * self.hidden_dim * 2
            expected_bytes = valid_tokens * self.hidden_dim * 2
            bytes_read = 0
            while bytes_read < expected_bytes:
                current = os.preadv(
                    context_fd,
                    [target[bytes_read:]],
                    byte_offset + bytes_read,
                )
                if current == 0:
                    break
                bytes_read += current
            if bytes_read != expected_bytes:
                raise OSError(
                    f"Short packed-context read for task {task_index}: "
                    f"got {bytes_read}, expected {expected_bytes} bytes."
                )
        mask = torch.arange(self.context_len) < valid_tokens
        return context, mask

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime_pid"] = None
        state["_context_fds"] = {}
        state["_offset_fds"] = {}
        return state

    def __del__(self) -> None:
        try:
            self._close_runtime()
        except Exception:
            pass
