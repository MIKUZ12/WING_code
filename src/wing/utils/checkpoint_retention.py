"""Conservative retention for WING training checkpoints.

Only complete ``weights/step_XXXXXX.pt`` + ``state/step_XXXXXX`` pairs are
eligible.  Planning is separate from deletion so callers can log a dry run
before applying it; incomplete directories, orphan files, and symlinks are
left untouched.

Separate weight/state retention uses a small ``.weights_only`` sidecar when a
complete state is removed.  This distinguishes intentionally retained weight
checkpoints from unknown orphan files without making them eligible for resume.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_STEP_RE = re.compile(r"^step_(\d{6,})$")
_WEIGHTS_STEP_RE = re.compile(r"^step_(\d{6,})\.pt$")


@dataclass(frozen=True)
class CheckpointGroup:
    step: int
    weights_path: Path
    state_path: Path


@dataclass(frozen=True)
class WeightCheckpoint:
    step: int
    weights_path: Path
    marker_path: Path | None


def _read_step(path: Path) -> int | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return int(payload["global_step"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _checkpoint_roots(checkpoint_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(checkpoint_root).absolute()
    if root == Path(root.anchor) or root.is_symlink():
        raise ValueError(f"Unsafe checkpoint root: {root}")
    weights_dir = root / "weights"
    state_dir = root / "state"
    for directory in (root, weights_dir, state_dir):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Checkpoint directory is missing or unsafe: {directory}")
    return root, weights_dir, state_dir


def _validated_group(weights_dir: Path, state_dir: Path, state_path: Path) -> CheckpointGroup | None:
    match = _STEP_RE.fullmatch(state_path.name)
    if match is None or state_path.parent != state_dir:
        return None
    if not state_path.is_dir() or state_path.is_symlink():
        return None
    step = int(match.group(1))
    weights_path = weights_dir / f"step_{step:06d}.pt"
    if not weights_path.is_file() or weights_path.is_symlink():
        return None
    if _read_step(state_path / ".complete") != step:
        return None
    if _read_step(state_path / "trainer_state.json") != step:
        return None
    return CheckpointGroup(step=step, weights_path=weights_path, state_path=state_path)


def _parse_weight(weights_dir: Path, weights_path: Path) -> tuple[int, Path] | None:
    if weights_path.parent != weights_dir:
        return None
    match = _WEIGHTS_STEP_RE.fullmatch(weights_path.name)
    if match is None or not weights_path.is_file() or weights_path.is_symlink():
        return None
    return int(match.group(1)), weights_path


def _weights_only_marker(weights_path: Path) -> Path:
    return weights_path.with_name(f"{weights_path.name}.weights_only")


def _validated_retained_weight(
    weights_dir: Path, state_dir: Path, weights_path: Path
) -> WeightCheckpoint | None:
    parsed = _parse_weight(weights_dir, weights_path)
    if parsed is None:
        return None
    step, weights_path = parsed
    state_path = state_dir / f"step_{step:06d}"
    if _validated_group(weights_dir, state_dir, state_path) is not None:
        marker_path = _weights_only_marker(weights_path)
        marker = marker_path if _read_step(marker_path) == step else None
        return WeightCheckpoint(step=step, weights_path=weights_path, marker_path=marker)
    marker_path = _weights_only_marker(weights_path)
    if _read_step(marker_path) != step:
        return None
    return WeightCheckpoint(step=step, weights_path=weights_path, marker_path=marker_path)


def complete_checkpoint_groups(checkpoint_root: str | Path) -> tuple[CheckpointGroup, ...]:
    """Return validated complete checkpoint pairs ordered by optimizer step."""
    _, weights_dir, state_dir = _checkpoint_roots(checkpoint_root)
    groups = []
    for state_path in state_dir.iterdir():
        group = _validated_group(weights_dir, state_dir, state_path)
        if group is not None:
            groups.append(group)
    return tuple(sorted(groups, key=lambda group: group.step))


def retained_weight_checkpoints(checkpoint_root: str | Path) -> tuple[WeightCheckpoint, ...]:
    """Return complete or explicitly demoted weight checkpoints, ordered by step."""
    _, weights_dir, state_dir = _checkpoint_roots(checkpoint_root)
    weights = []
    for weights_path in weights_dir.iterdir():
        checkpoint = _validated_retained_weight(weights_dir, state_dir, weights_path)
        if checkpoint is not None:
            weights.append(checkpoint)
    return tuple(sorted(weights, key=lambda checkpoint: checkpoint.step))


def plan_checkpoint_prune(
    checkpoint_root: str | Path, *, keep_last: int
) -> tuple[CheckpointGroup, ...]:
    """Plan removal of old complete pairs without mutating the filesystem."""
    keep_last = int(keep_last)
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    groups = complete_checkpoint_groups(checkpoint_root)
    return groups[:-keep_last] if len(groups) > keep_last else ()


def plan_checkpoint_state_prune(
    checkpoint_root: str | Path, *, keep_last: int
) -> tuple[CheckpointGroup, ...]:
    """Plan removal of old complete training states while retaining their weights."""
    keep_last = int(keep_last)
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    groups = complete_checkpoint_groups(checkpoint_root)
    return groups[:-keep_last] if len(groups) > keep_last else ()


def plan_checkpoint_weights_prune(
    checkpoint_root: str | Path, *, keep_last: int
) -> tuple[WeightCheckpoint, ...]:
    """Plan removal of old verified weights-only or complete checkpoints."""
    keep_last = int(keep_last)
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    weights = retained_weight_checkpoints(checkpoint_root)
    return weights[:-keep_last] if len(weights) > keep_last else ()


def apply_checkpoint_prune(
    checkpoint_root: str | Path, groups: Iterable[CheckpointGroup]
) -> tuple[int, ...]:
    """Delete an already-planned set of old pairs after revalidating every target."""
    _, weights_dir, state_dir = _checkpoint_roots(checkpoint_root)
    removed = []
    for planned in tuple(groups):
        current = _validated_group(weights_dir, state_dir, planned.state_path)
        if current != planned:
            raise RuntimeError(
                f"Checkpoint changed after prune dry-run; refusing deletion: {planned.state_path}"
            )
        # State first: an interrupted prune can leave an orphan weights file, but never a
        # complete state directory that auto-resume could select after its pair was removed.
        shutil.rmtree(current.state_path)
        current.weights_path.unlink()
        removed.append(current.step)
    return tuple(removed)


def apply_checkpoint_state_prune(
    checkpoint_root: str | Path, groups: Iterable[CheckpointGroup]
) -> tuple[int, ...]:
    """Demote planned full states to verified weights-only checkpoints."""
    _, weights_dir, state_dir = _checkpoint_roots(checkpoint_root)
    removed = []
    for planned in tuple(groups):
        current = _validated_group(weights_dir, state_dir, planned.state_path)
        if current != planned:
            raise RuntimeError(
                f"Checkpoint changed after state-prune dry-run; refusing deletion: "
                f"{planned.state_path}"
            )
        marker_path = _weights_only_marker(current.weights_path)
        existing_step = _read_step(marker_path) if marker_path.exists() else current.step
        if existing_step != current.step or marker_path.is_symlink():
            raise RuntimeError(f"Unsafe weights-only marker: {marker_path}")
        tmp_marker = marker_path.with_name(f".{marker_path.name}.tmp")
        if tmp_marker.exists() or tmp_marker.is_symlink():
            raise RuntimeError(f"Stale weights-only marker temporary file: {tmp_marker}")
        with tmp_marker.open("x", encoding="utf-8") as handle:
            json.dump({"global_step": current.step}, handle, ensure_ascii=True)
        tmp_marker.replace(marker_path)
        # The marker is durable before removing full state.  An interruption can therefore
        # only leave an extra resumable state or a verified weights-only checkpoint.
        shutil.rmtree(current.state_path)
        removed.append(current.step)
    return tuple(removed)


def apply_checkpoint_weights_prune(
    checkpoint_root: str | Path, checkpoints: Iterable[WeightCheckpoint]
) -> tuple[int, ...]:
    """Delete planned old weights after confirming they are no longer resumable states."""
    _, weights_dir, state_dir = _checkpoint_roots(checkpoint_root)
    removed = []
    for planned in tuple(checkpoints):
        current = _validated_retained_weight(weights_dir, state_dir, planned.weights_path)
        if current is None or current.step != planned.step:
            raise RuntimeError(
                f"Checkpoint changed after weights-prune dry-run; refusing deletion: "
                f"{planned.weights_path}"
            )
        state_path = state_dir / f"step_{current.step:06d}"
        if _validated_group(weights_dir, state_dir, state_path) is not None:
            raise RuntimeError(
                f"Refusing to delete weights with a complete training state: {current.weights_path}"
            )
        marker_path = _weights_only_marker(current.weights_path)
        if current.marker_path != marker_path or _read_step(marker_path) != current.step:
            raise RuntimeError(f"Missing verified weights-only marker: {marker_path}")
        current.weights_path.unlink()
        marker_path.unlink()
        removed.append(current.step)
    return tuple(removed)


__all__ = [
    "CheckpointGroup",
    "WeightCheckpoint",
    "apply_checkpoint_prune",
    "apply_checkpoint_state_prune",
    "apply_checkpoint_weights_prune",
    "complete_checkpoint_groups",
    "plan_checkpoint_prune",
    "plan_checkpoint_state_prune",
    "plan_checkpoint_weights_prune",
    "retained_weight_checkpoints",
]
