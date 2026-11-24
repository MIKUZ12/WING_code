"""Pickle-free cache records for WING-LAM pair supervision."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairSupervision:
    points0: np.ndarray
    points1: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    hand_seed: np.ndarray
    background: np.ndarray
    interaction: np.ndarray
    global_matrix: np.ndarray
    camera_displacement: np.ndarray
    destination_residual: np.ndarray
    canonical_residual: np.ndarray
    fit_confidence: float
    metadata: dict[str, Any]
    hand_masks: np.ndarray | None = None


def _validate_pair(record: PairSupervision) -> int:
    points0 = np.asarray(record.points0)
    points1 = np.asarray(record.points1)
    if points0.ndim != 2 or points0.shape[1] != 2 or points1.shape != points0.shape:
        raise ValueError(f"points0/points1 must share shape [K,2], got {points0.shape}/{points1.shape}")
    count = len(points0)
    for name in ("camera_displacement", "destination_residual", "canonical_residual"):
        value = np.asarray(getattr(record, name))
        if value.shape != (count, 2):
            raise ValueError(f"{name} must have shape [{count},2], got {value.shape}")
    for name in ("visibility", "confidence"):
        value = np.asarray(getattr(record, name))
        if value.shape != (count,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape [{count}], got {value.shape}")
    for name in ("valid", "hand_seed", "background", "interaction"):
        value = np.asarray(getattr(record, name))
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape [{count}], got {value.shape}")
    valid = np.asarray(record.valid, dtype=bool)
    hand_seed = np.asarray(record.hand_seed, dtype=bool)
    background = np.asarray(record.background, dtype=bool)
    interaction = np.asarray(record.interaction, dtype=bool)
    if np.any(hand_seed & ~valid) or np.any(background & ~valid) or np.any(interaction & ~valid):
        raise ValueError("hand_seed/background/interaction must be subsets of valid")
    if np.any(hand_seed & background):
        raise ValueError("hand_seed and background must be disjoint")
    matrix = np.asarray(record.global_matrix)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("global_matrix must be finite with shape [3,3]")
    if not np.isfinite(record.fit_confidence) or not 0 <= record.fit_confidence <= 1:
        raise ValueError("fit_confidence must be finite and in [0,1]")
    if record.hand_masks is not None:
        masks = np.asarray(record.hand_masks)
        if masks.ndim != 3 or masks.shape[0] != 2:
            raise ValueError(f"hand_masks must have shape [2,H,W], got {masks.shape}")
    try:
        json.dumps(record.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata must be finite JSON data: {error}") from error
    return count


def save_pair_supervision(path: str | Path, record: PairSupervision) -> None:
    """Atomically save one pair record as a compressed, pickle-free NPZ."""

    _validate_pair(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(record.metadata)
    arrays: dict[str, np.ndarray] = {
        "points0": np.asarray(record.points0, dtype=np.float32),
        "points1": np.asarray(record.points1, dtype=np.float32),
        "visibility": np.asarray(record.visibility, dtype=np.float32),
        "confidence": np.asarray(record.confidence, dtype=np.float32),
        "valid": np.asarray(record.valid, dtype=np.bool_),
        "hand_seed": np.asarray(record.hand_seed, dtype=np.bool_),
        "background": np.asarray(record.background, dtype=np.bool_),
        "interaction": np.asarray(record.interaction, dtype=np.bool_),
        "global_matrix": np.asarray(record.global_matrix, dtype=np.float64),
        "camera_displacement": np.asarray(record.camera_displacement, dtype=np.float32),
        "destination_residual": np.asarray(record.destination_residual, dtype=np.float32),
        "canonical_residual": np.asarray(record.canonical_residual, dtype=np.float32),
        "fit_confidence": np.asarray(record.fit_confidence, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)),
    }
    if record.hand_masks is not None:
        arrays["hand_masks"] = np.asarray(record.hand_masks, dtype=np.bool_)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pair_supervision(path: str | Path) -> PairSupervision:
    """Load and validate a pair record without enabling NumPy pickle."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        record = PairSupervision(
            points0=archive["points0"].copy(),
            points1=archive["points1"].copy(),
            visibility=archive["visibility"].copy(),
            confidence=archive["confidence"].copy(),
            valid=archive["valid"].copy(),
            hand_seed=archive["hand_seed"].copy(),
            background=archive["background"].copy(),
            interaction=archive["interaction"].copy(),
            global_matrix=archive["global_matrix"].copy(),
            camera_displacement=archive["camera_displacement"].copy(),
            destination_residual=archive["destination_residual"].copy(),
            canonical_residual=archive["canonical_residual"].copy(),
            fit_confidence=float(archive["fit_confidence"].item()),
            metadata=metadata,
            hand_masks=archive["hand_masks"].copy() if "hand_masks" in archive.files else None,
        )
    _validate_pair(record)
    return record
