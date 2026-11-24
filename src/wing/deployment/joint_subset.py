"""Helpers for deploying partial-arm (N<14) WING models on a 14-d robot.

When a model is trained with ``--keep-dims "7:14"`` (right arm only), its
inputs/outputs are 7-d. The robot controller always speaks 14-d (left arm 7 +
right arm 7). The functions here implement the bidirectional mapping:

  robot_state[14] -> model_state[N]   = slice_state(state, keep_dims)
  model_action[N] -> robot_action[14] = pad_action(action, keep_dims, pad_template)

Kept as a small standalone module so both the RTC server (``WINGPolicyClient``)
and the legacy ckpt-inference entrypoints can use the same logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import DictConfig


# Robot-side dimensionality. Kept in sync with robot_runtime.ACTION_DIM and
# DualPiperController's 14-d joint layout (left arm 7 + right arm 7).
ROBOT_FULL_DIM = 14

# Robot joint names in the canonical 14-d order used by the retained datasets
# and the controller. The metadata check below catches ordering mismatches.
ROBOT_JOINT_NAMES: tuple[str, ...] = (
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
)


def parse_keep_dims(spec: str | None) -> list[int] | None:
    """Parse a keep_dims spec into a list of robot-joint indices to keep.

    Accepts:
      - None / "" -> None (no conversion; model is already robot-dim)
      - "start:stop" -> list(range(start, stop)) (Python slice semantics)
      - "i,j,k,..." -> [i, j, k, ...]
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"keep_dims slice must be 'start:stop', got {spec!r}")
        return list(range(int(parts[0]), int(parts[1])))
    return [int(x) for x in text.split(",") if x.strip()]


def validate_keep_dims(keep_dims: list[int], proprio_dim: int) -> None:
    """Validate keep_dims length and index range. Raises ValueError on mismatch."""
    if len(keep_dims) != proprio_dim:
        raise ValueError(
            f"keep_dims has {len(keep_dims)} indices but the model expects "
            f"proprio_output_dim={proprio_dim}; they must match."
        )
    for idx in keep_dims:
        if idx < 0 or idx >= ROBOT_FULL_DIM:
            raise ValueError(
                f"keep_dims index {idx} out of range [0, {ROBOT_FULL_DIM})."
            )


def build_pad_template(full_action_pad: Iterable[float] | None) -> np.ndarray:
    """Construct the 14-d pad template used when reconstructing robot actions.

    The template is required because silently filling an uncontrolled arm with
    zeros can command a large and unsafe pose change.
    """
    if full_action_pad is None:
        raise ValueError(
            "full_action_pad is required when keep_dims is configured; use the "
            "calibrated 14D initial pose for the physical robot."
        )
    pad = np.asarray(list(full_action_pad), dtype=np.float32)
    if pad.shape != (ROBOT_FULL_DIM,):
        raise ValueError(
            f"full_action_pad must be a list of {ROBOT_FULL_DIM} floats, "
            f"got shape {pad.shape}."
        )
    return pad


def slice_state(state_14d: np.ndarray, keep_dims: list[int]) -> np.ndarray:
    """Slice a 14-d robot state to N-d model state."""
    state = np.asarray(state_14d, dtype=np.float32)
    if state.shape != (ROBOT_FULL_DIM,):
        raise ValueError(
            f"Expected robot state of shape ({ROBOT_FULL_DIM},), got {state.shape}."
        )
    return state[keep_dims]


def pad_action(
    actions_nd: np.ndarray,
    keep_dims: list[int],
    pad_template: np.ndarray,
) -> np.ndarray:
    """Pad an [T, N] model-action chunk back to [T, 14] robot-action chunk.

    Non-kept dims are filled with ``pad_template`` (typically the held-still
    arm's init pose); ``keep_dims`` indices are overwritten with the model's
    N-d output.
    """
    if actions_nd.ndim != 2:
        raise ValueError(f"Expected actions of shape [T, N], got {actions_nd.shape}.")
    num_steps = actions_nd.shape[0]
    padded = np.tile(pad_template[None, :], (num_steps, 1)).astype(np.float32)
    padded[:, keep_dims] = actions_nd
    return padded


def load_full_action_pad_from_file(path: str | Path) -> list[float]:
    """Read a 14-float pad template from a JSON file.

    Accepted JSON formats:
      - bare list of floats: ``[0.077, 0.706, ..., 0.0]``
      - dict with key ``full_action_pad``: ``{"full_action_pad": [...]}``
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "full_action_pad" not in payload:
            raise ValueError(
                f"{path} must contain a `full_action_pad` key or be a bare list."
            )
        payload = payload["full_action_pad"]
    return [float(x) for x in payload]


def verify_keep_dims_against_lerobot(cfg: DictConfig, keep_dims: list[int]) -> None:
    """Cross-check deploy-time keep_dims against the training dataset's joint names.

    The training-time `--keep-dims` value is reflected in the lerobot meta as
    ``features.action.names`` (e.g. ``["right_waist", ..., "right_gripper"]``).
    The deploy-time ``keep_dims`` maps the model's N-dim output back to the
    robot's 14-dim joint indices. These two must agree: if training used
    ``--keep-dims "7:14"`` (right arm) but deploy sets ``keep_dims="0:7"``
    (left arm), both pass length validation but the model would command the
    wrong arm.

    On any inconsistency this raises ValueError. If the dataset dir or
    info.json cannot be located, this logs a warning and returns (deploy
    proceeds without the check).

    Note: only the FIRST entry in ``cfg.data.train.dataset_dirs`` is checked;
    if multiple datasets are concatenated for training, they are assumed to
    share the same joint subset (which is the only sensible setup).
    """
    dataset_dirs = None
    if "data" in cfg and "train" in cfg.data:
        dataset_dirs = cfg.data.train.get("dataset_dirs")
    if dataset_dirs is None or len(dataset_dirs) == 0:
        print("[joint_subset] keep_dims cross-check skipped: no dataset_dirs in task config.")
        return

    info_path = Path(str(dataset_dirs[0])) / "meta" / "info.json"
    if not info_path.exists():
        print(f"[joint_subset] keep_dims cross-check skipped: {info_path} not found.")
        return

    try:
        with info_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        trained_names = list(meta["features"]["action"]["names"])
    except (KeyError, ValueError, OSError) as exc:
        print(f"[joint_subset] keep_dims cross-check skipped: cannot read {info_path}: {exc}")
        return

    expected_names = [ROBOT_JOINT_NAMES[i] for i in keep_dims]
    if trained_names != expected_names:
        raise ValueError(
            "keep_dims mismatch between deploy and training dataset.\n"
            f"  deploy keep_dims={keep_dims} -> expects joint names {expected_names}\n"
            f"  training dataset {info_path} has features.action.names={trained_names}\n"
            "Either change deploy `keep_dims` to match the training data, "
            "or use a checkpoint that was actually trained on these joints."
        )
