"""Adapters between the official RoboCasa-GR1 Gym API and WING."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# This is the cropped camera used to produce observation.images.ego_view in
# the released training data. The similarly shaped uncropped key is not
# equivalent and silently causes a train/eval mismatch.
ROBOCASA_VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
ROBOCASA_PROMPT_KEY = "annotation.human.coarse_action"
GR1_FULL_DIM = 44
GR1_CONTROLLED_DIM = 29

GR1_MODALITIES = (
    ("left_arm", 7),
    ("left_hand", 6),
    ("left_leg", 6),
    ("neck", 3),
    ("right_arm", 7),
    ("right_hand", 6),
    ("right_leg", 6),
    ("waist", 3),
)
GR1_SIMULATED_MODALITIES = frozenset(
    {"left_arm", "left_hand", "right_arm", "right_hand", "waist"}
)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"Expected {name} shape ({size},), got {result.shape}.")
    return result


def robocasa_observation_to_policy(
    obs: Mapping[str, Any], *, policy_dim: int = GR1_FULL_DIM
) -> dict[str, Any]:
    """Convert one GR1 Gym observation to a 29-D or legacy 44-D payload."""
    if ROBOCASA_VIDEO_KEY not in obs:
        raise KeyError(f"RoboCasa observation is missing {ROBOCASA_VIDEO_KEY!r}.")

    if policy_dim not in (GR1_CONTROLLED_DIM, GR1_FULL_DIM):
        raise ValueError(
            f"RoboCasa GR1 policy_dim must be {GR1_CONTROLLED_DIM} or {GR1_FULL_DIM}, "
            f"got {policy_dim}."
        )

    state_parts = []
    for name, size in GR1_MODALITIES:
        if name in GR1_SIMULATED_MODALITIES:
            state_parts.append(_vector(obs[f"state.{name}"], size, f"state.{name}"))
        elif policy_dim == GR1_FULL_DIM:
            state_parts.append(np.zeros(size, dtype=np.float32))
    state = np.concatenate(state_parts)
    assert state.shape == (policy_dim,)
    return {
        "images": {"ego_view": np.asarray(obs[ROBOCASA_VIDEO_KEY], dtype=np.uint8)},
        "state": state,
        "prompt": str(obs.get(ROBOCASA_PROMPT_KEY, "")),
    }


def wing_action_to_robocasa(action: Any) -> dict[str, np.ndarray]:
    """Convert a compact 29-D or legacy 44-D policy action to Gym groups."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] not in (GR1_CONTROLLED_DIM, GR1_FULL_DIM):
        raise ValueError(
            f"Expected WING action dim {GR1_CONTROLLED_DIM} or {GR1_FULL_DIM}, "
            f"got {action.shape}."
        )
    compact = action.shape[0] == GR1_CONTROLLED_DIM
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, size in GR1_MODALITIES:
        if name in GR1_SIMULATED_MODALITIES:
            values = action[offset : offset + size]
            result[f"action.{name}"] = values.copy()
            offset += size
        elif not compact:
            offset += size
    assert offset == action.shape[0]
    return result
