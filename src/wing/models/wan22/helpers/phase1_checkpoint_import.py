"""State-only compatibility import for released Phase-I planner checkpoints.

The canonical runtime never constructs the historical planner class.  This
module recognizes its exact predictor state layout and returns only the trunk
tensors shared with the native fixed Full-K4 predictor.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


_OUTPUT_KEYS = frozenset({"out.weight", "out.bias"})
_RELEASED_60K_ONLY_KEYS = frozenset(
    {"gate_head.weight", "gate_head.bias", "missing_proprio_token"}
)


def _require_tensor_state(
    state: Mapping[str, torch.Tensor], *, label: str
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} must be a state-dict mapping.")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} must contain only string-to-tensor entries.")
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(f"{label}[{key!r}] uses unsupported tensor storage.")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{label}[{key!r}] contains NaN or Inf.")
    return state


def _validate_exact_tensor(
    source: torch.Tensor, target: torch.Tensor, *, label: str
) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"{label} shape mismatch: source={tuple(source.shape)} "
            f"target={tuple(target.shape)}."
        )
    source_float = source.is_floating_point()
    target_float = target.is_floating_point()
    if source_float != target_float or source.is_complex() != target.is_complex():
        raise ValueError(
            f"{label} dtype-family mismatch: source={source.dtype} target={target.dtype}."
        )
    if not source_float and not source.is_complex() and source.dtype != target.dtype:
        raise ValueError(
            f"{label} non-floating dtype mismatch: "
            f"source={source.dtype} target={target.dtype}."
        )
    canonical = source.to(dtype=target.dtype)
    if (canonical.is_floating_point() or canonical.is_complex()) and not bool(
        torch.isfinite(canonical).all().item()
    ):
        raise ValueError(f"{label} becomes non-finite after conversion.")


def import_phase1_planner_backbone(
    source_state: Mapping[str, torch.Tensor],
    target_fixed_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate a supported Phase-I state and return its reusable fixed trunk.

    Accepted inputs are either a native fixed Full-K4 predictor state or the
    exact 24-key layout of the released Phase-I 60k predictor.  Output-head
    tensors are intentionally excluded because Phase II has a newly initialized
    robot-target coefficient head.
    """
    source = _require_tensor_state(source_state, label="Phase-I predictor")
    target = _require_tensor_state(target_fixed_state, label="Phase-II fixed predictor")
    target_keys = frozenset(target)
    if not _OUTPUT_KEYS.issubset(target_keys):
        raise ValueError("Phase-II fixed predictor is missing its coefficient output head.")

    native_keys = target_keys
    released_60k_keys = target_keys | _RELEASED_60K_ONLY_KEYS
    source_keys = frozenset(source)
    if source_keys not in {native_keys, released_60k_keys}:
        raise ValueError(
            "Unsupported Phase-I predictor state layout; expected native fixed "
            "Full-K4 or the released Phase-I 60k compatibility layout."
        )

    trunk_keys = target_keys - _OUTPUT_KEYS
    imported = {key: source[key] for key in trunk_keys}
    for key in sorted(trunk_keys):
        _validate_exact_tensor(
            imported[key], target[key], label=f"Phase-I planner backbone.{key}"
        )
    return imported
