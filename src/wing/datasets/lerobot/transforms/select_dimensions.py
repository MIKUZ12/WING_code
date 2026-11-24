"""Dimension selection for structured action / state batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class SelectDimensions:
    """Keep the same ordered dimensions in action, state, and their masks."""

    def __init__(self, indices: Sequence[int]):
        self.indices = tuple(int(index) for index in indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("`indices` must be non-empty and unique.")
        if min(self.indices) < 0:
            raise ValueError("`indices` cannot contain negative values.")

    def _select_mapping(self, values: Mapping[str, torch.Tensor]):
        if set(values) != {"default"}:
            raise ValueError(
                "SelectDimensions currently requires one `default` action/state group, "
                f"got {sorted(values)}."
            )
        tensor = values["default"]
        if max(self.indices) >= tensor.shape[-1]:
            raise ValueError(
                f"Dimension {max(self.indices)} is outside tensor width {tensor.shape[-1]}."
            )
        index = torch.as_tensor(self.indices, device=tensor.device, dtype=torch.long)
        return {"default": tensor.index_select(-1, index)}

    def forward(self, batch):
        batch = dict(batch)
        for key in ("action", "state", "action_mask", "state_mask"):
            if key in batch:
                batch[key] = self._select_mapping(batch[key])
        return batch
