"""Runtime placement of a group's native action/state into the fixed 32D unified
layout (eef-first), emitting the corresponding per-dim validity mask.

This avoids any on-disk repack: each source dataset stays in its native dimension
(e.g. EgoDex eef-14D, ALOHA joint-14D) and is scattered into the 32D layout
in-memory at load time, then RelativeUnified32DTransform makes the eef pose relative.

Legacy experimental layout (kept as a reusable transform, without a maintained training recipe):
    [0:14]  eef block  : [0:6] L eef pose, [6] L grip, [7:13] R eef pose, [13] R grip
    [14:30] joint block: [14:21] L joint(7), [21] L grip, [22:29] R joint(7), [29] R grip
    [30:32] base reserved
"""
from typing import Dict, List, Tuple

import torch

UNIFIED_DIM = 32

# A "block" is ((src_start, src_end), (tgt_start, tgt_end)) with equal length.
# Invariants (checked in __init__): every source dim is mapped exactly once, and
# target blocks are disjoint and within [0, UNIFIED_DIM).
Block = Tuple[Tuple[int, int], Tuple[int, int]]

LAYOUTS: Dict[str, Dict] = {
    # EgoDex native 14D eef (data/egodex_pnp_15tasks_eef14, sliced from legacy 28D
    # by scripts/data_process/slice_unified28_to_eef14.py): [0:6] L eef pose,
    # [6] L grip, [7:13] R eef pose, [13] R grip -- dim-for-dim the 32D eef block.
    # -> 32D eef block [0:14]; joint block [14:30] + base [30:32] stay masked.
    "egodex14_eef": {
        "src_dim": 14,
        "blocks": [((0, 14), (0, 14))],
    },
    # ALOHA/Piper native 14D joint: [0:6] L joint, [6] L grip, [7:13] R joint, [13] R grip.
    # -> 32D joint block [14:30]; 6-DoF arms leave the reserved 7th joint slots
    # ([20] and [28]) masked off; eef block [0:14] + base [30:32] stay masked.
    "aloha14_joint": {
        "src_dim": 14,
        "blocks": [((0, 6), (14, 20)), ((6, 7), (21, 22)), ((7, 13), (22, 28)), ((13, 14), (29, 30))],
    },
}


class UnifiedSlotLayout:
    """Scatter native action/state into the fixed 32D layout + emit the per-dim mask.

    Use as the FIRST action_state_transform (before RelativeUnified32DTransform).
    Always (re)writes ``action_mask`` / ``state_mask`` with the layout-derived 32D
    mask, so the mask matches the placed vector whether or not the source dataset
    shipped its own on-disk mask.
    """

    def __init__(self, source_layout: str, keys: List[str]):
        if source_layout not in LAYOUTS:
            raise ValueError(f"Unknown source_layout {source_layout!r}; known: {sorted(LAYOUTS)}")
        self.source_layout = source_layout
        spec = LAYOUTS[source_layout]
        self.src_dim: int = int(spec["src_dim"])
        self.blocks: List[Block] = spec["blocks"]
        self.keys = list(keys)
        self._validate()

        mask = torch.zeros(UNIFIED_DIM, dtype=torch.bool)
        for (_ss, _se), (ts, te) in self.blocks:
            mask[ts:te] = True
        self._slot_mask = mask

    def _validate(self) -> None:
        covered_src: List[int] = []
        covered_tgt: List[int] = []
        for (ss, se), (ts, te) in self.blocks:
            assert 0 <= ss < se <= self.src_dim, f"{self.source_layout}: bad src block ({ss},{se}) for src_dim {self.src_dim}"
            assert 0 <= ts < te <= UNIFIED_DIM, f"{self.source_layout}: bad tgt block ({ts},{te})"
            assert se - ss == te - ts, f"{self.source_layout}: block length mismatch {(ss, se)} vs {(ts, te)}"
            covered_src.extend(range(ss, se))
            covered_tgt.extend(range(ts, te))
        # Source dims may be intentionally DROPPED (a source whose padding half
        # carries no unique content, as the legacy egodex28_eef layout did with
        # EgoDex's masked joint half). So full source coverage is NOT required;
        # we only forbid mapping any src or tgt dim twice.
        assert len(set(covered_src)) == len(covered_src), f"{self.source_layout}: source blocks overlap"
        assert len(set(covered_tgt)) == len(covered_tgt), f"{self.source_layout}: target blocks overlap"

    # --- core ops -------------------------------------------------------------
    def _place(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.src_dim:
            raise ValueError(f"{self.source_layout}: expected native last dim {self.src_dim}, got {x.shape[-1]}")
        out = torch.zeros(x.shape[:-1] + (UNIFIED_DIM,), dtype=x.dtype, device=x.device)
        for (ss, se), (ts, te) in self.blocks:
            out[..., ts:te] = x[..., ss:se]
        return out

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != UNIFIED_DIM:
            raise ValueError(f"{self.source_layout}: backward expects {UNIFIED_DIM}D, got {x.shape[-1]}")
        out = torch.zeros(x.shape[:-1] + (self.src_dim,), dtype=x.dtype, device=x.device)
        for (ss, se), (ts, te) in self.blocks:
            out[..., ss:se] = x[..., ts:te]
        return out

    def _mask_like(self, x: torch.Tensor) -> torch.Tensor:
        return self._slot_mask.to(device=x.device).expand(x.shape[:-1] + (UNIFIED_DIM,)).contiguous()

    def _apply_forward(self, batch: Dict, field: str, mask_field: str) -> None:
        if field not in batch:
            return
        placed = {k: self._place(batch[field][k]) for k in self.keys}
        masks = {k: self._mask_like(placed[k]) for k in self.keys}
        batch[field] = {**batch[field], **placed}
        batch[mask_field] = {**batch.get(mask_field, {}), **masks}

    # --- transform interface --------------------------------------------------
    def forward(self, batch: Dict) -> Dict:
        self._apply_forward(batch, "action", "action_mask")
        self._apply_forward(batch, "state", "state_mask")
        return batch

    def backward(self, batch: Dict) -> Dict:
        # Inverse scatter -> gather native dims back out of the 32D vector
        # (deploy/eval postprocess). Mask fields are left untouched.
        for field in ("action", "state"):
            if field not in batch:
                continue
            batch[field] = {**batch[field], **{k: self._gather(batch[field][k]) for k in self.keys}}
        return batch
