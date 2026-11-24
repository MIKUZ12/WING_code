"""Mix N per-group datasets (each a RobotVideoDataset with its own processor +
stats) into one weighted dataset for heterogeneous 32D co-training.

This is a reusable dataset primitive; no maintained official mixed-embodiment
training recipe is currently shipped.
  - Each group keeps its OWN processor (vision concat/size + transforms + per-group
    normalization stats) -> per-group normalization is automatic, no grouped
    normalizer surgery.
  - Each sample is tagged with an integer `group_id`: in per_embodiment head mode it is
    the embodiment_id that routes each sample to its own native-dim head/encoder; in shared
    head mode the model ignores it and it is used only at eval/deploy to pick the group's
    un-normalizer.
  - Weighting is baked into the index space via integer oversampling, so the
    existing resumable ResumableEpochSampler's uniform shuffle yields the target
    group mixture -- no custom sampler, resume/distributed behavior unchanged.
  - All groups must emit collatable samples (one shared video DiT): identical
    `pixel_values` shape, and `action`/`proprio` last dim == unified_dim. Checked
    once per group at init (fail fast on "length right, semantics wrong").
"""
import bisect
from typing import Optional, Sequence

import torch

from wing.utils.logging_config import get_logger

logger = get_logger(__name__)


class MultiEmbodimentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        weights: Optional[Sequence[float]] = None,
        group_names: Optional[Sequence[str]] = None,
        unified_dim: int = 32,
        assert_shapes: bool = True,
    ):
        if len(datasets) < 1:
            raise ValueError("MultiEmbodimentDataset needs >= 1 group")
        self.datasets = list(datasets)
        n = len(self.datasets)
        self.group_names = list(group_names) if group_names is not None else [f"group{i}" for i in range(n)]
        if len(self.group_names) != n:
            raise ValueError(f"group_names length {len(self.group_names)} != #datasets {n}")
        self.unified_dim = int(unified_dim)

        self.lengths = [len(d) for d in self.datasets]
        for nm, L in zip(self.group_names, self.lengths):
            if L <= 0:
                raise ValueError(f"group {nm!r} is empty (len={L})")

        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError(f"weights length {len(weights)} != #datasets {n}")
        w = [float(x) for x in weights]
        if any(x <= 0 for x in w):
            raise ValueError(f"weights must be > 0, got {w}")
        self.weights = w

        # Target fraction f_g = w_g / sum(w). Want repeats r_g s.t. r_g * L_g is
        # proportional to f_g (so a uniform draw over the flat index space lands in
        # group g with probability ~f_g). r_g = round((f_g/L_g) / min_k(f_k/L_k)),
        # clamped to >= 1 (the lowest-ratio group repeats once, others oversample).
        # If weights are proportional to sizes, all r_g == 1 (natural concat).
        s = sum(w)
        ratios = [(w[i] / s) / self.lengths[i] for i in range(n)]
        rmin = min(ratios)
        self.repeats = [max(1, int(round(ratios[i] / rmin))) for i in range(n)]

        self._seg_sizes = [self.repeats[i] * self.lengths[i] for i in range(n)]
        self._cum = []
        acc = 0
        for sz in self._seg_sizes:
            acc += sz
            self._cum.append(acc)
        self._total = acc

        realized = [self._seg_sizes[i] / self._total for i in range(n)]
        logger.info(
            "MultiEmbodimentDataset: groups=%s lengths=%s weights=%s repeats=%s "
            "realized_fraction=%s total=%d",
            self.group_names, self.lengths, self.weights, self.repeats,
            [round(x, 3) for x in realized], self._total,
        )

        if assert_shapes:
            self._assert_collatable()

    def _assert_collatable(self) -> None:
        ref_shape = None
        ref_group = None
        ref_video_key = None
        for i, d in enumerate(self.datasets):
            s = d[0]
            video_keys = [key for key in ("video", "video_latents") if key in s]
            if len(video_keys) != 1:
                raise ValueError(
                    f"group {self.group_names[i]!r} must contain exactly one of video/video_latents"
                )
            video_key = video_keys[0]
            vid_shape = tuple(s[video_key].shape)
            if ref_shape is None:
                ref_shape, ref_group, ref_video_key = vid_shape, self.group_names[i], video_key
            elif vid_shape != ref_shape or video_key != ref_video_key:
                raise ValueError(
                    f"group {self.group_names[i]!r} {video_key} {vid_shape} != group {ref_group!r} "
                    f"{ref_video_key} {ref_shape}: all groups must share one video shape and input representation."
                )
            for key in ("action", "proprio"):
                if key in s and s[key].shape[-1] != self.unified_dim:
                    raise ValueError(
                        f"group {self.group_names[i]!r} {key} last dim {s[key].shape[-1]} "
                        f"!= unified_dim {self.unified_dim}"
                    )

    def __len__(self) -> int:
        return self._total

    def _route(self, idx: int):
        if idx < 0 or idx >= self._total:
            raise IndexError(f"index {idx} out of range [0, {self._total})")
        g = bisect.bisect_right(self._cum, idx)
        seg_start = self._cum[g - 1] if g > 0 else 0
        local = (idx - seg_start) % self.lengths[g]
        return g, local

    def __getitem__(self, idx: int):
        g, local = self._route(idx)
        sample = self.datasets[g][local]
        sample["group_id"] = torch.tensor(g, dtype=torch.long)
        return sample
