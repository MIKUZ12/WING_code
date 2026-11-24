"""Current-only spectral latent action-plan components for phase-aware WING training.

The teacher trajectory stays in fixed WING-LAM coefficient coordinates.  The
policy predicts normalized low-frequency coefficients from deployable inputs,
then projects them to ActionDiT conditioning tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def orthonormal_dct_matrix(length: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return an orthonormal DCT-II matrix with frequency-indexed rows."""
    if length <= 0:
        raise ValueError("DCT length must be positive.")
    n = torch.arange(length, device=device, dtype=torch.float32)
    k = n[:, None]
    matrix = torch.cos(math.pi / float(length) * (n[None, :] + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / float(length))
    if length > 1:
        matrix[1:] *= math.sqrt(2.0 / float(length))
    return matrix.to(dtype=dtype or torch.float32)


class SpectralPlanNormalizer(nn.Module):
    """Frozen train-split mean/std for temporal latent action trajectory coefficients."""

    def __init__(self, *, horizon: int = 8, latent_dim: int = 32, low_frequencies: int = 4) -> None:
        super().__init__()
        if not 0 < low_frequencies <= horizon:
            raise ValueError("low_frequencies must be in [1, horizon].")
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.low_frequencies = int(low_frequencies)
        self.register_buffer("dct", orthonormal_dct_matrix(horizon))
        self.register_buffer("mean", torch.zeros(low_frequencies, latent_dim))
        self.register_buffer("std", torch.ones(low_frequencies, latent_dim))

    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (self.low_frequencies, self.latent_dim)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError(f"Spectral statistics must have shape {expected}.")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("Spectral statistics must be finite.")
        if bool((std <= 0).any()):
            raise ValueError("Spectral standard deviations must be positive.")
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.std.copy_(std.to(device=self.std.device, dtype=self.std.dtype))

    def coefficients(self, trajectory: torch.Tensor) -> torch.Tensor:
        expected_tail = (self.horizon, self.latent_dim)
        if trajectory.ndim != 3 or tuple(trajectory.shape[1:]) != expected_tail:
            raise ValueError(f"latent action trajectory must be [B,{self.horizon},{self.latent_dim}].")
        dct = self.dct.to(device=trajectory.device, dtype=trajectory.dtype)
        return torch.einsum("kh,bhd->bkd", dct, trajectory)[:, : self.low_frequencies]

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(trajectory)
        return (coefficients - self.mean.to(coefficients)) / (self.std.to(coefficients) + 1.0e-6)


class _PlanQueryBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        delta, _ = self.attn(
            self.query_norm(query), self.context_norm(context), self.context_norm(context),
            key_padding_mask=~context_mask.bool(), need_weights=False,
        )
        query = query + delta
        return query + self.mlp(self.mlp_norm(query))


class CurrentSpectralPlanPredictor(nn.Module):
    """Predict normalized `[B,K,D_latent]` coefficients using current-only context."""

    def __init__(
        self, *, visual_dim: int, text_dim: int, plan_dim: int, latent_dim: int,
        num_queries: int, num_heads: int = 8, depth: int = 1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(int(visual_dim), int(plan_dim))
        self.text_proj = nn.Linear(int(text_dim), int(plan_dim))
        self.queries = nn.Parameter(torch.randn(int(num_queries), int(plan_dim)) * 0.02)
        self.blocks = nn.ModuleList([_PlanQueryBlock(int(plan_dim), int(num_heads)) for _ in range(int(depth))])
        self.out = nn.Linear(int(plan_dim), int(latent_dim))

    def forward(
        self, *, current_visual_tokens: torch.Tensor, instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor, proprio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_visual_tokens.ndim != 3 or instruction_tokens.ndim != 3:
            raise ValueError("Current visual and instruction tokens must be rank 3.")
        if instruction_mask.shape != instruction_tokens.shape[:2]:
            raise ValueError("Instruction mask must align with instruction tokens.")
        visual = self.visual_proj(current_visual_tokens)
        text = self.text_proj(instruction_tokens)
        context = torch.cat([visual, text], dim=1)
        mask = torch.cat([
            torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool), instruction_mask.bool(),
        ], dim=1)
        if proprio_tokens is not None:
            context = torch.cat([context, self.text_proj(proprio_tokens)], dim=1)
            mask = torch.cat([mask, torch.ones(proprio_tokens.shape[:2], device=mask.device, dtype=torch.bool)], dim=1)
        query = self.queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, context, mask)
        return self.out(query)


class PlanTokenProjection(nn.Module):
    """Learned conversion from fixed coefficient coordinates to ActionDiT space."""

    def __init__(self, latent_dim: int, dit_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(latent_dim), int(dit_dim))

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 3 or coefficients.shape[-1] != self.proj.in_features:
            raise ValueError("Expected plan coefficients [B,K,D_latent].")
        return self.proj(coefficients)


class PlanConditionAdapter(nn.Module):
    """Cross-attend an ActionDiT hidden sequence to predicted plan tokens."""

    def __init__(self, dim: int, num_heads: int, init_gate: float = 0.05) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.plan_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def _prepare_plan_tokens(
        self,
        action_hidden: torch.Tensor,
        plan_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the fixed K0-K3 plan tokens used as adapter K/V."""
        if action_hidden.ndim != 3:
            raise ValueError("Action hidden must be rank 3.")
        if plan_tokens.ndim != 3:
            raise ValueError("Plan tokens must be rank 3.")
        if (
            action_hidden.shape[0] != plan_tokens.shape[0]
            or action_hidden.shape[-1] != plan_tokens.shape[-1]
        ):
            raise ValueError("Plan tokens must match ActionDiT batch and hidden dimensions.")
        # Keep the historical fixed-token computation and state layout. In
        # particular, ``self.gate`` remains named for checkpoint compatibility;
        # its public meaning is the adapter residual scale.
        return self.plan_norm(plan_tokens)

    def forward(
        self,
        action_hidden: torch.Tensor,
        plan_tokens: torch.Tensor,
    ) -> torch.Tensor:
        plan = self._prepare_plan_tokens(action_hidden, plan_tokens)
        delta, _ = self.attn(self.query_norm(action_hidden), plan, plan, need_weights=False)
        return action_hidden + self.gate.to(dtype=action_hidden.dtype) * delta


def spectral_plan_alignment_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, valid_mask: Optional[torch.Tensor] = None,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """MSE plus optional cosine loss in fixed normalized coefficient space."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Plan prediction and target must have matching [B,K,D_latent] shape.")
    target = target.detach()
    error = F.mse_loss(prediction.float(), target.float(), reduction="none")
    if valid_mask is None:
        mse = error.mean()
        cosine = (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()
    else:
        if valid_mask.shape != prediction.shape[:2]:
            raise ValueError("Plan valid mask must be [B,K].")
        weight = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
        mse = (error * weight).sum() / (weight.sum().clamp(min=1.0) * prediction.shape[-1])
        cosine_per_token = 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
        cosine_weight_mask = valid_mask.to(device=cosine_per_token.device, dtype=cosine_per_token.dtype)
        cosine = (cosine_per_token * cosine_weight_mask).sum() / cosine_weight_mask.sum().clamp(min=1.0)
    return mse + float(cosine_weight) * cosine


def intervene_plan_tokens(
    plan_tokens: torch.Tensor, *, mode: str = "predicted", donor_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return predicted, null, or externally supplied shuffled plan tokens."""
    mode = str(mode).strip().lower()
    if mode == "predicted":
        return plan_tokens
    if mode == "null":
        return torch.zeros_like(plan_tokens)
    if mode == "shuffle":
        if donor_tokens is None or donor_tokens.shape != plan_tokens.shape:
            raise ValueError("Shuffle intervention needs same-shape donor plan tokens.")
        return donor_tokens.to(device=plan_tokens.device, dtype=plan_tokens.dtype)
    raise ValueError("Plan intervention must be predicted, null, or shuffle.")


@dataclass(frozen=True)
class SpectralPlanDiagnostics:
    coefficient_variance: float
    coefficient_effective_rank: float
    batch_size: int


def spectral_plan_diagnostics(coefficients: torch.Tensor) -> SpectralPlanDiagnostics:
    flat = coefficients.detach().float().reshape(coefficients.shape[0], -1)
    if flat.shape[0] < 2:
        return SpectralPlanDiagnostics(0.0, 0.0, int(flat.shape[0]))
    variance = float(flat.var(dim=0, unbiased=False).mean().item())
    singular = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True))
    probability = singular.square() / singular.square().sum().clamp(min=1.0e-12)
    rank = float(torch.exp(-(probability * probability.clamp(min=1.0e-12).log()).sum()).item())
    return SpectralPlanDiagnostics(variance, rank, int(flat.shape[0]))
