"""Dual-factor latent action model built on a DreamDojo visual encoder.

The model does not claim causal or 3D disentanglement.  It learns a
camera-dominant affine/homography factor and an interaction-dominant residual
factor from frozen trajectory/hand-mask supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


GlobalModel = Literal["affine", "homography"]


@dataclass(frozen=True)
class WINGLAMLossWeights:
    track: float = 1.0
    global_motion: float = 1.0
    residual: float = 1.0
    cross_compose: float = 1.0
    interaction_invariance: float = 0.1
    kl_camera: float = 1e-6
    kl_interaction: float = 1e-6
    interaction_variance: float = 0.0
    interaction_covariance: float = 0.0
    interaction_std_target: float = 0.1
    active_residual_alpha: float = 0.0
    active_residual_epsilon: float = 1e-3
    active_residual_threshold_scale: float = 2.0
    active_residual_cap: float = 0.05


def transform_points(points: Tensor, matrix: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Apply batched homogeneous transforms to normalized ``[B,K,2]`` points."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points must have shape [B,K,2], got {tuple(points.shape)}")
    if matrix.shape != (points.shape[0], 3, 3):
        raise ValueError(f"matrix must have shape [B,3,3], got {tuple(matrix.shape)}")
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    transformed = torch.einsum("bij,bkj->bki", matrix, homogeneous)
    denominator = transformed[..., 2:3]
    safe_sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
    denominator = torch.where(denominator.abs() < epsilon, safe_sign * epsilon, denominator)
    return transformed[..., :2] / denominator


def compose_points(
    points0: Tensor,
    matrix: Tensor,
    canonical_residual: Tensor,
    interaction: Tensor,
) -> Tensor:
    """Compose ``W(p0 + M_int * r_int)`` in the source-canonical convention."""
    if canonical_residual.shape != points0.shape:
        raise ValueError("canonical_residual must match points0")
    if interaction.shape != points0.shape[:2]:
        raise ValueError("interaction must match points0[:2]")
    return transform_points(points0 + canonical_residual * interaction[..., None], matrix)


class MaskedPointPool(nn.Module):
    """Small gated weighted-mean pool with explicit empty-set behavior."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor, mask: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        if features.shape[:2] != mask.shape or mask.shape != weights.shape:
            raise ValueError("features, mask and weights must share [B,K]")
        encoded = self.point_mlp(features)
        effective = mask.to(encoded.dtype) * weights.to(encoded.dtype).clamp_min(0)
        effective = effective * torch.sigmoid(self.gate(encoded).squeeze(-1))
        denominator = effective.sum(dim=1, keepdim=True)
        pooled = (encoded * effective[..., None]).sum(dim=1) / denominator.clamp_min(1e-8)
        available = denominator.squeeze(1) > 0
        return pooled, available


class CoordinateResidualDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_frequencies: int = 4) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        coordinate_dim = 2 + 4 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(latent_dim + coordinate_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _encode_coordinates(self, points: Tensor) -> Tensor:
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies,
            device=points.device,
            dtype=points.dtype,
        )
        angles = 2.0 * torch.pi * points[..., None] * frequencies
        return torch.cat([points, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)

    def forward(self, points: Tensor, latent: Tensor) -> Tensor:
        expanded_latent = latent[:, None, :].expand(-1, points.shape[1], -1)
        return self.net(torch.cat([self._encode_coordinates(points), expanded_latent], dim=-1))


class WINGLAM(nn.Module):
    """Camera/interaction factorization using a pretrained DreamDojo encoder.

    ``lam`` is an already strict-loaded DreamDojo ``LatentActionModel``.  Only
    its action prompt and spatiotemporal encoder are registered here; the old
    RGB decoder is intentionally not part of the trajectory objective.
    """

    def __init__(
        self,
        lam: nn.Module,
        *,
        camera_latent_dim: int = 16,
        interaction_latent_dim: int = 32,
        pool_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        global_model: GlobalModel = "affine",
    ) -> None:
        super().__init__()
        if global_model not in ("affine", "homography"):
            raise ValueError(f"unsupported global_model {global_model!r}")
        for attribute in ("encoder", "action_prompt", "patch_size", "model_dim"):
            if not hasattr(lam, attribute):
                raise TypeError(f"lam is missing required attribute {attribute!r}")
        self.visual_encoder = lam.encoder
        self.action_prompt = lam.action_prompt
        self.patch_size = int(lam.patch_size)
        self.model_dim = int(lam.model_dim)
        self.camera_latent_dim = int(camera_latent_dim)
        self.interaction_latent_dim = int(interaction_latent_dim)
        self.global_model = global_model

        point_input_dim = 2 * self.model_dim + 9
        self.camera_pool = MaskedPointPool(point_input_dim, pool_hidden_dim)
        self.interaction_pool = MaskedPointPool(point_input_dim, pool_hidden_dim)
        self.camera_head = nn.Linear(pool_hidden_dim, 2 * self.camera_latent_dim)
        self.interaction_head = nn.Linear(pool_hidden_dim, 2 * self.interaction_latent_dim)

        matrix_parameters = 6 if global_model == "affine" else 8
        self.camera_decoder = nn.Sequential(
            nn.Linear(self.camera_latent_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, matrix_parameters),
        )
        nn.init.normal_(self.camera_decoder[-1].weight, std=1e-4)
        nn.init.zeros_(self.camera_decoder[-1].bias)
        self.interaction_decoder = CoordinateResidualDecoder(
            self.interaction_latent_dim,
            decoder_hidden_dim,
        )
        self.set_backbone_trainable(last_blocks=0)

    def set_backbone_trainable(self, *, last_blocks: int = 0) -> None:
        """Freeze the backbone, optionally unfreezing its last N temporal blocks."""
        blocks = self.visual_encoder.transformer_blocks
        if not 0 <= last_blocks <= len(blocks):
            raise ValueError(f"last_blocks must be in [0,{len(blocks)}], got {last_blocks}")
        self.action_prompt.requires_grad_(False)
        self.visual_encoder.requires_grad_(False)
        for block in blocks[len(blocks) - last_blocks :]:
            block.requires_grad_(True)

    def _patchify(self, videos: Tensor) -> tuple[Tensor, int, int]:
        if videos.ndim != 5 or videos.shape[1] != 2 or videos.shape[-1] != 3:
            raise ValueError(f"videos must have shape [B,2,H,W,3], got {tuple(videos.shape)}")
        batch, time, height, width, channels = videos.shape
        patch = self.patch_size
        patch_height, patch_width = height // patch, width // patch
        if patch_height == 0 or patch_width == 0:
            raise ValueError("video resolution is smaller than one patch")
        videos = videos[:, :, : patch_height * patch, : patch_width * patch]
        patches = videos.reshape(batch, time, patch_height, patch, patch_width, patch, channels)
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            batch,
            time,
            patch_height * patch_width,
            patch * patch * channels,
        )
        return patches, patch_height, patch_width

    def encode_visual(self, videos: Tensor) -> Tensor:
        patches, patch_height, patch_width = self._patchify(videos)
        action_pad = self.action_prompt.expand(videos.shape[0], 2, -1, -1)
        features = self.visual_encoder(torch.cat([action_pad, patches], dim=2))[:, :, 1:]
        return features.reshape(
            videos.shape[0],
            2,
            patch_height,
            patch_width,
            self.model_dim,
        )

    @staticmethod
    def _sample_frame_features(feature_map: Tensor, points: Tensor) -> Tensor:
        grid = points.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            feature_map.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def _moments(head: nn.Module, pooled: Tensor, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = head(pooled).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-12.0, 8.0)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return latent, mean, log_variance

    def encode_factors(
        self,
        *,
        videos: Tensor,
        points0: Tensor,
        points1: Tensor,
        canonical_residual: Tensor,
        valid: Tensor,
        background: Tensor,
        interaction: Tensor,
        visibility: Tensor,
        confidence: Tensor,
        sample_latents: bool | None = None,
    ) -> dict[str, Tensor]:
        if points0.shape != points1.shape or points0.shape != canonical_residual.shape:
            raise ValueError("points0, points1 and canonical_residual must share [B,K,2]")
        if valid.shape != points0.shape[:2]:
            raise ValueError("point masks must match points0[:2]")
        finite = (
            torch.all(torch.isfinite(points0), dim=-1)
            & torch.all(torch.isfinite(points1), dim=-1)
            & torch.all(torch.isfinite(canonical_residual), dim=-1)
        )
        effective_valid = valid & finite
        safe_points0 = torch.where(effective_valid[..., None], points0, torch.zeros_like(points0))
        safe_points1 = torch.where(effective_valid[..., None], points1, torch.zeros_like(points1))
        safe_residual = torch.where(
            effective_valid[..., None],
            canonical_residual,
            torch.zeros_like(canonical_residual),
        )
        safe_points0 = torch.nan_to_num(safe_points0)
        safe_points1 = torch.nan_to_num(safe_points1)
        safe_residual = torch.nan_to_num(safe_residual)
        visual = self.encode_visual(videos)
        feature0 = self._sample_frame_features(visual[:, 0], safe_points0)
        feature1 = self._sample_frame_features(visual[:, 1], safe_points1)
        reliability = visibility.to(visual.dtype).clamp(0, 1) * confidence.to(visual.dtype).clamp(0, 1)

        camera_aux = torch.cat(
            [
                safe_points0,
                safe_points1,
                safe_points1 - safe_points0,
                confidence[..., None],
                visibility[..., None],
                background.to(points0.dtype)[..., None],
            ],
            dim=-1,
        )
        canonical_endpoint = safe_points0 + safe_residual
        interaction_aux = torch.cat(
            [
                safe_points0,
                canonical_endpoint,
                safe_residual,
                confidence[..., None],
                visibility[..., None],
                interaction.to(points0.dtype)[..., None],
            ],
            dim=-1,
        )
        paired_visual = torch.cat([feature0, feature1], dim=-1)
        camera_pooled, camera_available = self.camera_pool(
            torch.cat([paired_visual, camera_aux.to(paired_visual.dtype)], dim=-1),
            effective_valid & background,
            reliability,
        )
        interaction_pooled, interaction_available = self.interaction_pool(
            torch.cat([paired_visual, interaction_aux.to(paired_visual.dtype)], dim=-1),
            effective_valid & interaction,
            reliability,
        )
        sample = self.training if sample_latents is None else sample_latents
        z_camera, mu_camera, logvar_camera = self._moments(self.camera_head, camera_pooled, sample)
        z_interaction, mu_interaction, logvar_interaction = self._moments(
            self.interaction_head,
            interaction_pooled,
            sample,
        )
        return {
            "z_camera": z_camera,
            "mu_camera": mu_camera,
            "logvar_camera": logvar_camera,
            "camera_available": camera_available,
            "z_interaction": z_interaction,
            "mu_interaction": mu_interaction,
            "logvar_interaction": logvar_interaction,
            "interaction_available": interaction_available,
        }

    def decode_camera(self, latent: Tensor) -> Tensor:
        parameters = self.camera_decoder(latent)
        batch = latent.shape[0]
        identity = torch.eye(3, device=latent.device, dtype=latent.dtype).expand(batch, -1, -1).clone()
        if self.global_model == "affine":
            identity[:, :2] += parameters.reshape(batch, 2, 3)
        else:
            identity[:, 0, 0] += parameters[:, 0]
            identity[:, 0, 1] += parameters[:, 1]
            identity[:, 0, 2] += parameters[:, 2]
            identity[:, 1, 0] += parameters[:, 3]
            identity[:, 1, 1] += parameters[:, 4]
            identity[:, 1, 2] += parameters[:, 5]
            identity[:, 2, 0] += parameters[:, 6]
            identity[:, 2, 1] += parameters[:, 7]
        return identity

    def decode(
        self,
        points0: Tensor,
        interaction: Tensor,
        z_camera: Tensor,
        z_interaction: Tensor,
    ) -> dict[str, Tensor]:
        matrix = self.decode_camera(z_camera)
        residual = self.interaction_decoder(points0, z_interaction)
        endpoint = compose_points(points0, matrix, residual, interaction)
        return {"matrix": matrix, "canonical_residual": residual, "endpoint": endpoint}

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        sample_latents: bool | None = None,
        compute_cross: bool = True,
    ) -> dict[str, Any]:
        if "view_videos" in batch:
            if compute_cross:
                raise ValueError("cross composition is not defined for generic multi-view training")
            return self._forward_multiview(batch, sample_latents=sample_latents)
        valid_points0 = batch["valid"] & torch.all(torch.isfinite(batch["points0"]), dim=-1)
        points0 = torch.where(
            valid_points0[..., None],
            batch["points0"],
            torch.zeros_like(batch["points0"]),
        )
        points0 = torch.nan_to_num(points0)
        original = self.encode_factors(
            videos=batch["videos"],
            points0=batch["points0"],
            points1=batch["points1"],
            canonical_residual=batch["canonical_residual"],
            valid=batch["valid"],
            background=batch["background"],
            interaction=batch["interaction"],
            visibility=batch["visibility"],
            confidence=batch["confidence"],
            sample_latents=sample_latents,
        )
        result: dict[str, Any] = {
            "original_factors": original,
            "original_prediction": self.decode(
                points0,
                batch["interaction"],
                original["z_camera"],
                original["z_interaction"],
            ),
        }
        if "videos_augmented" in batch:
            valid_augmented = batch.get("valid_augmented", batch["valid"])
            augmented = self.encode_factors(
                videos=batch["videos_augmented"],
                points0=batch["points0"],
                points1=batch["points1_augmented"],
                canonical_residual=batch["canonical_residual"],
                valid=valid_augmented,
                background=batch["background"],
                interaction=batch["interaction"],
                visibility=batch["visibility"],
                confidence=batch["confidence"],
                sample_latents=sample_latents,
            )
            result.update(
                {
                    "augmented_factors": augmented,
                    "augmented_prediction": self.decode(
                        points0,
                        batch["interaction"],
                        augmented["z_camera"],
                        augmented["z_interaction"],
                    ),
                }
            )
            if compute_cross:
                result.update(
                    {
                        "cross_to_augmented": self.decode(
                            points0,
                            batch["interaction"],
                            augmented["z_camera"],
                            original["z_interaction"],
                        ),
                        "cross_to_original": self.decode(
                            points0,
                            batch["interaction"],
                            original["z_camera"],
                            augmented["z_interaction"],
                        ),
                    }
                )
        return result

    def _forward_multiview(
        self,
        batch: dict[str, Tensor],
        *,
        sample_latents: bool | None,
    ) -> dict[str, Any]:
        if batch["view_videos"].ndim != 6 or batch["view_points1"].ndim != 4:
            raise ValueError("multi-view videos/points must have shapes [B,V,2,H,W,3]/[B,V,K,2]")
        view_count = batch["view_videos"].shape[1]
        if view_count < 2 or batch["view_points1"].shape[1] != view_count:
            raise ValueError("multi-view training requires at least two aligned views")
        valid_points0 = batch["view_valid"][:, 0] & torch.all(
            torch.isfinite(batch["points0"]), dim=-1
        )
        points0 = torch.where(
            valid_points0[..., None],
            batch["points0"],
            torch.zeros_like(batch["points0"]),
        )
        points0 = torch.nan_to_num(points0)
        factors = []
        predictions = []
        for view in range(view_count):
            current = self.encode_factors(
                videos=batch["view_videos"][:, view],
                points0=batch["points0"],
                points1=batch["view_points1"][:, view],
                canonical_residual=batch["canonical_residual"],
                valid=batch["view_valid"][:, view],
                background=batch["background"],
                interaction=batch["interaction"],
                visibility=batch["visibility"],
                confidence=batch["confidence"],
                sample_latents=sample_latents,
            )
            factors.append(current)
            predictions.append(
                self.decode(
                    points0,
                    batch["interaction"],
                    current["z_camera"],
                    current["z_interaction"],
                )
            )
        return {"view_factors": factors, "view_predictions": predictions}


def _weighted_huber(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(prediction.dtype).clamp_min(0)
    finite = torch.all(torch.isfinite(prediction) & torch.isfinite(target), dim=-1)
    weights = weights * finite.to(weights.dtype)
    active = weights > 0
    safe_prediction = torch.where(active[..., None], prediction, torch.zeros_like(prediction))
    safe_target = torch.where(active[..., None], target, torch.zeros_like(target))
    loss = F.smooth_l1_loss(safe_prediction, safe_target, reduction="none", beta=0.01).mean(dim=-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _kl_loss(mean: Tensor, log_variance: Tensor, available: Tensor) -> Tensor:
    per_sample = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean(dim=-1)
    weights = available.to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def active_residual_signal(
    batch: dict[str, Tensor],
    weights: WINGLAMLossWeights,
) -> Tensor:
    """Return residual magnitude above each sample's robust background noise floor."""
    residual = torch.nan_to_num(batch["canonical_residual"])
    magnitude = torch.linalg.norm(residual, dim=-1)
    signals = []
    for index in range(len(magnitude)):
        background = batch["valid"][index] & batch["background"][index]
        background = background & torch.isfinite(magnitude[index])
        if torch.any(background):
            values = magnitude[index][background]
            median = values.median()
            mad = (values - median).abs().median()
            threshold = median + weights.active_residual_threshold_scale * 1.4826 * mad
        else:
            threshold = magnitude.new_zeros(())
        signals.append(
            (magnitude[index] - threshold)
            .clamp_min(0.0)
            .clamp_max(weights.active_residual_cap)
        )
    return torch.stack(signals)


def _active_residual_multiplier(
    batch: dict[str, Tensor],
    weights: WINGLAMLossWeights,
) -> Tensor:
    """Return robust per-point weights emphasizing motion above background noise."""
    if weights.active_residual_alpha <= 0:
        return torch.ones_like(batch["visibility"])
    signal = active_residual_signal(batch, weights)
    return (weights.active_residual_epsilon + signal).pow(weights.active_residual_alpha)


def _latent_spread_loss(
    mean: Tensor,
    available: Tensor,
    target_std: float,
) -> tuple[Tensor, Tensor]:
    selected = mean[available]
    zero = mean.new_zeros(())
    if len(selected) < 2:
        return zero, zero
    centered = selected - selected.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
    variance_loss = F.relu(target_std - standard_deviation).square().mean()
    normalized = centered / standard_deviation.clamp_min(1e-4)
    covariance = normalized.T @ normalized / len(selected)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / max(mean.shape[1] * (mean.shape[1] - 1), 1)
    return variance_loss, covariance_loss


def lam_loss(
    batch: dict[str, Tensor],
    outputs: dict[str, Any],
    weights: WINGLAMLossWeights = WINGLAMLossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute self-reconstruction and paired cross-composition objectives."""
    if "view_predictions" in outputs:
        return _lam_multiview_loss(batch, outputs, weights)
    reliability = batch["visibility"].clamp(0, 1) * batch["confidence"].clamp(0, 1)
    valid = batch["valid"].to(reliability.dtype)
    background = batch["background"].to(reliability.dtype)
    interaction = batch["interaction"].to(reliability.dtype)
    fit_confidence = batch["fit_confidence"].to(reliability.dtype).reshape(-1, 1)
    target_camera = transform_points(batch["points0"], batch["global_matrix"])
    original_prediction = outputs["original_prediction"]
    original_factors = outputs["original_factors"]
    residual_multiplier = _active_residual_multiplier(batch, weights)
    residual_fit_weight = (
        fit_confidence if weights.active_residual_alpha > 0 else torch.ones_like(fit_confidence)
    )

    loss_global = _weighted_huber(
        transform_points(batch["points0"], original_prediction["matrix"]),
        target_camera,
        reliability * valid * background * fit_confidence,
    )
    loss_residual = _weighted_huber(
        original_prediction["canonical_residual"],
        batch["canonical_residual"],
        reliability * valid * interaction * residual_multiplier * residual_fit_weight,
    )
    loss_track = _weighted_huber(
        original_prediction["endpoint"],
        batch["points1"],
        reliability * valid,
    )
    loss_cross = loss_track.new_zeros(())
    loss_invariance = loss_track.new_zeros(())

    if "augmented_prediction" in outputs:
        valid_augmented = batch.get("valid_augmented", batch["valid"]).to(reliability.dtype)
        target_camera_augmented = transform_points(batch["points0"], batch["global_matrix_augmented"])
        augmented_prediction = outputs["augmented_prediction"]
        augmented_factors = outputs["augmented_factors"]
        loss_global = 0.5 * (
            loss_global
            + _weighted_huber(
                transform_points(batch["points0"], augmented_prediction["matrix"]),
                target_camera_augmented,
                reliability * valid_augmented * background * fit_confidence,
            )
        )
        loss_residual = 0.5 * (
            loss_residual
            + _weighted_huber(
                augmented_prediction["canonical_residual"],
                batch["canonical_residual"],
                reliability
                * valid_augmented
                * interaction
                * residual_multiplier
                * residual_fit_weight,
            )
        )
        loss_track = 0.5 * (
            loss_track
            + _weighted_huber(
                augmented_prediction["endpoint"],
                batch["points1_augmented"],
                reliability * valid_augmented,
            )
        )
        if "cross_to_augmented" in outputs:
            loss_cross = 0.5 * (
                _weighted_huber(
                    outputs["cross_to_augmented"]["endpoint"],
                    batch["points1_augmented"],
                    reliability * valid_augmented,
                )
                + _weighted_huber(
                    outputs["cross_to_original"]["endpoint"],
                    batch["points1"],
                    reliability * valid,
                )
            )
        invariant_samples = (
            original_factors["interaction_available"] & augmented_factors["interaction_available"]
        ).to(reliability.dtype)
        invariant_samples = invariant_samples * batch.get(
            "orbit_weight",
            torch.ones_like(invariant_samples),
        ).to(reliability.dtype)
        invariant_error = F.smooth_l1_loss(
            original_factors["mu_interaction"],
            augmented_factors["mu_interaction"],
            reduction="none",
            beta=0.1,
        ).mean(dim=-1)
        loss_invariance = (invariant_error * invariant_samples).sum() / invariant_samples.sum().clamp_min(1.0)
        loss_kl_camera = 0.5 * (
            _kl_loss(
                original_factors["mu_camera"],
                original_factors["logvar_camera"],
                original_factors["camera_available"],
            )
            + _kl_loss(
                augmented_factors["mu_camera"],
                augmented_factors["logvar_camera"],
                augmented_factors["camera_available"],
            )
        )
        loss_kl_interaction = 0.5 * (
            _kl_loss(
                original_factors["mu_interaction"],
                original_factors["logvar_interaction"],
                original_factors["interaction_available"],
            )
            + _kl_loss(
                augmented_factors["mu_interaction"],
                augmented_factors["logvar_interaction"],
                augmented_factors["interaction_available"],
            )
        )
    else:
        loss_kl_camera = _kl_loss(
            original_factors["mu_camera"],
            original_factors["logvar_camera"],
            original_factors["camera_available"],
        )
        loss_kl_interaction = _kl_loss(
            original_factors["mu_interaction"],
            original_factors["logvar_interaction"],
            original_factors["interaction_available"],
        )

    spread_mean = original_factors["mu_interaction"]
    spread_available = original_factors["interaction_available"]
    if "augmented_factors" in outputs:
        spread_mean = torch.cat(
            [spread_mean, outputs["augmented_factors"]["mu_interaction"]],
            dim=0,
        )
        spread_available = torch.cat(
            [spread_available, outputs["augmented_factors"]["interaction_available"]],
            dim=0,
        )
    loss_interaction_variance, loss_interaction_covariance = _latent_spread_loss(
        spread_mean,
        spread_available,
        weights.interaction_std_target,
    )

    components = {
        "track": loss_track,
        "global_motion": loss_global,
        "residual": loss_residual,
        "cross_compose": loss_cross,
        "interaction_invariance": loss_invariance,
        "kl_camera": loss_kl_camera,
        "kl_interaction": loss_kl_interaction,
        "interaction_variance": loss_interaction_variance,
        "interaction_covariance": loss_interaction_covariance,
    }
    total = (
        weights.track * loss_track
        + weights.global_motion * loss_global
        + weights.residual * loss_residual
        + weights.cross_compose * loss_cross
        + weights.interaction_invariance * loss_invariance
        + weights.kl_camera * loss_kl_camera
        + weights.kl_interaction * loss_kl_interaction
        + weights.interaction_variance * loss_interaction_variance
        + weights.interaction_covariance * loss_interaction_covariance
    )
    return total, components


def _lam_multiview_loss(
    batch: dict[str, Tensor],
    outputs: dict[str, Any],
    weights: WINGLAMLossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average reconstruction over aligned views and invariance over every view pair."""
    factors = outputs["view_factors"]
    predictions = outputs["view_predictions"]
    if len(factors) != len(predictions) or len(factors) < 2:
        raise ValueError("multi-view outputs must contain at least two aligned views")
    reliability = batch["visibility"].clamp(0, 1) * batch["confidence"].clamp(0, 1)
    background = batch["background"].to(reliability.dtype)
    interaction = batch["interaction"].to(reliability.dtype)
    residual_multiplier = _active_residual_multiplier(batch, weights)
    track_losses = []
    global_losses = []
    residual_losses = []
    kl_camera_losses = []
    kl_interaction_losses = []
    for view, (factor, prediction) in enumerate(zip(factors, predictions, strict=True)):
        valid = batch["view_valid"][:, view].to(reliability.dtype)
        fit_confidence = batch["view_fit_confidence"][:, view].to(reliability.dtype).reshape(-1, 1)
        target_camera = transform_points(batch["points0"], batch["view_global_matrix"][:, view])
        global_losses.append(
            _weighted_huber(
                transform_points(batch["points0"], prediction["matrix"]),
                target_camera,
                reliability * valid * background * fit_confidence,
            )
        )
        residual_fit = fit_confidence if weights.active_residual_alpha > 0 else 1.0
        residual_losses.append(
            _weighted_huber(
                prediction["canonical_residual"],
                batch["canonical_residual"],
                reliability * valid * interaction * residual_multiplier * residual_fit,
            )
        )
        track_losses.append(
            _weighted_huber(
                prediction["endpoint"],
                batch["view_points1"][:, view],
                reliability * valid,
            )
        )
        kl_camera_losses.append(
            _kl_loss(factor["mu_camera"], factor["logvar_camera"], factor["camera_available"])
        )
        kl_interaction_losses.append(
            _kl_loss(
                factor["mu_interaction"],
                factor["logvar_interaction"],
                factor["interaction_available"],
            )
        )
    loss_track = torch.stack(track_losses).mean()
    loss_global = torch.stack(global_losses).mean()
    loss_residual = torch.stack(residual_losses).mean()
    loss_kl_camera = torch.stack(kl_camera_losses).mean()
    loss_kl_interaction = torch.stack(kl_interaction_losses).mean()

    pair_losses = []
    for first in range(len(factors)):
        for second in range(first + 1, len(factors)):
            available = (
                factors[first]["interaction_available"]
                & factors[second]["interaction_available"]
            ).to(reliability.dtype)
            available = available * batch["view_orbit_weight"].to(reliability.dtype)
            error = F.smooth_l1_loss(
                factors[first]["mu_interaction"],
                factors[second]["mu_interaction"],
                reduction="none",
                beta=0.1,
            ).mean(dim=-1)
            pair_losses.append((error * available).sum() / available.sum().clamp_min(1.0))
    loss_invariance = torch.stack(pair_losses).mean()

    spread_mean = torch.cat([factor["mu_interaction"] for factor in factors], dim=0)
    spread_available = torch.cat([factor["interaction_available"] for factor in factors], dim=0)
    loss_variance, loss_covariance = _latent_spread_loss(
        spread_mean,
        spread_available,
        weights.interaction_std_target,
    )
    zero = loss_track.new_zeros(())
    components = {
        "track": loss_track,
        "global_motion": loss_global,
        "residual": loss_residual,
        "cross_compose": zero,
        "interaction_invariance": loss_invariance,
        "kl_camera": loss_kl_camera,
        "kl_interaction": loss_kl_interaction,
        "interaction_variance": loss_variance,
        "interaction_covariance": loss_covariance,
    }
    total = (
        weights.track * loss_track
        + weights.global_motion * loss_global
        + weights.residual * loss_residual
        + weights.interaction_invariance * loss_invariance
        + weights.kl_camera * loss_kl_camera
        + weights.kl_interaction * loss_kl_interaction
        + weights.interaction_variance * loss_variance
        + weights.interaction_covariance * loss_covariance
    )
    return total, components
