"""RGB-only student for trajectory-assisted WING-LAM interaction codes.

The teacher uses tracks, masks, and canonical residuals.  This student only
accepts two RGB frames at inference and is trained against frozen teacher
``z_int`` targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DreamDojoFramePairBackbone(nn.Module):
    """Expose DreamDojo visual tokens without trajectory-side inputs."""

    def __init__(self, lam: nn.Module) -> None:
        super().__init__()
        for name in ("encoder", "action_prompt", "patch_size", "model_dim"):
            if not hasattr(lam, name):
                raise TypeError(f"LAM is missing {name!r}")
        self.encoder = lam.encoder
        self.action_prompt = lam.action_prompt
        self.patch_size = int(lam.patch_size)
        self.output_dim = int(lam.model_dim)

    def set_trainable(self, *, last_blocks: int = 4) -> None:
        blocks = self.encoder.transformer_blocks
        if not 0 <= last_blocks <= len(blocks):
            raise ValueError(f"last_blocks must be in [0,{len(blocks)}]")
        self.action_prompt.requires_grad_(False)
        self.encoder.requires_grad_(False)
        for block in blocks[len(blocks) - last_blocks :]:
            block.requires_grad_(True)

    def load_wing_lam_visual_state(self, checkpoint: str | Path) -> None:
        """Strictly initialize from the teacher's fine-tuned visual state."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        source = payload.get("model", payload)
        if "action_prompt" not in source:
            raise KeyError("WING-LAM checkpoint has no action_prompt")
        prefix = "visual_encoder."
        mapped = {"action_prompt": source["action_prompt"]}
        mapped.update(
            {
                f"encoder.{key[len(prefix):]}": value
                for key, value in source.items()
                if key.startswith(prefix)
            }
        )
        result = self.load_state_dict(mapped, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"incomplete WING-LAM visual initialization: {result}")

    def _patchify(self, videos: Tensor) -> Tensor:
        if videos.ndim != 5 or videos.shape[1] != 2 or videos.shape[-1] != 3:
            raise ValueError(f"videos must have shape [B,2,H,W,3], got {tuple(videos.shape)}")
        batch, time, height, width, channels = videos.shape
        patch = self.patch_size
        patch_height, patch_width = height // patch, width // patch
        if patch_height == 0 or patch_width == 0:
            raise ValueError("video resolution is smaller than one patch")
        cropped = videos[:, :, : patch_height * patch, : patch_width * patch]
        patches = cropped.reshape(
            batch, time, patch_height, patch, patch_width, patch, channels
        )
        return patches.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            batch,
            time,
            patch_height * patch_width,
            patch * patch * channels,
        )

    def forward(self, videos: Tensor) -> Tensor:
        patches = self._patchify(videos)
        action_pad = self.action_prompt.expand(videos.shape[0], 2, -1, -1)
        features = self.encoder(torch.cat([action_pad, patches], dim=2))[:, :, 1:]
        return features.flatten(1, 2)


class RGBLatentActionStudent(nn.Module):
    """Map two RGB frames to one continuous latent action."""

    def __init__(self, backbone: nn.Module, *, feature_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.backbone = backbone
        self.latent_dim = int(latent_dim)
        self.head = nn.Linear(feature_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, videos: Tensor) -> Tensor:
        tokens = self.backbone(videos)
        if tokens.ndim != 3 or tokens.shape[0] != videos.shape[0]:
            raise ValueError("backbone must return [B,N,D] visual tokens")
        return self.head(tokens.mean(dim=1))

    @staticmethod
    def relation_code(raw_latent: Tensor, epsilon: float = 1e-6) -> Tensor:
        return F.normalize(raw_latent, dim=-1, eps=epsilon)


def build_rgb_student(
    *,
    lam_checkpoint: str | Path,
    teacher_checkpoint: str | Path | None = None,
    latent_dim: int = 32,
) -> RGBLatentActionStudent:
    """Build the student architecture and optionally initialize teacher visuals."""
    from .source_lam import load_lam

    lam = load_lam(str(lam_checkpoint), device="cpu")
    backbone = DreamDojoFramePairBackbone(lam)
    del lam
    if teacher_checkpoint is not None:
        backbone.load_wing_lam_visual_state(teacher_checkpoint)
    return RGBLatentActionStudent(
        backbone,
        feature_dim=backbone.output_dim,
        latent_dim=latent_dim,
    )


def load_rgb_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    lam_checkpoint: str | Path | None = None,
) -> tuple[RGBLatentActionStudent, dict[str, Any]]:
    """Strict-load a student, optionally relocating its architecture checkpoint.

    Older student payloads record the absolute LAM path from the training
    machine.  ``lam_checkpoint`` changes only the local architecture source;
    the student's complete trained state is still strict-loaded afterwards.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    metadata = dict(payload["metadata"])
    resolved_lam_checkpoint = (
        Path(lam_checkpoint) if lam_checkpoint is not None else Path(metadata["lam_checkpoint"])
    )
    if not resolved_lam_checkpoint.is_file():
        raise FileNotFoundError(
            f"LAM architecture checkpoint not found: {resolved_lam_checkpoint}. "
            "Pass lam_checkpoint=... to relocate a student trained on another machine."
        )
    model = build_rgb_student(
        lam_checkpoint=resolved_lam_checkpoint,
        latent_dim=int(metadata.get("latent_dim", 32)),
    )
    model.load_state_dict(payload["model"], strict=True)
    metadata["resolved_lam_checkpoint"] = str(resolved_lam_checkpoint.resolve())
    return model.to(device), metadata
