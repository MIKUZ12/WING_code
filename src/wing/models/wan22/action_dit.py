import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence

from wing.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    # Canonical task-specific endpoint namespaces. Partial resume derives its fully-qualified
    # `mixtures.action.*` classifier from this single definition; everything else is backbone.
    ACTION_ENDPOINT_PREFIXES = ("action_encoder.", "head.", "action_encoders.", "heads.")
    # Backward-compatible alias for ActionDiT.from_pretrained's historical backbone-only load.
    ACTION_BACKBONE_SKIP_PREFIXES = ACTION_ENDPOINT_PREFIXES
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        gradient_checkpointing_interval: int = 1,
        action_head_mode: str = "shared",
        num_embodiments: int = 1,
        action_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        # Action-head mode: "shared" = one Linear over the (padded) `action_dim`;
        # "per_embodiment" = one native-dim Linear per source, picked by embodiment_id.
        # `action_dim` is the collated tensor width (= max native dim) in both modes.
        self.action_head_mode = str(action_head_mode)
        if self.action_head_mode not in ("shared", "per_embodiment"):
            raise ValueError(
                f"`action_head_mode` must be 'shared' or 'per_embodiment', got {self.action_head_mode!r}"
            )
        self.num_embodiments = int(num_embodiments)
        if self.action_head_mode == "per_embodiment":
            if action_dims is None or len(action_dims) != self.num_embodiments:
                raise ValueError(
                    "`per_embodiment` action head needs `action_dims` with one native dim per "
                    f"embodiment (num_embodiments={self.num_embodiments}), got {action_dims}"
                )
            self.action_dims = [int(d) for d in action_dims]
            if max(self.action_dims) > action_dim:
                raise ValueError(
                    f"`action_dim` ({action_dim}) must be >= max native dim ({max(self.action_dims)})."
                )
        else:
            self.action_dims = None

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        if self.action_head_mode == "per_embodiment":
            self.action_encoders = nn.ModuleList([nn.Linear(d, hidden_dim) for d in self.action_dims])
        else:
            self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        if self.action_head_mode == "per_embodiment":
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, d) for d in self.action_dims])
        else:
            self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.gradient_checkpointing_interval = int(gradient_checkpointing_interval)
        if self.gradient_checkpointing_interval < 1:
            raise ValueError("`gradient_checkpointing_interval` must be >= 1.")
        if self.use_gradient_checkpointing:
            checkpointed_blocks = (
                num_layers + self.gradient_checkpointing_interval - 1
            ) // self.gradient_checkpointing_interval
            logger.info(
                "Using gradient checkpointing for %d/%d ActionDiT blocks "
                "(interval=%d). This will save memory but use more computation.",
                checkpointed_blocks,
                num_layers,
                self.gradient_checkpointing_interval,
            )
        # Non-persistent provenance for fail-closed fresh Phase-II phase_init.
        # This is deliberately absent from state_dict so native fixed resume
        # keeps the existing model and optimizer layout unchanged.
        self.pretrained_source_video_checkpoint: Optional[dict[str, Any]] = None

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(
                f"`meta` must be a mapping in {action_dit_pretrained_path}, "
                f"got {type(meta).__name__}"
            )
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        source_video_checkpoint = meta.get("source_video_checkpoint")
        if source_video_checkpoint is not None and not isinstance(
            source_video_checkpoint, Mapping
        ):
            raise ValueError(
                f"`meta.source_video_checkpoint` must be a mapping in "
                f"{action_dit_pretrained_path}, got "
                f"{type(source_video_checkpoint).__name__}"
            )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        action_expert.pretrained_source_video_checkpoint = (
            None
            if source_video_checkpoint is None
            else dict(source_video_checkpoint)
        )
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def _encode_action_grouped(self, action_tokens: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """per_embodiment action encoder: each source's rows go through its own native-dim
        Linear (reading only the first `action_dims[c]` dims of the padded input). Returns
        the uniform hidden-dim token tensor [B, T, hidden_dim]."""
        if embodiment_id is None:
            raise ValueError("`embodiment_id` is required when `action_head_mode='per_embodiment'`.")
        batch_size, seq_len = action_tokens.shape[0], action_tokens.shape[1]
        tokens = action_tokens.new_zeros(batch_size, seq_len, self.hidden_dim)
        for c in torch.unique(embodiment_id).tolist():
            idx = (embodiment_id == c).nonzero(as_tuple=True)[0]
            enc = self.action_encoders[c](action_tokens.index_select(0, idx)[:, :, : self.action_dims[c]])
            tokens = tokens.index_copy(0, idx, enc.to(tokens.dtype))
        return tokens

    def _decode_action_grouped(self, tokens: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """per_embodiment action head: each source's rows go through its own native-dim
        Linear; outputs are right-padded back to `action_dim` (the collated width) so the
        result is one [B, T, action_dim] tensor. The pad region is zero and never supervised
        (the loss slices each source's native dims)."""
        if embodiment_id is None:
            raise ValueError("`embodiment_id` is required when `action_head_mode='per_embodiment'`.")
        batch_size, seq_len = tokens.shape[0], tokens.shape[1]
        out = tokens.new_zeros(batch_size, seq_len, self.action_dim)
        for c in torch.unique(embodiment_id).tolist():
            idx = (embodiment_id == c).nonzero(as_tuple=True)[0]
            pred = self.heads[c](tokens.index_select(0, idx))
            pad = self.action_dim - self.action_dims[c]
            if pad > 0:
                pred = F.pad(pred, (0, pad))
            out = out.index_copy(0, idx, pred.to(out.dtype))
        return out

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        if self.action_head_mode == "per_embodiment":
            tokens = self._encode_action_grouped(action_tokens, embodiment_id)
        else:
            tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Dict[str, Any],
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_head_mode == "per_embodiment":
            return self._decode_action_grouped(tokens, embodiment_id)
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            embodiment_id=embodiment_id,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block_index, block in enumerate(self.blocks):
            checkpoint_block = (
                self.use_gradient_checkpointing
                and block_index % self.gradient_checkpointing_interval == 0
            )
            if checkpoint_block:
                x = gradient_checkpoint_forward(
                    block,
                    True,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state, embodiment_id=embodiment_id)
