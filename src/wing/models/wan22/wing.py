from collections.abc import Mapping
import os
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from wing.utils.logging_config import get_logger
from .helpers.phase1_checkpoint_import import import_phase1_planner_backbone

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .spectral_plan import (
    CurrentSpectralPlanPredictor,
    PlanTokenProjection,
    SpectralPlanNormalizer,
    intervene_plan_tokens,
    spectral_plan_alignment_loss,
    spectral_plan_diagnostics,
)

logger = get_logger(__name__)

_MOT_ACTION_PREFIX = "mixtures.action."
_ACTION_ENDPOINT_PREFIXES = tuple(
    f"{_MOT_ACTION_PREFIX}{prefix}" for prefix in ActionDiT.ACTION_ENDPOINT_PREFIXES
)
_PARTIAL_ENDPOINT_POLICIES = frozenset({"reset", "load_exact"})
TRAINING_PHASE_PHASE1 = "phase1_ego_planner"
TRAINING_PHASE_POLICY = "policy"
TRAINING_PHASES = frozenset({TRAINING_PHASE_PHASE1, TRAINING_PHASE_POLICY})


def normalize_training_phase(value: Any) -> Optional[str]:
    """Normalize the opt-in two-phase training mode while preserving legacy ``None``."""
    if value is None:
        return None
    phase = str(value).strip().lower()
    if phase in {"", "none", "null"}:
        return None
    if phase == "phase2_robot_grounding":
        return TRAINING_PHASE_POLICY
    if phase not in TRAINING_PHASES:
        raise ValueError(
            "training_phase must be null, phase1_ego_planner, or "
            f"policy; got {value!r}."
        )
    return phase


def _debug_nonfinite_enabled() -> bool:
    """Return whether expensive NaN/Inf checks should run inside the training forward."""
    return os.environ.get("WING_DEBUG_NONFINITE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_tensor_nonfinite_report(name: str, tensor: torch.Tensor) -> str | None:
    """Build a short diagnostic string for tensors that contain NaN or Inf."""
    if not torch.is_tensor(tensor) or not (tensor.is_floating_point() or tensor.is_complex()):
        return None
    data = tensor.detach()
    if data.is_complex():
        data = data.abs()
    finite_mask = torch.isfinite(data)
    if bool(finite_mask.all().item()):
        return None
    finite_values = data[finite_mask]
    nonfinite = int((~finite_mask).sum().item())
    total = int(data.numel())
    if finite_values.numel() > 0:
        finite_summary = (
            f"finite_min={float(finite_values.min().float().item()):.6g} "
            f"finite_max={float(finite_values.max().float().item()):.6g} "
            f"finite_mean={float(finite_values.float().mean().item()):.6g}"
        )
    else:
        finite_summary = "no_finite_values"
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nonfinite={nonfinite}/{total} {finite_summary}"
    )


def _debug_assert_finite(stage: str, **values: Any) -> None:
    """Raise immediately when debug mode finds NaN/Inf in key forward tensors."""
    if not _debug_nonfinite_enabled():
        return
    reports = []
    for name, value in values.items():
        if isinstance(value, Mapping):
            for key, item in value.items():
                report = _debug_tensor_nonfinite_report(f"{name}.{key}", item)
                if report is not None:
                    reports.append(report)
        else:
            report = _debug_tensor_nonfinite_report(name, value)
            if report is not None:
                reports.append(report)
    if reports:
        raise RuntimeError(
            f"WING_DEBUG_NONFINITE detected non-finite tensor at {stage}:\n"
            + "\n".join(reports[:12])
        )


def _require_tensor_mapping(state, *, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} must be a state-dict mapping, got {type(state).__name__}.")
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} contains a non-string key: {key!r}.")
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"{label}[{key!r}] must be a tensor, got {type(value).__name__}."
            )
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(
                f"{label}[{key!r}] uses unsupported tensor storage: "
                f"layout={value.layout}, quantized={value.is_quantized}."
            )
    return state


def _require_finite(tensor: torch.Tensor, *, label: str) -> None:
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise ValueError(f"{label} contains NaN or Inf.")


def _validate_linear_state(
    state: Mapping[str, torch.Tensor],
    *,
    weight_key: str,
    bias_key: str,
    label: str,
) -> tuple[int, int]:
    weight = state[weight_key]
    bias = state[bias_key]
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
        raise ValueError(
            f"Malformed {label}: expected Linear weight[out,in] and bias[out], got "
            f"weight={tuple(weight.shape)} bias={tuple(bias.shape)}."
        )
    return int(weight.shape[1]), int(weight.shape[0])


def _validate_action_endpoint_topology(
    state: Mapping[str, torch.Tensor], *, label: str
) -> tuple[str, frozenset[str]]:
    """Validate a complete shared or per-embodiment action endpoint before classifying it."""
    endpoint_keys = frozenset(
        key for key in state if any(key.startswith(prefix) for prefix in _ACTION_ENDPOINT_PREFIXES)
    )
    if not endpoint_keys:
        raise ValueError(f"{label} contains no canonical action encoder/head endpoint.")

    local_keys = {key.removeprefix(_MOT_ACTION_PREFIX) for key in endpoint_keys}
    has_shared = any(
        key.startswith("action_encoder.") or key.startswith("head.") for key in local_keys
    )
    has_per = any(
        key.startswith("action_encoders.") or key.startswith("heads.") for key in local_keys
    )
    if has_shared and has_per:
        raise ValueError(f"{label} mixes shared and per-embodiment action endpoints.")

    if has_shared:
        expected_local = {
            "action_encoder.weight",
            "action_encoder.bias",
            "head.weight",
            "head.bias",
        }
        if local_keys != expected_local:
            raise ValueError(
                f"Malformed shared {label}: expected={sorted(expected_local)} "
                f"actual={sorted(local_keys)}."
            )
        prefix = _MOT_ACTION_PREFIX
        action_dim, hidden_from_encoder = _validate_linear_state(
            state,
            weight_key=f"{prefix}action_encoder.weight",
            bias_key=f"{prefix}action_encoder.bias",
            label=f"{label}.action_encoder",
        )
        hidden_from_head, output_dim = _validate_linear_state(
            state,
            weight_key=f"{prefix}head.weight",
            bias_key=f"{prefix}head.bias",
            label=f"{label}.head",
        )
        if action_dim != output_dim or hidden_from_encoder != hidden_from_head:
            raise ValueError(
                f"Malformed shared {label}: encoder/head dimensions are not an atomic pair "
                f"(encoder in/out={action_dim}/{hidden_from_encoder}, "
                f"head in/out={hidden_from_head}/{output_dim})."
            )
        return "shared", endpoint_keys

    parsed: dict[str, dict[int, set[str]]] = {
        "action_encoders": {},
        "heads": {},
    }
    for local_key in local_keys:
        parts = local_key.split(".")
        if len(parts) != 3 or parts[0] not in parsed or parts[2] not in {"weight", "bias"}:
            raise ValueError(f"Malformed per-embodiment {label} key: {local_key!r}.")
        index_text = parts[1]
        if not index_text.isdigit() or str(int(index_text)) != index_text:
            raise ValueError(f"Non-canonical group index in {label}: {local_key!r}.")
        parsed[parts[0]].setdefault(int(index_text), set()).add(parts[2])

    encoder_indices = set(parsed["action_encoders"])
    head_indices = set(parsed["heads"])
    expected_indices = set(range(len(encoder_indices)))
    if (
        not encoder_indices
        or encoder_indices != head_indices
        or encoder_indices != expected_indices
    ):
        raise ValueError(
            f"Malformed per-embodiment {label} groups: encoders={sorted(encoder_indices)} "
            f"heads={sorted(head_indices)} expected={sorted(expected_indices)}."
        )
    for family, groups in parsed.items():
        incomplete = {index: fields for index, fields in groups.items() if fields != {"weight", "bias"}}
        if incomplete:
            raise ValueError(f"Malformed {label} {family}: incomplete groups={incomplete}.")
    for index in sorted(encoder_indices):
        prefix = f"{_MOT_ACTION_PREFIX}action_encoders.{index}"
        action_dim, hidden_from_encoder = _validate_linear_state(
            state,
            weight_key=f"{prefix}.weight",
            bias_key=f"{prefix}.bias",
            label=prefix,
        )
        prefix = f"{_MOT_ACTION_PREFIX}heads.{index}"
        hidden_from_head, output_dim = _validate_linear_state(
            state,
            weight_key=f"{prefix}.weight",
            bias_key=f"{prefix}.bias",
            label=prefix,
        )
        if action_dim != output_dim or hidden_from_encoder != hidden_from_head:
            raise ValueError(
                f"Malformed {label} group {index}: encoder/head dimensions are not an atomic pair."
            )
    return "per_embodiment", endpoint_keys


def _validate_proprio_topology(
    state: Mapping[str, torch.Tensor], *, mode: str, label: str
) -> frozenset[str]:
    state = _require_tensor_mapping(state, label=label)
    keys = frozenset(state)
    if mode == "shared":
        expected = frozenset({"weight", "bias"})
        if keys != expected:
            raise ValueError(
                f"Malformed shared {label}: expected={sorted(expected)} actual={sorted(keys)}."
            )
        _validate_linear_state(
            state, weight_key="weight", bias_key="bias", label=label
        )
        return keys
    if mode != "per_embodiment":
        raise ValueError(f"Unsupported {label} mode: {mode!r}.")

    groups: dict[int, set[str]] = {}
    for key in keys:
        parts = key.split(".")
        if len(parts) != 2 or parts[1] not in {"weight", "bias"}:
            raise ValueError(f"Malformed per-embodiment {label} key: {key!r}.")
        index_text = parts[0]
        if not index_text.isdigit() or str(int(index_text)) != index_text:
            raise ValueError(f"Non-canonical group index in {label}: {key!r}.")
        groups.setdefault(int(index_text), set()).add(parts[1])
    expected_indices = set(range(len(groups)))
    if not groups or set(groups) != expected_indices:
        raise ValueError(
            f"Malformed per-embodiment {label} groups: actual={sorted(groups)} "
            f"expected={sorted(expected_indices)}."
        )
    for index, fields in groups.items():
        if fields != {"weight", "bias"}:
            raise ValueError(f"Malformed {label} group {index}: fields={sorted(fields)}.")
        _validate_linear_state(
            state,
            weight_key=f"{index}.weight",
            bias_key=f"{index}.bias",
            label=f"{label}.{index}",
        )
    return keys


def _validate_exact_transfer_tensor(
    source: torch.Tensor, target: torch.Tensor, *, label: str
) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"{label} shape mismatch: source={tuple(source.shape)} target={tuple(target.shape)}."
        )
    source_float = source.is_floating_point()
    target_float = target.is_floating_point()
    if source_float != target_float or source.is_complex() != target.is_complex():
        raise ValueError(
            f"{label} dtype-family mismatch: source={source.dtype} target={target.dtype}."
        )
    if not source_float and not source.is_complex() and source.dtype != target.dtype:
        raise ValueError(
            f"{label} non-floating dtype mismatch: source={source.dtype} target={target.dtype}."
        )
    _require_finite(source, label=f"{label} source")
    canonical = source.to(dtype=target.dtype)
    _require_finite(canonical, label=f"{label} canonical cast")


def _preflight_exact_state_transfer(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    """Validate an exact module state without mutating the target module."""
    source = _require_tensor_mapping(source_state, label=f"checkpoint {label}")
    target = _require_tensor_mapping(target_state, label=f"target {label}")
    if set(source) != set(target):
        raise ValueError(
            f"{label} key-set mismatch: "
            f"missing={sorted(set(target) - set(source))[:10]} "
            f"unexpected={sorted(set(source) - set(target))[:10]}."
        )
    for key in sorted(target):
        _validate_exact_transfer_tensor(
            source[key], target[key], label=f"{label}.{key}"
        )
    return source


class WING(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        proprio_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        # Action-head mode mirrors the action expert (single source of truth). In
        # per_embodiment mode the proprio encoder also becomes one native-dim Linear per
        # source; `proprio_dim` stays the collated width (= max native dim) used for
        # input validation.
        self.action_head_mode = self.action_expert.action_head_mode
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.action_head_mode == "per_embodiment" and self.proprio_dim is not None:
            num_emb = self.action_expert.num_embodiments
            if proprio_dims is None or len(proprio_dims) != num_emb:
                raise ValueError(
                    "`per_embodiment` proprio needs `proprio_dims` with one native dim per "
                    f"embodiment (num_embodiments={num_emb}), got {proprio_dims}"
                )
            self.proprio_dims = [int(d) for d in proprio_dims]
            if max(self.proprio_dims) > self.proprio_dim:
                raise ValueError(
                    f"`proprio_dim` ({self.proprio_dim}) must be >= max native dim ({max(self.proprio_dims)})."
                )
            self.proprio_encoder = None
            self.proprio_encoders = nn.ModuleList(
                [nn.Linear(d, self.text_dim) for d in self.proprio_dims]
            ).to(torch_dtype)
        elif self.proprio_dim is not None:
            self.proprio_dims = None
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_dims = None
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        # Optional spectral-plan path.  It is configured after Wan construction so
        # default WING construction and checkpoints remain unchanged.
        self.spectral_plan_enabled = False
        self.spectral_plan_normalizer: Optional[SpectralPlanNormalizer] = None
        self.spectral_plan_predictor: Optional[CurrentSpectralPlanPredictor] = None
        self.spectral_plan_token_projection: Optional[PlanTokenProjection] = None
        self.spectral_plan_lambda = 0.0
        self.spectral_plan_cosine_weight = 0.0
        self.spectral_plan_training_mode = "joint"
        self.spectral_plan_detach_backbone = True
        self.spectral_plan_routing_mode = "fixed"
        # Opt-in orchestration only. ``None`` preserves every historical path.
        self.training_phase: Optional[str] = None

        self.to(self.device)

    def configure_training_phase(self, value: Any) -> None:
        self.training_phase = normalize_training_phase(value)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        dit_checkpoint_path: str | None = None,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        proprio_dims: Optional[Sequence[int]] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        mot_gradient_checkpointing_interval: int = 1,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for WING.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for WING.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            dit_checkpoint_path=dit_checkpoint_path,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            mot_gradient_checkpointing_interval=mot_gradient_checkpointing_interval,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            proprio_dims=proprio_dims,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def configure_spectral_plan(self, options: Optional[Mapping[str, Any]]) -> None:
        """Enable the canonical fixed Full-K4 latent-action condition.

        ``training_mode=joint`` augments native full-video WING training;
        ``training_mode=current_only`` retains the deployment-aligned ablation.
        ``None``/``enabled=false`` restores the exact native WING path.
        """
        options = {} if options is None else dict(options)
        enabled = bool(options.get("enabled", False))
        self.spectral_plan_enabled = enabled
        if not enabled:
            self.spectral_plan_normalizer = None
            self.spectral_plan_predictor = None
            self.spectral_plan_token_projection = None
            self.spectral_plan_lambda = 0.0
            self.spectral_plan_cosine_weight = 0.0
            self.spectral_plan_training_mode = "joint"
            self.spectral_plan_detach_backbone = True
            self.spectral_plan_routing_mode = "fixed"
            self.mot.configure_plan_adapters(enabled=False)
            return
        training_mode = str(options.get("training_mode", "joint")).strip().lower()
        if training_mode not in {"joint", "current_only"}:
            raise ValueError(
                "spectral_plan.training_mode must be one of ['joint', 'current_only'], "
                f"got {training_mode!r}."
            )
        self.spectral_plan_training_mode = training_mode
        self.spectral_plan_detach_backbone = bool(options.get("detach_backbone", True))
        routing_mode = str(options.get("routing_mode", "fixed")).strip().lower()
        if routing_mode != "fixed":
            raise ValueError(
                "Canonical spectral_plan.routing_mode must be 'fixed', "
                f"got {routing_mode!r}."
            )
        self.spectral_plan_routing_mode = routing_mode
        horizon = int(options.get("horizon", 8))
        latent_dim = int(options.get("latent_dim", 32))
        low_frequencies = int(options.get("low_frequencies", 4))
        if low_frequencies != 4:
            raise ValueError(
                "Canonical spectral planning requires fixed Full-K4 (low_frequencies=4)."
            )
        plan_dim = int(options.get("plan_dim", self.action_expert.hidden_dim))
        plan_heads = int(options.get("plan_heads", 8))
        plan_depth = int(options.get("plan_depth", 1))
        if plan_dim != int(self.action_expert.hidden_dim):
            raise ValueError("spectral plan_dim must equal ActionDiT hidden_dim.")
        self.spectral_plan_normalizer = SpectralPlanNormalizer(
            horizon=horizon, latent_dim=latent_dim, low_frequencies=low_frequencies
        )
        predictor_kwargs = {
            "visual_dim": int(self.video_expert.hidden_dim),
            "text_dim": self.text_dim,
            "plan_dim": plan_dim,
            "latent_dim": latent_dim,
            "num_queries": low_frequencies,
            "num_heads": plan_heads,
            "depth": plan_depth,
        }
        self.spectral_plan_predictor = CurrentSpectralPlanPredictor(**predictor_kwargs)
        self.spectral_plan_token_projection = PlanTokenProjection(latent_dim, plan_dim)
        self.spectral_plan_lambda = float(options.get("lambda_plan", 0.1))
        self.spectral_plan_cosine_weight = float(options.get("cosine_weight", 0.0))
        if self.spectral_plan_lambda < 0.0 or self.spectral_plan_cosine_weight < 0.0:
            raise ValueError("spectral loss weights must be non-negative.")
        default_layers = (self.mot.num_layers // 3 - 1, (2 * self.mot.num_layers) // 3 - 1)
        layer_indices = tuple(int(value) for value in options.get("adapter_layers", default_layers))
        if "adapter_init_scale" in options:
            adapter_init_scale = float(options["adapter_init_scale"])
            if (
                "init_gate" in options
                and float(options["init_gate"]) != adapter_init_scale
            ):
                raise ValueError(
                    "spectral_plan.adapter_init_scale and deprecated init_gate "
                    "must not disagree."
                )
        else:
            # Exact alias retained so native fixed checkpoints whose serialized
            # config predates the public-name cleanup can still be resumed.
            adapter_init_scale = float(options.get("init_gate", 0.05))
        self.mot.configure_plan_adapters(
            enabled=True, num_heads=int(options.get("adapter_heads", plan_heads)),
            layer_indices=layer_indices, adapter_init_scale=adapter_init_scale,
        )
        self.to(self.device, dtype=self.torch_dtype)

    def load_spectral_plan_statistics(self, path: str) -> None:
        if not self.spectral_plan_enabled or self.spectral_plan_normalizer is None:
            raise RuntimeError("Spectral plan is not enabled.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or "mean" not in payload or "std" not in payload:
            raise ValueError(f"Spectral statistics must contain mean/std: {path}")
        expected_metadata = {
            "horizon": self.spectral_plan_normalizer.horizon,
            "target_dim": self.spectral_plan_normalizer.latent_dim,
            "low_frequencies": self.spectral_plan_normalizer.low_frequencies,
        }
        for key, expected in expected_metadata.items():
            if key not in payload:
                raise ValueError(f"Spectral statistics are missing {key!r}: {path}")
            if int(payload[key]) != int(expected):
                raise ValueError(
                    f"Spectral statistics {key} mismatch: file={payload[key]}, model={expected}."
                )
        self.spectral_plan_normalizer.set_statistics(payload["mean"], payload["std"])

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _encode_proprio_grouped(self, proprio: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """per_embodiment proprio encoder: each source's row goes through its own native-dim
        Linear (reading only the first `proprio_dims[c]` dims). Returns [B, 1, text_dim]."""
        if embodiment_id is None:
            raise ValueError("`embodiment_id` is required for per_embodiment proprio encoding.")
        out = proprio.new_zeros(proprio.shape[0], 1, self.text_dim)
        for c in torch.unique(embodiment_id).tolist():
            idx = (embodiment_id == c).nonzero(as_tuple=True)[0]
            enc = self.proprio_encoders[c](proprio.index_select(0, idx)[:, : self.proprio_dims[c]].unsqueeze(1))
            out = out.index_copy(0, idx, enc.to(out.dtype))
        return out

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        embodiment_id: Optional[torch.Tensor] = None,
        proprio_available: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_disabled = self.proprio_encoder is None and getattr(self, "proprio_encoders", None) is None
        if proprio_disabled:
            return context, context_mask
        if proprio_available is None:
            proprio_available = torch.ones(
                context.shape[0], dtype=torch.bool, device=context.device
            )
        else:
            proprio_available = proprio_available.to(
                device=context.device, dtype=torch.bool
            ).reshape(-1)
            if proprio_available.shape[0] != context.shape[0]:
                raise ValueError(
                    "`proprio_available` batch mismatch: "
                    f"got {tuple(proprio_available.shape)} vs context batch {context.shape[0]}."
                )
        if proprio is None:
            if bool(proprio_available.any()):
                raise ValueError("Available proprio rows require a proprio tensor.")
            # The value is inert because the explicit availability mask is false.
            proprio_token = context.new_zeros(context.shape[0], 1, self.text_dim)
        else:
            if proprio.ndim != 2:
                raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
            if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=context.dtype)
            if self.action_head_mode == "per_embodiment":
                proprio_token = self._encode_proprio_grouped(proprio, embodiment_id).to(dtype=context.dtype)  # [B, 1, D]
            else:
                proprio_token = self.proprio_encoder(proprio.unsqueeze(1)).to(dtype=context.dtype)  # [B, 1, D]
        proprio_mask = proprio_available[:, None].to(device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @staticmethod
    def _normalize_proprio_availability(
        value: Any, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        availability = torch.as_tensor(value, device=device)
        if availability.dtype != torch.bool:
            raise TypeError("`proprio_available` must have boolean dtype.")
        if availability.ndim == 0:
            availability = availability.expand(batch_size)
        elif availability.ndim == 2 and availability.shape[1] == 1:
            availability = availability[:, 0]
        elif availability.ndim != 1:
            raise ValueError(
                "`proprio_available` must be scalar, [B], or [B,1], "
                f"got shape {tuple(availability.shape)}."
            )
        if availability.shape[0] != batch_size:
            raise ValueError(
                "`proprio_available` batch mismatch: "
                f"got {availability.shape[0]} vs expected {batch_size}."
            )
        return availability

    def _append_inference_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Append the available proprio token to the canonical fixed context."""
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask, None

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] < 1 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [B,3,H,W] or [3,H,W] with "
                f"B >= 1, got {tuple(input_image.shape)}"
            )
        images = input_image.to(device=self.device).unsqueeze(2)
        z = self.vae.encode(images, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = torch.stack(z)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(
        self,
        sample,
        tiled: bool = False,
        *,
        encode_current_video_only: bool = False,
        require_action: bool = True,
    ):
        video = sample.get("video")
        cached_latents = sample.get("video_latents")
        if (video is None) == (cached_latents is None):
            raise ValueError(
                "WING training requires exactly one of `sample['video']` or "
                "`sample['video_latents']`."
            )
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "WING training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        proprio_available_value = sample.get("proprio_available", None)
        if video is not None:
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[1] != 3:
                raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")
            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
                )
        else:
            if cached_latents.ndim != 5:
                raise ValueError(
                    "`sample['video_latents']` must be 5D [B,C,T,H,W], got shape "
                    f"{tuple(cached_latents.shape)}"
                )
            batch_size = int(cached_latents.shape[0])
            image_is_pad_for_shape = sample.get("image_is_pad")
            if image_is_pad_for_shape is None or image_is_pad_for_shape.ndim != 2:
                raise ValueError(
                    "Cached video latents require `sample['image_is_pad']` with shape [B,T_rgb] "
                    "to validate the action/video temporal contract."
                )
            num_frames = int(image_is_pad_for_shape.shape[1])
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        action = None
        action_is_pad = None
        action_mask = None
        if require_action:
            if "action" not in sample:
                raise ValueError("`sample['action']` is required for WING training.")

            action = sample["action"]
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            action_horizon = int(action.shape[1])
            if action_horizon % (num_frames - 1) != 0:
                raise ValueError(
                    f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
                )

            action_is_pad = sample.get("action_is_pad", None)
            if action_is_pad is not None:
                if action_is_pad.ndim != 2:
                    raise ValueError(
                        f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                    )
                if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                    raise ValueError(
                        "`sample['action_is_pad']` shape mismatch: "
                        f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                    )

            action_mask = sample.get("action_mask", None)
            if action_mask is not None:
                if action_mask.ndim != 3:
                    raise ValueError(
                        f"`sample['action_mask']` must be 3D [B, T, a_dim], got shape {tuple(action_mask.shape)}"
                    )
                if action_mask.shape != action.shape:
                    raise ValueError(
                        "`sample['action_mask']` shape mismatch: "
                        f"got {tuple(action_mask.shape)} vs action {tuple(action.shape)}"
                    )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        # per_embodiment routing label (= MultiEmbodimentDataset group_id). Shared mode ignores it.
        embodiment_id = sample.get("group_id", None) if require_action else None
        if require_action and self.action_head_mode == "per_embodiment":
            if embodiment_id is None:
                if self.action_expert.num_embodiments == 1:
                    embodiment_id = torch.zeros(batch_size, dtype=torch.long)
                else:
                    raise ValueError(
                        "`sample['group_id']` is required in per_embodiment mode with "
                        f"num_embodiments={self.action_expert.num_embodiments} (use `embodiment_datasets`)."
                    )
            embodiment_id = embodiment_id.reshape(-1).to(device=self.device, dtype=torch.long)
            if embodiment_id.shape[0] != batch_size:
                raise ValueError(
                    f"`group_id` batch {tuple(embodiment_id.shape)} != video batch {batch_size}"
                )
            if int(embodiment_id.max()) >= self.action_expert.num_embodiments or int(embodiment_id.min()) < 0:
                raise ValueError(
                    f"`group_id` out of range [0,{self.action_expert.num_embodiments}): "
                    f"min={int(embodiment_id.min())} max={int(embodiment_id.max())}"
                )
        else:
            embodiment_id = None

        if cached_latents is not None:
            input_latents = cached_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            temporal_factor = int(self.vae.temporal_downsample_factor)
            if temporal_factor <= 0:
                raise ValueError(
                    f"vae.temporal_downsample_factor must be positive, got {temporal_factor}."
                )
            expected_latent_frames = 1 + (num_frames - 1) // temporal_factor
            if input_latents.shape[2] != expected_latent_frames:
                raise ValueError(
                    "Cached VAE latent temporal shape mismatch: "
                    f"got T={input_latents.shape[2]}, expected {expected_latent_frames} "
                    f"from {num_frames} RGB frames and factor {temporal_factor}."
                )
            if encode_current_video_only:
                input_latents = input_latents[:, :, :1]
        else:
            # The current-only ablation keeps future RGB solely for the offline
            # target; joint/native training encodes the complete clip.
            video_for_policy = video[:, :, :1] if encode_current_video_only else video
            input_video = video_for_policy.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        predictor_proprio_available = None
        # proprio is enabled iff `proprio_dim` is set: shared mode -> single Linear `proprio_encoder`;
        # per_embodiment mode -> native-dim `proprio_encoders` ModuleList (and `proprio_encoder` is
        # None). Gate on `proprio_dim` (not `proprio_encoder`) so per_embodiment proprio is NOT
        # silently skipped; `_append_proprio_to_context` routes to the right encoder by mode.
        if self.proprio_dim is not None:
            proprio_available = WING._normalize_proprio_availability(
                proprio_available_value, batch_size=batch_size, device=context.device
            )
            if proprio is not None:
                if proprio.ndim != 3:
                    raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
                if proprio.shape[2] != self.proprio_dim:
                    raise ValueError(
                        f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                    )
                proprio = proprio[:, 0, :] # [B, D]
            elif proprio_available_value is None or bool(proprio_available.any()):
                raise ValueError(
                    "Missing `sample['proprio']` requires explicit all-false "
                    "`sample['proprio_available']`."
                )
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=(
                    None
                    if proprio is None
                    else proprio.to(device=self.device, dtype=self.torch_dtype)
                ),
                embodiment_id=embodiment_id,
                proprio_available=proprio_available,
            )
            predictor_proprio_available = proprio_available
        elif proprio_available_value is not None:
            predictor_proprio_available = WING._normalize_proprio_availability(
                proprio_available_value, batch_size=batch_size, device=context.device
            )
            if bool(predictor_proprio_available.any()):
                raise ValueError(
                    "`proprio_available=true` requires an enabled proprio encoder."
                )
        if action is not None:
            action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_mask is not None:
            action_mask = action_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "spectral_plan_proprio_available": predictor_proprio_available,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_mask": action_mask,
            "image_is_pad": image_is_pad,
            "embodiment_id": embodiment_id,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _training_loss_phase1_ego_planner(self, sample, tiled: bool = False):
        """Train VideoDiT and the deployable coefficient planner without ActionDiT."""
        if (
            self.spectral_plan_normalizer is None
            or not isinstance(self.spectral_plan_predictor, CurrentSpectralPlanPredictor)
        ):
            raise RuntimeError("Phase I requires the fixed Full-K4 planner and normalizer.")
        if "latent_target" not in sample:
            raise ValueError("Phase-I ego training requires sample['latent_target'].")
        if self.proprio_dim is not None:
            raise ValueError("Phase-I ego training requires proprio_dim=null.")
        if bool(getattr(self.video_expert, "action_conditioned", False)):
            raise ValueError("Phase-I video-only training requires an action-free VideoDiT.")
        patch_size = tuple(int(value) for value in getattr(self.video_expert, "patch_size", ()))
        if len(patch_size) < 1 or patch_size[0] != 1:
            raise ValueError("Phase-I current-token contract requires temporal patch_size=1.")
        if not bool(getattr(self.video_expert, "seperated_timestep", False)):
            raise ValueError("Phase-I current-token contract requires separated video timesteps.")

        inputs = self.build_inputs(sample, tiled=tiled, require_action=False)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            raise ValueError(
                "Phase-I planner requires fuse_vae_embedding_in_latents so its first-frame "
                "pre_dit tokens are clean and use timestep zero."
            )

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        latents[:, :, 0:1] = first_frame_latents

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        predictor_visual, predictor_context = self._spectral_predictor_inputs(
            video_pre["tokens"][:, :tokens_per_frame], inputs["context"]
        )
        coeff_pred = self._run_spectral_plan_predictor(
            current_visual_tokens=predictor_visual,
            instruction_tokens=predictor_context,
            instruction_mask=inputs["context_mask"],
        )
        if not isinstance(coeff_pred, torch.Tensor):
            raise RuntimeError("Phase-I coefficient-only planner returned an invalid payload.")

        pred_video = self.video_expert.forward_from_pre_dit(video_pre)
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_plan, plan_metrics = self._compute_spectral_plan_supervision(coeff_pred, sample)
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.spectral_plan_lambda * loss_plan
        )
        _debug_assert_finite(
            "phase1_loss",
            loss_video=loss_video,
            loss_plan=loss_plan,
            loss_total=loss_total,
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_plan": self.spectral_plan_lambda * float(loss_plan.detach().item()),
            **plan_metrics,
        }

    def _training_loss_spectral_plan(self, sample, tiled: bool = False):
        """Legacy current-only real-action FM plus alignment to an offline latent action trajectory.

        The video expert receives exactly one current-frame latent.  Future
        video and `latent_target` are never ActionDiT inputs; the latter appears
        only on the detached alignment-loss right-hand side.
        """
        if (
            self.spectral_plan_normalizer is None
            or self.spectral_plan_predictor is None
            or self.spectral_plan_token_projection is None
        ):
            raise RuntimeError("Spectral plan modules are not configured.")
        inputs = self.build_inputs(
            sample,
            tiled=tiled,
            encode_current_video_only=True,
        )
        if "latent_target" not in sample:
            raise ValueError("Current-only spectral training requires sample['latent_target'].")
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_mask = inputs["action_mask"]
        embodiment_id = inputs["embodiment_id"]
        batch_size = int(action.shape[0])
        if action_mask is not None:
            action = action * action_mask.to(device=action.device, dtype=action.dtype)

        noise_action = torch.randn_like(action)
        if action_mask is not None:
            noise_action = noise_action * action_mask.to(device=action.device, dtype=action.dtype)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        current_latents = inputs["input_latents"][:, :, :1]
        timestep_video = torch.zeros(batch_size, device=self.device, dtype=current_latents.dtype)
        video_pre = self.video_expert.pre_dit(
            x=current_latents, timestep=timestep_video,
            context=inputs["context"], context_mask=inputs["context_mask"], action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        predictor_visual, predictor_context = self._spectral_predictor_inputs(
            video_pre["tokens"], inputs["context"]
        )
        plan_condition = self._build_spectral_plan_condition(
            current_visual_tokens=predictor_visual,
            instruction_tokens=predictor_context,
            instruction_mask=inputs["context_mask"],
        )
        coeff_pred = plan_condition["coefficients"]
        plan_tokens = plan_condition["tokens"]
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=inputs["context"], context_mask=inputs["context_mask"],
            embodiment_id=embodiment_id,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1], action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]), device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            plan_tokens=plan_tokens,
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre, embodiment_id=embodiment_id)
        action_error = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        valid = torch.ones_like(action_error)
        if action_mask is not None:
            valid = valid * action_mask.to(device=valid.device, dtype=valid.dtype)
        if action_is_pad is not None:
            valid = valid * (~action_is_pad).to(device=valid.device, dtype=valid.dtype).unsqueeze(-1)
        if embodiment_id is not None:
            native_dims = torch.tensor(self.action_expert.action_dims, device=valid.device)
            dim_valid = torch.arange(valid.shape[-1], device=valid.device)[None] < native_dims[embodiment_id, None]
            valid = valid * dim_valid.unsqueeze(1).to(valid.dtype)
        per_sample = (action_error * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1.0)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(per_sample)
        loss_action = (per_sample * action_weight).mean()

        loss_plan, plan_metrics = self._compute_spectral_plan_supervision(coeff_pred, sample)
        loss_total = (
            self.loss_lambda_action * loss_action
            + self.spectral_plan_lambda * loss_plan
        )
        metrics = {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_plan": self.spectral_plan_lambda * float(loss_plan.detach().item()),
            **plan_metrics,
        }
        for index, adapter in self.mot.plan_adapters.items():
            metrics[f"plan_adapter_scale_{index}"] = float(adapter.gate.detach().item())
        return loss_total, metrics

    def _spectral_plan_target(
        self, coeff_pred: torch.Tensor, sample
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.spectral_plan_normalizer is None:
            raise RuntimeError("Spectral plan normalizer is not configured.")
        if "latent_target" not in sample:
            raise ValueError("Spectral training requires sample['latent_target'].")
        latent_target = sample["latent_target"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        with torch.no_grad():
            coeff_target = self.spectral_plan_normalizer(latent_target)
        pair_mask = sample.get("latent_target_mask")
        coefficient_mask = None
        if pair_mask is not None:
            pair_mask = pair_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if pair_mask.shape != latent_target.shape[:2]:
                raise ValueError("latent_target_mask must have shape [B,H_latent].")
            coefficient_mask = pair_mask.all(dim=1, keepdim=True).expand(-1, coeff_pred.shape[1])
        return coeff_target, coefficient_mask

    def _compute_spectral_plan_supervision(
        self, coeff_pred: torch.Tensor, sample
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coeff_target, coefficient_mask = self._spectral_plan_target(coeff_pred, sample)
        loss_plan = spectral_plan_alignment_loss(
            coeff_pred,
            coeff_target,
            valid_mask=coefficient_mask,
            cosine_weight=self.spectral_plan_cosine_weight,
        )
        error = F.mse_loss(
            coeff_pred.detach().float(), coeff_target.detach().float(), reduction="none"
        ).mean(dim=-1)
        if coefficient_mask is None:
            per_frequency = error.mean(dim=0)
        else:
            weight = coefficient_mask.to(device=error.device, dtype=error.dtype)
            per_frequency = (error * weight).sum(dim=0) / weight.sum(dim=0).clamp(min=1.0)
        diagnostics = spectral_plan_diagnostics(coeff_pred)
        metrics = {
            "loss_plan_raw": float(loss_plan.detach().item()),
            "plan_variance": diagnostics.coefficient_variance,
            "plan_effective_rank": diagnostics.coefficient_effective_rank,
        }
        for index, value in enumerate(per_frequency):
            metrics[f"plan_mse_k{index}"] = float(value.item())
        return loss_plan, metrics

    def _compute_spectral_plan_loss(self, coeff_pred: torch.Tensor, sample) -> torch.Tensor:
        """Compatibility wrapper for callers that only need the scalar plan loss."""
        return self._compute_spectral_plan_supervision(coeff_pred, sample)[0]

    def _run_spectral_plan_predictor(
        self,
        *,
        current_visual_tokens: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict every fixed K0-K3 coefficient from deployable context."""
        if self.spectral_plan_predictor is None:
            raise RuntimeError("Spectral plan predictor is not configured.")
        return self.spectral_plan_predictor(
            current_visual_tokens=current_visual_tokens,
            instruction_tokens=instruction_tokens,
            instruction_mask=instruction_mask,
        )

    def _build_spectral_plan_condition(
        self,
        *,
        current_visual_tokens: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
        plan_intervention: str = "predicted",
        donor_plan: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if self.spectral_plan_predictor is None or self.spectral_plan_token_projection is None:
            raise RuntimeError("Spectral plan modules are not configured.")
        prediction = self._run_spectral_plan_predictor(
            current_visual_tokens=current_visual_tokens,
            instruction_tokens=instruction_tokens,
            instruction_mask=instruction_mask,
        )
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("Fixed spectral predictor returned an incompatible payload.")
        if not isinstance(self.spectral_plan_token_projection, PlanTokenProjection):
            raise RuntimeError("Fixed spectral token projection is not configured.")
        # Preserve the original Full-K4 Joint semantics: the action loss flows
        # through projection into the coefficient head and planner trunk. Only
        # `_spectral_predictor_inputs` detaches the VideoDiT/text source tokens.
        predicted_tokens = self.spectral_plan_token_projection(prediction)
        return {
            "coefficients": prediction,
            "tokens": intervene_plan_tokens(
                predicted_tokens, mode=plan_intervention, donor_tokens=donor_plan
            ),
        }

    def _spectral_predictor_inputs(
        self, visual_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the explicit auxiliary-loss gradient boundary."""
        if self.spectral_plan_detach_backbone:
            return visual_tokens.detach(), context.detach()
        return visual_tokens, context

    def training_loss(self, sample, tiled: bool = False):
        if getattr(self, "training_phase", None) == TRAINING_PHASE_PHASE1:
            return self._training_loss_phase1_ego_planner(sample, tiled=tiled)
        if self.spectral_plan_enabled and self.spectral_plan_training_mode == "current_only":
            return self._training_loss_spectral_plan(sample, tiled=tiled)
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_mask = inputs["action_mask"]
        image_is_pad = inputs["image_is_pad"]
        embodiment_id = inputs["embodiment_id"]
        if action_mask is not None:
            action = action * action_mask.to(device=action.device, dtype=action.dtype)
        _debug_assert_finite(
            "build_inputs",
            input_latents=input_latents,
            context=context,
            action=action,
            first_frame_latents=inputs["first_frame_latents"],
        )

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]
        _debug_assert_finite(
            "video_noise_target",
            noise_video=noise_video,
            timestep_video=timestep_video,
            latents=latents,
            target_video=target_video,
        )

        noise_action = torch.randn_like(action)
        if action_mask is not None:
            noise_action = noise_action * action_mask.to(device=noise_action.device, dtype=noise_action.dtype)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        _debug_assert_finite(
            "action_noise_target",
            noise_action=noise_action,
            timestep_action=timestep_action,
            noisy_action=noisy_action,
            target_action=target_action,
        )

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        _debug_assert_finite("video_pre_dit", video_pre=video_pre)

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            embodiment_id=embodiment_id,
        )
        _debug_assert_finite("action_pre_dit", action_pre=action_pre)

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        coeff_pred = None
        plan_tokens = None
        plan_condition = None
        if self.spectral_plan_enabled:
            if self.spectral_plan_predictor is None or self.spectral_plan_token_projection is None:
                raise RuntimeError("Spectral plan modules are not configured.")
            tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            predictor_visual, predictor_context = self._spectral_predictor_inputs(
                video_tokens[:, :tokens_per_frame], context
            )
            plan_condition = self._build_spectral_plan_condition(
                current_visual_tokens=predictor_visual,
                instruction_tokens=predictor_context,
                instruction_mask=context_mask,
            )
            coeff_pred = plan_condition["coefficients"]
            plan_tokens = plan_condition["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            plan_tokens=plan_tokens,
        )
        _debug_assert_finite("mot_output", tokens_out=tokens_out)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre, embodiment_id=embodiment_id)
        _debug_assert_finite(
            "post_dit",
            pred_video=pred_video,
            pred_action=pred_action,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        _debug_assert_finite(
            "video_loss",
            loss_video_per_sample=loss_video_per_sample,
            video_weight=video_weight,
            loss_video=loss_video,
        )

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none") # [B, T, D]
        valid = torch.ones_like(action_loss_token, dtype=action_loss_token.dtype)
        if action_mask is not None:
            valid = valid * action_mask.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        if action_is_pad is not None:
            valid = valid * (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype).unsqueeze(-1)
        if embodiment_id is not None:
            # per_embodiment: supervise only each source's native dims [0:native_c]; the
            # collation pad [native_c:max] (head output is zero there) carries no loss.
            a_dims = torch.tensor(self.action_expert.action_dims, device=action_loss_token.device)
            native_c = a_dims[embodiment_id]  # [B]
            dim_idx = torch.arange(action_loss_token.shape[-1], device=action_loss_token.device)  # [D]
            dim_valid = (dim_idx.unsqueeze(0) < native_c.unsqueeze(1)).to(action_loss_token.dtype)  # [B, D]
            valid = valid * dim_valid.unsqueeze(1)  # broadcast over T
        valid_sum = valid.sum(dim=(1, 2)).clamp(min=1.0)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=(1, 2)) / valid_sum

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        _debug_assert_finite(
            "action_loss",
            action_loss_token=action_loss_token,
            action_loss_per_sample=action_loss_per_sample,
            action_weight=action_weight,
            loss_action=loss_action,
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_plan = None
        plan_metrics = None
        if coeff_pred is not None:
            loss_plan, plan_metrics = self._compute_spectral_plan_supervision(coeff_pred, sample)
            loss_total = loss_total + self.spectral_plan_lambda * loss_plan
        _debug_assert_finite("loss_total", loss_total=loss_total)

        if embodiment_id is not None:
            # Keep every per-embodiment head/encoder in the autograd graph even when its group
            # is absent from this batch (grouped forward only touches present groups). Without
            # this, DeepSpeed/DDP errors on params that received no gradient under multi-GPU.
            # Zero value + zero gradient -> no numerical effect.
            touch_modules = [self.action_expert.action_encoders, self.action_expert.heads]
            if getattr(self, "proprio_encoders", None) is not None:
                touch_modules.append(self.proprio_encoders)
            touch = sum(p.sum() for mod in touch_modules for p in mod.parameters())
            loss_total = loss_total + 0.0 * touch.to(loss_total.dtype)
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if coeff_pred is not None and loss_plan is not None and plan_metrics is not None:
            loss_dict.update({
                "loss_plan": self.spectral_plan_lambda * float(loss_plan.detach().item()),
                **plan_metrics,
            })
            for index, adapter in self.mot.plan_adapters.items():
                loss_dict[f"plan_adapter_scale_{index}"] = float(adapter.gate.detach().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        plan_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            plan_tokens=plan_tokens,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        plan_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            plan_tokens=plan_tokens,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        plan_intervention: str = "predicted",
        shuffled_plan_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        context, context_mask, _ = (
            self._append_inference_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        plan_tokens = None
        coeff_pred = None
        plan_condition = None
        if self.spectral_plan_enabled:
            if self.spectral_plan_predictor is None or self.spectral_plan_token_projection is None:
                raise RuntimeError("Spectral plan modules are not configured.")
            plan_condition = self._build_spectral_plan_condition(
                current_visual_tokens=video_pre["tokens"],
                # `proprio`, when supplied, was appended above as one current token.
                instruction_tokens=context,
                instruction_mask=context_mask,
                plan_intervention=plan_intervention,
                donor_plan=shuffled_plan_tokens,
            )
            coeff_pred = plan_condition["coefficients"]
            plan_tokens = plan_condition["tokens"]
        elif str(plan_intervention).strip().lower() != "predicted":
            raise ValueError("Plan intervention requires spectral_plan.enabled=true.")
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                plan_tokens=plan_tokens,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if coeff_pred is not None:
            result["coeff_pred"] = coeff_pred[0].detach().to(device="cpu", dtype=torch.float32)
            result["plan_intervention"] = str(plan_intervention).strip().lower()
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "training_phase": normalize_training_phase(
                getattr(self, "training_phase", None)
            ),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if getattr(self, "proprio_encoders", None) is not None:
            payload["proprio_encoders"] = self.proprio_encoders.state_dict()
        if bool(getattr(self, "spectral_plan_enabled", False)):
            if (
                self.spectral_plan_normalizer is None
                or self.spectral_plan_predictor is None
                or self.spectral_plan_token_projection is None
            ):
                raise RuntimeError("Enabled spectral plan is incomplete; refusing to save checkpoint.")
            payload["spectral_plan"] = {
                "normalizer": self.spectral_plan_normalizer.state_dict(),
                "predictor": self.spectral_plan_predictor.state_dict(),
                "token_projection": self.spectral_plan_token_projection.state_dict(),
            }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(
        self, path, optimizer=None, *, enforce_training_phase: bool = False
    ):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Checkpoint payload must be a mapping, got {type(payload).__name__}: {path}"
            )
        target_phase = normalize_training_phase(getattr(self, "training_phase", None))
        source_phase = normalize_training_phase(payload.get("training_phase"))
        if (
            (enforce_training_phase or target_phase is not None)
            and source_phase != target_phase
        ):
            raise ValueError(
                "Training-phase checkpoint mismatch: "
                f"source={source_phase!r} target={target_phase!r}. "
                "Use phase_init for the only supported Phase-I -> Phase-II transfer."
            )
        spectral_plan_enabled = bool(getattr(self, "spectral_plan_enabled", False))
        if spectral_plan_enabled and "mot" not in payload:
            raise ValueError("Spectral-plan checkpoint must contain the full `mot` state including plan adapters.")
        spectral = None
        source_mot = None
        source_normalizer = None
        source_predictor = None
        source_projection = None
        source_proprio_encoder = None
        source_proprio_encoders = None
        if spectral_plan_enabled:
            if "spectral_plan" not in payload or not isinstance(payload["spectral_plan"], Mapping):
                raise ValueError("Spectral-plan model requires checkpoint spectral_plan payload.")
            spectral = payload["spectral_plan"]
            if (
                self.spectral_plan_normalizer is None
                or self.spectral_plan_predictor is None
                or self.spectral_plan_token_projection is None
            ):
                raise RuntimeError("Enabled spectral plan is incomplete.")
            source_mot = _preflight_exact_state_transfer(
                payload["mot"], self.mot.state_dict(), label="fixed Full-K4 mot"
            )
            source_normalizer = _preflight_exact_state_transfer(
                spectral.get("normalizer"),
                self.spectral_plan_normalizer.state_dict(),
                label="fixed Full-K4 normalizer",
            )
            source_predictor = _require_tensor_mapping(
                spectral.get("predictor"), label="checkpoint fixed Full-K4 predictor"
            )
            target_predictor = _require_tensor_mapping(
                self.spectral_plan_predictor.state_dict(), label="target fixed Full-K4 predictor"
            )
            source_projection = _require_tensor_mapping(
                spectral.get("token_projection"), label="checkpoint fixed Full-K4 projection"
            )
            target_projection = _require_tensor_mapping(
                self.spectral_plan_token_projection.state_dict(), label="target fixed Full-K4 projection"
            )
            if set(source_predictor) != set(target_predictor) or set(source_projection) != set(target_projection):
                raise ValueError(
                    "Canonical resume requires a native fixed Full-K4 checkpoint. "
                    "Incompatible historical weights must use the explicit Phase-I "
                    "phase_init compatibility importer, not resume."
                )
            source_predictor = _preflight_exact_state_transfer(
                source_predictor, target_predictor, label="fixed Full-K4 predictor"
            )
            source_projection = _preflight_exact_state_transfer(
                source_projection, target_projection, label="fixed Full-K4 projection"
            )

            if self.proprio_encoder is not None:
                if "proprio_encoder" not in payload:
                    raise ValueError(
                        "Canonical fixed Full-K4 checkpoint is missing proprio_encoder."
                    )
                source_proprio_encoder = _preflight_exact_state_transfer(
                    payload["proprio_encoder"],
                    self.proprio_encoder.state_dict(),
                    label="fixed Full-K4 proprio_encoder",
                )
            elif "proprio_encoder" in payload:
                raise ValueError(
                    "Canonical fixed Full-K4 checkpoint has proprio_encoder but the "
                    "target model does not."
                )
            if getattr(self, "proprio_encoders", None) is not None:
                if "proprio_encoders" not in payload:
                    raise ValueError(
                        "Canonical fixed Full-K4 checkpoint is missing proprio_encoders."
                    )
                source_proprio_encoders = _preflight_exact_state_transfer(
                    payload["proprio_encoders"],
                    self.proprio_encoders.state_dict(),
                    label="fixed Full-K4 proprio_encoders",
                )
            elif "proprio_encoders" in payload:
                raise ValueError(
                    "Canonical fixed Full-K4 checkpoint has proprio_encoders but the "
                    "target model does not."
                )

            # Canonical resume is transactional for model tensors: every module
            # above has passed exact preflight before the first state write.
            self.mot.load_state_dict(source_mot, strict=True)
            if source_proprio_encoder is not None:
                self.proprio_encoder.load_state_dict(source_proprio_encoder, strict=True)
            if source_proprio_encoders is not None:
                self.proprio_encoders.load_state_dict(source_proprio_encoders, strict=True)
            self.spectral_plan_normalizer.load_state_dict(source_normalizer, strict=True)
            self.spectral_plan_predictor.load_state_dict(source_predictor, strict=True)
            self.spectral_plan_token_projection.load_state_dict(source_projection, strict=True)
        else:
            if "mot" in payload:
                result = self.mot.load_state_dict(payload["mot"], strict=False)
                _warn_on_incomplete_load(path, "mot", result)
            elif "dit" in payload:
                logger.warning("Loading legacy `dit` checkpoint into video expert only.")
                result = self.video_expert.load_state_dict(payload["dit"], strict=False)
                _warn_on_incomplete_load(path, "dit", result)
            else:
                raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
            if self.proprio_encoder is not None:
                if "proprio_encoder" in payload:
                    self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                else:
                    logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
            elif "proprio_encoder" in payload:
                logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
            if getattr(self, "proprio_encoders", None) is not None:
                if "proprio_encoders" in payload:
                    self.proprio_encoders.load_state_dict(payload["proprio_encoders"], strict=True)
                else:
                    logger.warning("Checkpoint has no `proprio_encoders` weights; keeping current per-embodiment proprio params.")
            elif "proprio_encoders" in payload:
                logger.warning("Checkpoint contains `proprio_encoders` weights but current model has no per-embodiment proprio; ignoring.")

        if not spectral_plan_enabled and "spectral_plan" in payload:
            logger.warning("Checkpoint contains spectral_plan but current model disables it; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def load_phase1_planner_checkpoint(
        self, path, *, random_action_expert: bool = False
    ) -> dict[str, int]:
        """Initialize Phase II from only the assets actually learned in Phase I.

        The transfer is deliberately narrow: complete VideoDiT plus the planner
        backbone. Ego normalization and the ego coefficient head (``out``),
        action/proprio, projection, adapters, and optimizer state remain at the
        Phase-II robot target initialization.
        """
        if getattr(self, "training_phase", None) != TRAINING_PHASE_POLICY:
            raise ValueError("Phase-I selective initialization is only valid for Phase II.")
        if (
            self.spectral_plan_normalizer is None
            or not isinstance(self.spectral_plan_predictor, CurrentSpectralPlanPredictor)
        ):
            raise ValueError("Phase-I selective initialization requires a fixed Full-K4 target planner.")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Phase-I checkpoint must be a mapping: {path}")
        if normalize_training_phase(payload.get("training_phase")) != TRAINING_PHASE_PHASE1:
            raise ValueError("Selective initialization requires a phase1_ego_planner checkpoint.")
        if not isinstance(random_action_expert, bool):
            raise ValueError("random_action_expert must be a boolean.")
        phase1_step = payload.get("step")
        action_source = getattr(
            self.action_expert, "pretrained_source_video_checkpoint", None
        )
        if random_action_expert and action_source is not None:
            raise ValueError(
                "Random Action expert Phase-II initialization requires an ActionDiT "
                "with no pretrained source provenance."
            )
        if not random_action_expert and not isinstance(action_source, Mapping):
            raise ValueError(
                "Fresh Phase-II phase_init requires an ActionDiT backbone with "
                "meta.source_video_checkpoint provenance."
            )
        if not random_action_expert:
            try:
                action_source_phase = normalize_training_phase(
                    action_source.get("training_phase")
                )
                action_source_step = int(action_source.get("step"))
                phase1_step_int = int(phase1_step)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ActionDiT source provenance and Phase-I checkpoint must both "
                    "declare valid training_phase/step metadata."
                ) from exc
            if (
                action_source_phase != TRAINING_PHASE_PHASE1
                or action_source_step != phase1_step_int
            ):
                raise ValueError(
                    "ActionDiT backbone provenance does not match the Phase-I phase_init "
                    "checkpoint: "
                    f"action_source_phase={action_source_phase!r} "
                    f"action_source_step={action_source_step} "
                    f"phase_init_step={phase1_step_int}."
                )
        if "mot" not in payload or "spectral_plan" not in payload:
            raise ValueError("Phase-I checkpoint requires mot and spectral_plan payloads.")

        source_mot = _require_tensor_mapping(payload["mot"], label="Phase-I mot")
        video_prefix = "mixtures.video."
        source_video = {
            key[len(video_prefix):]: value
            for key, value in source_mot.items()
            if key.startswith(video_prefix)
        }
        target_video = _require_tensor_mapping(
            self.video_expert.state_dict(), label="Phase-II video expert"
        )
        if set(source_video) != set(target_video):
            raise ValueError(
                "Phase-I VideoDiT key-set mismatch: "
                f"missing={sorted(set(target_video) - set(source_video))[:10]} "
                f"unexpected={sorted(set(source_video) - set(target_video))[:10]}."
            )
        for key in sorted(target_video):
            _validate_exact_transfer_tensor(
                source_video[key], target_video[key], label=f"Phase-I VideoDiT.{key}"
            )

        spectral = payload["spectral_plan"]
        if not isinstance(spectral, Mapping):
            raise ValueError("Phase-I spectral_plan payload must be a mapping.")
        if "predictor" not in spectral:
            raise ValueError("Phase-I spectral_plan requires a predictor state.")

        target_predictor_all = _require_tensor_mapping(
            self.spectral_plan_predictor.state_dict(), label="Phase-II predictor"
        )
        source_predictor = import_phase1_planner_backbone(
            spectral["predictor"], target_predictor_all
        )

        # All checks above are transactional: no target tensor is touched until
        # every selected section has passed key/shape/type/finiteness validation.
        self.video_expert.load_state_dict(source_video, strict=True)
        merged_predictor = dict(target_predictor_all)
        merged_predictor.update(source_predictor)
        self.spectral_plan_predictor.load_state_dict(merged_predictor, strict=True)
        return {
            "video_tensors": len(source_video),
            "planner_backbone_tensors": len(source_predictor),
            "random_action_expert": int(random_action_expert),
        }

    def load_checkpoint_partial(
        self,
        path,
        *,
        action_endpoint_policy: str,
        proprio_endpoint_policy: str,
    ):
        """Initialize from a source checkpoint under a fail-closed transfer contract.

        The complete non-endpoint MoT/DiT backbone is mandatory and exact. Action and proprio
        endpoints are independent atomic units: ``reset`` preserves target initialization;
        ``load_exact`` transfers the complete module after the caller has established schema
        compatibility. All topology/type/shape/finiteness checks run before the first tensor write.
        """
        action_endpoint_policy = str(action_endpoint_policy).strip().lower()
        proprio_endpoint_policy = str(proprio_endpoint_policy).strip().lower()
        for name, policy in (
            ("action", action_endpoint_policy),
            ("proprio", proprio_endpoint_policy),
        ):
            if policy not in _PARTIAL_ENDPOINT_POLICIES:
                raise ValueError(
                    f"{name}_endpoint_policy must be one of "
                    f"{sorted(_PARTIAL_ENDPOINT_POLICIES)}, got {policy!r}."
                )

        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Partial checkpoint must be a mapping: {path}")
        if "mot" not in payload:
            raise ValueError(f"Partial checkpoint loading requires `mot` weights: {path}")
        source_mot_state = _require_tensor_mapping(payload["mot"], label="source mot")
        target_mot_state = _require_tensor_mapping(self.mot.state_dict(), label="target mot")

        # Ensure the canonical action endpoint definition refers to the exact module mixed by MoT.
        mixtures = getattr(self.mot, "mixtures", None)
        if mixtures is None or "action" not in mixtures:
            raise ValueError("Target MoT has no canonical `mixtures.action` expert.")
        mot_action_expert = mixtures["action"]
        configured_action_expert = getattr(self, "action_expert", mot_action_expert)
        if configured_action_expert is not mot_action_expert:
            raise ValueError("self.action_expert is not the module registered as mot.mixtures.action.")

        source_action_mode, source_action_keys = _validate_action_endpoint_topology(
            source_mot_state, label="source action endpoint"
        )
        for key in sorted(source_action_keys):
            _require_finite(source_mot_state[key], label=f"source action endpoint.{key}")
        target_action_mode, target_action_keys = _validate_action_endpoint_topology(
            target_mot_state, label="target action endpoint"
        )
        configured_head_mode = getattr(self, "action_head_mode", target_action_mode)
        if configured_head_mode != target_action_mode:
            raise ValueError(
                "Target action_head_mode disagrees with endpoint topology: "
                f"configured={configured_head_mode!r} topology={target_action_mode!r}."
            )

        source_backbone_keys = frozenset(source_mot_state) - source_action_keys
        target_backbone_keys = frozenset(target_mot_state) - target_action_keys
        missing_backbone = sorted(target_backbone_keys - source_backbone_keys)
        unexpected_backbone = sorted(source_backbone_keys - target_backbone_keys)
        if missing_backbone or unexpected_backbone:
            raise ValueError(
                "Partial checkpoint backbone key-set mismatch; every non-endpoint Video DiT, "
                "Action DiT and MoT tensor must come from source. "
                f"missing={missing_backbone[:10]} unexpected={unexpected_backbone[:10]}."
            )
        if not target_backbone_keys:
            raise ValueError("Partial checkpoint contract found an empty MoT/DiT backbone.")

        for key in sorted(target_backbone_keys):
            _validate_exact_transfer_tensor(
                source_mot_state[key], target_mot_state[key], label=f"backbone.{key}"
            )

        if action_endpoint_policy == "load_exact":
            if source_action_mode != target_action_mode or source_action_keys != target_action_keys:
                raise ValueError(
                    "action endpoint load_exact requires identical shared/per topology and key-set; "
                    f"source_mode={source_action_mode} target_mode={target_action_mode}."
                )
            for key in sorted(target_action_keys):
                _validate_exact_transfer_tensor(
                    source_mot_state[key], target_mot_state[key], label=f"action_endpoint.{key}"
                )
            action_keys_to_load = target_action_keys
        else:
            for key in sorted(target_action_keys):
                _require_finite(target_mot_state[key], label=f"target action init.{key}")
            action_keys_to_load = frozenset()

        source_has_shared_proprio = "proprio_encoder" in payload
        source_has_per_proprio = "proprio_encoders" in payload
        if source_has_shared_proprio and source_has_per_proprio:
            raise ValueError(
                "Source checkpoint cannot contain both proprio_encoder and proprio_encoders."
            )
        source_proprio_mode = None
        source_proprio_state = None
        source_proprio_key = None
        if source_has_shared_proprio:
            source_proprio_mode = "shared"
            source_proprio_key = "proprio_encoder"
            source_proprio_state = payload[source_proprio_key]
            _validate_proprio_topology(
                source_proprio_state, mode=source_proprio_mode, label="source proprio_encoder"
            )
        elif source_has_per_proprio:
            source_proprio_mode = "per_embodiment"
            source_proprio_key = "proprio_encoders"
            source_proprio_state = payload[source_proprio_key]
            _validate_proprio_topology(
                source_proprio_state, mode=source_proprio_mode, label="source proprio_encoders"
            )
        if source_proprio_state is not None:
            for key, value in source_proprio_state.items():
                _require_finite(value, label=f"source proprio endpoint.{key}")
        if source_proprio_mode is not None and source_proprio_mode != source_action_mode:
            raise ValueError(
                "Source action/proprio endpoint modes disagree: "
                f"action={source_action_mode} proprio={source_proprio_mode}."
            )

        target_shared_proprio = getattr(self, "proprio_encoder", None)
        target_per_proprio = getattr(self, "proprio_encoders", None)
        if target_shared_proprio is not None and target_per_proprio is not None:
            raise ValueError("Target cannot have both proprio_encoder and proprio_encoders.")
        target_proprio_mode = None
        target_proprio_module = None
        target_proprio_key = None
        target_proprio_state = None
        if target_shared_proprio is not None:
            target_proprio_mode = "shared"
            target_proprio_module = target_shared_proprio
            target_proprio_key = "proprio_encoder"
        elif target_per_proprio is not None:
            target_proprio_mode = "per_embodiment"
            target_proprio_module = target_per_proprio
            target_proprio_key = "proprio_encoders"
        if target_proprio_module is not None:
            if target_proprio_mode != target_action_mode:
                raise ValueError(
                    "Target action/proprio endpoint modes disagree: "
                    f"action={target_action_mode} proprio={target_proprio_mode}."
                )
            target_proprio_state = target_proprio_module.state_dict()
            _validate_proprio_topology(
                target_proprio_state,
                mode=target_proprio_mode,
                label=f"target {target_proprio_key}",
            )

        proprio_loaded = False
        if proprio_endpoint_policy == "load_exact":
            if target_proprio_module is None or source_proprio_state is None:
                raise ValueError(
                    "proprio endpoint load_exact requires both source and target projectors."
                )
            if source_proprio_mode != target_proprio_mode or source_proprio_key != target_proprio_key:
                raise ValueError(
                    "proprio endpoint load_exact requires identical shared/per topology."
                )
            if set(source_proprio_state) != set(target_proprio_state):
                raise ValueError(
                    "proprio endpoint load_exact requires an identical complete key-set: "
                    f"source={sorted(source_proprio_state)} target={sorted(target_proprio_state)}."
                )
            for key in sorted(target_proprio_state):
                _validate_exact_transfer_tensor(
                    source_proprio_state[key],
                    target_proprio_state[key],
                    label=f"proprio_endpoint.{key}",
                )
        elif target_proprio_state is not None:
            for key, value in target_proprio_state.items():
                _require_finite(value, label=f"target proprio init.{key}")

        # All predictable validation is complete. Only now mutate the target model.
        mot_keys_to_load = target_backbone_keys | action_keys_to_load
        loadable_mot_state = {key: source_mot_state[key] for key in mot_keys_to_load}
        result = self.mot.load_state_dict(loadable_mot_state, strict=False)
        expected_missing = target_action_keys - action_keys_to_load
        actual_missing = set(getattr(result, "missing_keys", []) or [])
        actual_unexpected = set(getattr(result, "unexpected_keys", []) or [])
        if actual_missing != set(expected_missing) or actual_unexpected:
            raise RuntimeError(
                "Unexpected PyTorch load result after successful partial preflight: "
                f"missing={sorted(actual_missing)} expected={sorted(expected_missing)} "
                f"unexpected={sorted(actual_unexpected)}."
            )
        if proprio_endpoint_policy == "load_exact":
            target_proprio_module.load_state_dict(source_proprio_state, strict=True)
            proprio_loaded = True

        summary = {
            "mot_backbone_loaded": len(target_backbone_keys),
            "mot_backbone_expected": len(target_backbone_keys),
            "source_action_mode": source_action_mode,
            "target_action_mode": target_action_mode,
            "action_endpoint_policy": action_endpoint_policy,
            "action_endpoint_loaded": action_endpoint_policy == "load_exact",
            "action_endpoint_tensors": len(target_action_keys),
            "proprio_endpoint_policy": proprio_endpoint_policy,
            "proprio_loaded": proprio_loaded,
            "proprio_endpoint_tensors": 0 if target_proprio_state is None else len(target_proprio_state),
        }
        logger.info("Strict partial checkpoint load from %s: %s", path, summary)
        return {
            **summary,
            # Compatibility count for older monitoring code; this now includes only actually
            # transferred MoT tensors and no permissively skipped backbone.
            "mot_loaded": len(loadable_mot_state),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


def _warn_on_incomplete_load(path, section, result) -> None:
    """Log missing / unexpected keys after a strict=False load_state_dict.

    PyTorch raises on shape-mismatch even with strict=False, but missing/unexpected
    keys are silently swallowed. Surfacing them here helps catch architecture or
    naming drift between checkpoint and current model (e.g. action_encoder/head
    skipped when reusing a 14-dim ActionDiT backbone at 7 dims).
    """
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if not missing and not unexpected:
        return
    logger.warning(
        "load_checkpoint(%s) section=%s: missing_keys=%d unexpected_keys=%d",
        path, section, len(missing), len(unexpected),
    )
    preview_n = 10
    if missing:
        logger.warning("  missing[:%d]=%s%s", preview_n, missing[:preview_n], " ..." if len(missing) > preview_n else "")
    if unexpected:
        logger.warning("  unexpected[:%d]=%s%s", preview_n, unexpected[:preview_n], " ..." if len(unexpected) > preview_n else "")
