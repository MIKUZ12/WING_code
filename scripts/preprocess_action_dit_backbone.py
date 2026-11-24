import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from wing.models.wan22.action_dit import ActionDiT
from wing.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(
    path: Path, *, action_dim: int = 7
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    loaded = OmegaConf.load(str(path))
    if "defaults" not in loaded and "video_dit_config" in loaded and "action_dit_config" in loaded:
        cfg = loaded
    else:
        config_root = path.resolve().parent.parent
        with initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"model={path.stem}"]).model

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")

    if _is_unresolved_interpolation(video_cfg.get("action_dim")):
        video_cfg["action_dim"] = action_dim

    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        action_cfg["action_dim"] = action_dim

    for key in ["num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"]:
        action_cfg[key] = _resolve_from_video_cfg(action_cfg.get(key), video_cfg)

    checkpoint_interval = cfg.get("mot_gradient_checkpointing_interval", 1)
    for branch in (video_cfg, action_cfg):
        if branch.get("gradient_checkpointing_interval") == "${model.mot_gradient_checkpointing_interval}":
            branch["gradient_checkpointing_interval"] = checkpoint_interval
        for key, value in list(branch.items()):
            if not isinstance(value, str) or not value.startswith("${model."):
                continue
            model_key = value[len("${model.") : -1]
            resolved = cfg.get(model_key)
            if resolved is not None and not _is_unresolved_interpolation(resolved):
                branch[key] = resolved

    return video_cfg, action_cfg, cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _load_video_expert_checkpoint(
    video_expert: torch.nn.Module, checkpoint_path: Path
) -> dict[str, Any]:
    """Load a trained VideoDiT into ``video_expert`` before ActionDiT interpolation.

    Phase-I WING checkpoints store the VideoDiT under
    ``mot.mixtures.video.*``.  A direct VideoDiT state dict is also accepted so
    the documented video-only finetuning workflow remains usable.
    """
    payload = torch.load(
        str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"VideoDiT checkpoint must contain a mapping, got {type(payload)}: "
            f"{checkpoint_path}"
        )

    source: Mapping[str, Any]
    source_prefix = ""
    if "mot" in payload:
        mot = payload["mot"]
        if not isinstance(mot, Mapping):
            raise ValueError("Phase-I checkpoint field `mot` must be a mapping.")
        source = mot
        source_prefix = "mixtures.video."
    elif "dit" in payload and isinstance(payload["dit"], Mapping):
        source = payload["dit"]
    elif "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        source = payload["state_dict"]
    else:
        source = payload

    if source_prefix:
        video_state = {
            str(key)[len(source_prefix):]: value
            for key, value in source.items()
            if str(key).startswith(source_prefix)
        }
    else:
        target_keys = set(video_expert.state_dict())
        video_state = {str(key): value for key, value in source.items()}
        if set(video_state) != target_keys:
            for prefix in ("video_expert.", "dit.", "module."):
                candidate = {
                    key[len(prefix):]: value
                    for key, value in video_state.items()
                    if key.startswith(prefix)
                }
                if set(candidate) == target_keys:
                    video_state = candidate
                    break

    target_state = video_expert.state_dict()
    missing = sorted(set(target_state) - set(video_state))
    unexpected = sorted(set(video_state) - set(target_state))
    if missing or unexpected:
        raise ValueError(
            "VideoDiT checkpoint key-set mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    for key, target in target_state.items():
        source_tensor = video_state[key]
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError(f"VideoDiT checkpoint value is not a tensor: {key}")
        if tuple(source_tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"VideoDiT checkpoint shape mismatch for {key}: "
                f"source={tuple(source_tensor.shape)}, target={tuple(target.shape)}"
            )

    video_expert.load_state_dict(video_state, strict=True)
    return {
        "path": str(checkpoint_path.resolve()),
        "training_phase": payload.get("training_phase"),
        "step": payload.get("step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml, e.g. configs/model/wing.yaml")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument(
        "--dit-checkpoint",
        default=None,
        help=(
            "Optional trained VideoDiT checkpoint. Phase-I WING .pt files "
            "with mot.mixtures.video.* and direct VideoDiT state dicts are supported."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    parser.add_argument(
        "--redirect-common-files",
        default=None,
        choices=["true", "false"],
        help="Optional override for model-config redirect_common_files.",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    video_cfg, action_cfg, cfg = _load_model_config(
        model_config_path, action_dim=args.action_dim
    )
    torch_dtype = _parse_dtype(args.dtype)
    redirect_common_files = _parse_bool(cfg.get("redirect_common_files", False))
    if args.redirect_common_files is not None:
        redirect_common_files = _parse_bool(args.redirect_common_files)

    int_fields = ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(f"[INFO] Loaded model config from {model_config_path}. "
          f"Preprocessing ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
          f"apply_alpha_scaling={apply_alpha_scaling}.")
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        load_text_encoder=False,
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
    )
    video_expert = components.dit
    source_checkpoint_meta = None
    if args.dit_checkpoint:
        dit_checkpoint_path = Path(args.dit_checkpoint)
        if not dit_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"VideoDiT checkpoint does not exist: {dit_checkpoint_path}"
            )
        source_checkpoint_meta = _load_video_expert_checkpoint(
            video_expert, dit_checkpoint_path
        )
        print(
            "[INFO] Loaded trained VideoDiT before ActionDiT interpolation: "
            f"{source_checkpoint_meta}"
        )

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
            "source_video_checkpoint": source_checkpoint_meta,
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
