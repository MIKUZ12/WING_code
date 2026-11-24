"""WING policy adapter with an RTC-compatible client interface."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from wing.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving
from wing.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wing.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wing.deployment.joint_subset import (
    ROBOT_FULL_DIM,
    ROBOT_JOINT_NAMES,
    build_pad_template,
    pad_action,
    parse_keep_dims,
    slice_state,
    validate_keep_dims,
    verify_keep_dims_against_lerobot,
)
from wing.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


ROBOTWIN_IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
WING_INFERENCE_FACTORIES = {
    "wing.runtime.create_wing",
    "wing.runtime.create_wing_joint",
    "wing.runtime.create_wing_idm",
}


@dataclass(frozen=True)
class WINGPolicyConfig:
    """Configuration needed to load WING as a policy backend."""

    ckpt: str
    dataset_stats: str
    task: str = "libero"
    config_dir: str = "configs"
    train_config: str | None = None
    device: str = "cuda:0"
    mixed_precision: str = "bf16"
    prompt: str = "clean the table"
    context_cache_dir: str | None = None
    use_text_encoder: bool = False
    action_horizon: int = 32
    num_inference_steps: int = 10
    seed: int = 42
    rand_device: str = "cpu"
    # When a subset checkpoint models only part of the robot (for example, the
    # right arm at indices 7:14), map between its N-dimensional space and the
    # robot's 14-dimensional interface:
    # input  state[14]  -> state[keep_dims]  (14 -> N)
    # output action[N]  -> action[14] with non-kept dims = full_action_pad
    # `keep_dims` accepts slice notation "start:stop" or comma-separated indices.
    # `keep_dims is None` -> model is already 14-dim, no conversion (backward compat).
    keep_dims: str | None = None
    # Required 14-d template used to fill non-kept dimensions during action
    # denormalization. It must match the calibrated dual-arm initial pose.
    full_action_pad: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], base_dir: str | Path | None = None) -> "WINGPolicyConfig":
        """Build config from a deployment YAML mapping.

        The mapping may either contain these keys directly or under a
        top-level ``wing`` section.
        """
        raw = payload.get("wing", payload)
        if not isinstance(raw, Mapping):
            raise TypeError("Expected a mapping or a top-level `wing` mapping.")

        def get(name: str, default: Any = None) -> Any:
            return raw[name] if name in raw else default

        def path_value(name: str, required: bool = False) -> str | None:
            value = get(name)
            if value is None or not str(value).strip():
                if required:
                    raise ValueError(f"Missing required WING policy config key: {name}")
                return None
            path = Path(str(value)).expanduser()
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir).expanduser() / path
            return str(path)

        pad_raw = get("full_action_pad")
        pad_tuple = tuple(float(x) for x in pad_raw) if pad_raw is not None else None

        keep_raw = get("keep_dims")
        keep_str = str(keep_raw) if keep_raw is not None else None

        return cls(
            ckpt=path_value("ckpt", required=True) or "",
            dataset_stats=path_value("dataset_stats", required=True) or "",
            task=str(get("task", cls.task)),
            config_dir=path_value("config_dir") or cls.config_dir,
            train_config=path_value("train_config"),
            device=str(get("device", cls.device)),
            mixed_precision=str(get("mixed_precision", cls.mixed_precision)),
            prompt=str(get("prompt", cls.prompt)),
            context_cache_dir=path_value("context_cache_dir"),
            use_text_encoder=bool(get("use_text_encoder", cls.use_text_encoder)),
            action_horizon=int(get("action_horizon", cls.action_horizon)),
            num_inference_steps=int(get("num_inference_steps", cls.num_inference_steps)),
            seed=int(get("seed", cls.seed)),
            rand_device=str(get("rand_device", cls.rand_device)),
            keep_dims=keep_str,
            full_action_pad=pad_tuple,
        )


@dataclass
class WINGObservation:
    images: dict[str, np.ndarray]
    state: np.ndarray
    prompt: str


class WINGPolicyClient:
    """Local WING policy client compatible with RTC-style deployment code."""

    def __init__(self, config: WINGPolicyConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, WINGPolicyConfig) else WINGPolicyConfig.from_mapping(config)
        self.cfg = self._load_config()
        model_dtype = dtype_from_mixed_precision(self.config.mixed_precision)
        self.model = instantiate_inference_model(
            self.cfg,
            model_dtype=model_dtype,
            device=self.config.device,
        )
        self.model.load_checkpoint(self.config.ckpt)
        self.model.eval()
        self._joint_infer_kwargs = _joint_inference_video_frame_kwargs(self.model, self.cfg)

        self._policy_data_cfg = _base_robot_dataset_config(self.cfg)
        self.processor = instantiate(self._policy_data_cfg.processor)
        self._image_keys = tuple(
            str(meta["key"]) for meta in self._policy_data_cfg.shape_meta.images
        )
        if not self._image_keys:
            raise ValueError("WING policy config must declare at least one image key.")
        stats = load_dataset_stats_from_json(self.config.dataset_stats)
        self.processor.set_normalizer_from_stats(stats)
        self.processor.eval()

        self.observation: WINGObservation | None = None
        self._context_cache: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

        # 7<->14 dim adaptation. When keep_dims is None, the model is already
        # robot-dim and no conversion happens (backward compatible with the
        # previous 14-dim deploy path). All actual logic lives in joint_subset.
        self._keep_dims: list[int] | None = parse_keep_dims(self.config.keep_dims)
        self._pad_template: np.ndarray | None = None
        if self._keep_dims is not None:
            validate_keep_dims(self._keep_dims, int(self.processor.proprio_output_dim))
            self._pad_template = build_pad_template(self.config.full_action_pad)
            verify_keep_dims_against_lerobot(self.cfg, self._keep_dims)

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        """Store the latest RTC-style observation for the next ``get_action`` call."""
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an `images` mapping.")

        missing = [key for key in self._image_keys if key not in images or images[key] is None]
        if missing:
            raise ValueError(f"Observation is missing required image keys: {missing}")

        if "state" not in obs:
            raise ValueError("Observation must contain a `state` vector.")

        prompt = str(obs.get("prompt", self.config.prompt))
        self.observation = WINGObservation(
            images={key: np.asarray(images[key]) for key in self._image_keys},
            state=np.asarray(obs["state"], dtype=np.float32),
            prompt=prompt,
        )

    def get_action(self) -> np.ndarray:
        """Run WING inference and return a denormalized ``[T,D]`` action chunk.

        The model outputs ``[T, action_output_dim]`` where the action dimension
        may be 7 (right-arm-only models) or 14 (full
        dual-arm models). Native policies return their checkpoint dimension.
        When ``keep_dims`` is configured, the server accepts 14D robot state,
        slices it to the model subset, and pads model actions back to 14D.
        """
        if self.observation is None:
            raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")

        # Adapter mode receives 14D robot state and slices it to the model's
        # input dims. Native mode forwards the runtime's 7D or 14D state.
        if self._keep_dims is not None:
            model_state = slice_state(self.observation.state, self._keep_dims)
        else:
            model_state = self.observation.state

        prompt, context, context_mask = self._resolve_text_condition(self.observation.prompt)
        image = preprocess_policy_images(
            self.observation,
            self._policy_data_cfg,
            self.model.device,
            self.model.torch_dtype,
        )
        proprio = normalize_proprio(self.processor, model_state)

        with torch.no_grad():
            output = self.model.infer_action(
                prompt=prompt,
                input_image=image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                action_horizon=self.config.action_horizon,
                num_inference_steps=self.config.num_inference_steps,
                seed=self.config.seed,
                rand_device=self.config.rand_device,
                **self._joint_infer_kwargs,
            )

        actions = denormalize_actions(self.processor, output["action"])
        actions_np = actions.detach().cpu().numpy().astype(np.float32)

        # Pad model-dim action back to robot-dim. Non-kept dims keep the
        # configured pad values (e.g. left-arm init pose), while kept dims are
        # overwritten by the model output.
        if self._keep_dims is not None:
            assert self._pad_template is not None  # constructed in __init__
            return pad_action(actions_np, self._keep_dims, self._pad_template)

        return actions_np

    def metadata(self) -> dict[str, Any]:
        """Return cheap deployment metadata without running inference."""
        model_state_dim = int(self.processor.proprio_output_dim)
        model_action_dim = int(self.processor.action_output_dim)
        interface_dim = ROBOT_FULL_DIM if self._keep_dims is not None else model_state_dim
        interface_action_dim = ROBOT_FULL_DIM if self._keep_dims is not None else model_action_dim
        return {
            "proprio_output_dim": interface_dim,
            "action_output_dim": interface_action_dim,
            "model_proprio_output_dim": model_state_dim,
            "model_action_output_dim": model_action_dim,
            "keep_dims": None if self._keep_dims is None else list(self._keep_dims),
            "image_keys": list(self._image_keys),
            "action_horizon": int(self.config.action_horizon),
            "prompt": str(self.config.prompt),
        }

    def reset(self) -> None:
        """Clear per-episode observation state."""
        self.observation = None

    def _load_config(self) -> DictConfig:
        if self.config.train_config:
            cfg = OmegaConf.load(self.config.train_config)
            if not isinstance(cfg, DictConfig):
                raise ValueError(f"Expected DictConfig in {self.config.train_config}, got {type(cfg)}")
            if self.config.use_text_encoder:
                cfg.model.load_text_encoder = True
            return cfg

        config_dir = str(Path(self.config.config_dir).expanduser().resolve())
        overrides = [f"task={self.config.task}", f"mixed_precision={self.config.mixed_precision}"]
        if self.config.use_text_encoder:
            overrides.append("model.load_text_encoder=true")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            return compose(config_name="train", overrides=overrides)

    def _resolve_text_condition(self, task_prompt: str) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
        formatted_prompt = format_prompt(task_prompt)
        if self.config.use_text_encoder:
            return formatted_prompt, None, None

        if formatted_prompt not in self._context_cache:
            cache_dir = self.config.context_cache_dir or self._policy_data_cfg.get("text_embedding_cache_dir")
            if cache_dir is None:
                raise ValueError("No context cache dir found. Set context_cache_dir or use_text_encoder=true.")
            context_len = int(self._policy_data_cfg.get("context_len", 128))
            self._context_cache[formatted_prompt] = load_cached_context(cache_dir, formatted_prompt, context_len)

        context, context_mask = self._context_cache[formatted_prompt]
        return None, context, context_mask


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def instantiate_inference_model(
    cfg: DictConfig,
    *,
    model_dtype: torch.dtype,
    device: str,
) -> torch.nn.Module:
    """Construct checkpoint topology without obsolete training initializers."""
    target = str(cfg.model.get("_target_", ""))
    if target not in WING_INFERENCE_FACTORIES:
        raise ValueError(
            f"Unsupported WING deployment model factory {target!r}; expected one of "
            f"{sorted(WING_INFERENCE_FACTORIES)}."
        )
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    cfg.model.dit_checkpoint_path = None
    return instantiate(cfg.model, model_dtype=model_dtype, device=device)


def _base_robot_dataset_config(cfg: DictConfig) -> DictConfig:
    """Return the RobotVideoDataset node beneath optional training wrappers."""
    dataset_cfg = cfg.data.train
    while dataset_cfg is not None:
        if dataset_cfg.get("processor", None) is not None and dataset_cfg.get(
            "shape_meta", None
        ) is not None:
            return dataset_cfg
        dataset_cfg = dataset_cfg.get("base_dataset", None)
    raise ValueError(
        "Policy inference requires processor and shape_meta on data.train or "
        "one of its base_dataset nodes."
    )


def _configured_num_video_frames(cfg: DictConfig) -> int:
    """Derive policy video frames with the same action/video sampling convention as eval.

    Direct robot datasets keep these fields under ``data.train``; latent action spectral
    training wraps the same dataset under one or more ``base_dataset`` nodes.
    """
    dataset_cfg = cfg.data.train
    while dataset_cfg is not None:
        raw_num_frames = dataset_cfg.get("num_frames", None)
        action_video_freq_ratio = dataset_cfg.get("action_video_freq_ratio", None)
        if raw_num_frames is not None and action_video_freq_ratio is not None:
            raw_num_frames = int(raw_num_frames)
            action_video_freq_ratio = int(action_video_freq_ratio)
            if raw_num_frames < 1:
                raise ValueError(f"data.train num_frames must be positive, got {raw_num_frames}.")
            if action_video_freq_ratio < 1:
                raise ValueError(
                    "data.train action_video_freq_ratio must be positive, "
                    f"got {action_video_freq_ratio}."
                )
            return (raw_num_frames - 1) // action_video_freq_ratio + 1
        dataset_cfg = dataset_cfg.get("base_dataset", None)
    raise ValueError(
        "Joint policy inference requires num_frames and action_video_freq_ratio "
        "on data.train or its base_dataset."
    )


def _joint_inference_video_frame_kwargs(model: torch.nn.Module, cfg: DictConfig) -> dict[str, int]:
    """Supply the joint-only argument without breaking native WING policies."""
    if "num_video_frames" not in inspect.signature(model.infer_action).parameters:
        return {}
    return {"num_video_frames": _configured_num_video_frames(cfg)}


def format_prompt(task_prompt: str) -> str:
    return DEFAULT_PROMPT.format(task=task_prompt)


def load_cached_context(cache_dir: str | Path, formatted_prompt: str, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this prompt, or set use_text_encoder=true."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context length mismatch in {cache_path}: "
            f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
        )
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask


def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
    rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def preprocess_robotwin_images(
    observation: WINGObservation,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    missing = [key for key in ROBOTWIN_IMAGE_KEYS if key not in observation.images]
    if missing:
        raise ValueError(f"RoboTwin image layout is missing keys: {missing}")
    cam_high = transforms_F.resize(
        rgb_to_float_chw(observation.images["cam_high"]),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        rgb_to_float_chw(observation.images["cam_left_wrist"]),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        rgb_to_float_chw(observation.images["cam_right_wrist"]),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high = transforms_F.resize(
        cam_high,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        cam_left,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        cam_right,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    image = torch.cat([cam_high, bottom], dim=-2)
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    crop = CenterCrop(args={"img_w": 320, "img_h": 384})
    image = crop(resize(image))
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def preprocess_policy_images(
    observation: WINGObservation,
    data_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply the camera layout and final crop declared by RobotVideoDataset."""
    image_keys = tuple(str(meta["key"]) for meta in data_cfg["shape_meta"]["images"])
    layout = data_cfg.get("concat_multi_camera", None)
    if layout == "robotwin":
        return preprocess_robotwin_images(observation, device, dtype)

    images = [rgb_to_float_chw(observation.images[key]) for key in image_keys]
    if len(images) == 1:
        image = images[0]
    elif layout == "horizontal":
        image = torch.cat(images, dim=-1)
    elif layout == "vertical":
        image = torch.cat(images, dim=-2)
    else:
        raise ValueError(
            f"Multiple policy cameras require horizontal/vertical/robotwin layout, got {layout!r}."
        )

    video_h, video_w = (int(value) for value in data_cfg["video_size"])
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": video_w, "img_h": video_h})
    crop = CenterCrop(args={"img_w": video_w, "img_h": video_h})
    image = crop(resize(image)).mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def denormalize_actions(processor: Any, action_norm: torch.Tensor) -> torch.Tensor:
    if action_norm.ndim == 3 and action_norm.shape[0] == 1:
        action_norm = action_norm.squeeze(0)
    if action_norm.ndim == 1:
        action_norm = action_norm.view(1, -1)
    if action_norm.ndim != 2:
        raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}")
    dummy_state = torch.zeros(
        1,
        action_norm.shape[0],
        int(processor.proprio_output_dim),
        dtype=torch.float32,
        device=action_norm.device,
    )
    batch = {
        "action": {"default": action_norm.unsqueeze(0).to(torch.float32)},
        "state": {"default": dummy_state},
    }
    batch = processor.normalizer.backward(batch)
    return batch["action"]["default"].squeeze(0)


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_state_vector(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if "state" not in payload:
            raise ValueError(f"{path} must contain a `state` key or be a raw list.")
        payload = payload["state"]
    state = np.asarray(payload, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"State must be a 1D vector, got shape {state.shape}")
    return state
