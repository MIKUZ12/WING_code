"""WING policy adapter for the RoboTwin 2.0 evaluator contract."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from wing.datasets.lerobot.processors.wing_processor import WINGProcessor
from wing.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wing.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wing.utils.config_resolvers import register_default_resolvers


REPO_ROOT = Path(__file__).resolve().parents[3]
CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
def _model_runtime_root(args: dict[str, Any]) -> Path:
    return Path(args.get("model_runtime_root", REPO_ROOT)).expanduser().resolve()


def _as_rgb_tensor(value: Any, *, camera_name: str) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"RoboTwin {camera_name} RGB must have shape [H,W,3], got {array.shape}."
        )
    tensor = torch.as_tensor(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    else:
        tensor = tensor.to(dtype=torch.float32)
        if float(tensor.max().item()) > 1.0:
            tensor = tensor.div(255.0)
    return tensor.permute(2, 0, 1).contiguous()


def _configured_camera_sizes(dataset_cfg: DictConfig | None) -> list[tuple[int, int]]:
    if dataset_cfg is None:
        return [(240, 320)] * len(CAMERA_NAMES)
    image_meta = dataset_cfg.shape_meta["images"]
    if len(image_meta) != len(CAMERA_NAMES):
        raise ValueError(
            f"RoboTwin evaluation expects {len(CAMERA_NAMES)} image entries, "
            f"got {len(image_meta)}."
        )
    sizes = []
    for index, meta in enumerate(image_meta):
        shape = tuple(int(x) for x in meta["shape"])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(
                f"RoboTwin shape_meta.images[{index}].shape must be [3,H,W], got {shape}."
            )
        sizes.append((shape[1], shape[2]))
    return sizes


def compose_robotwin_rgb(
    observation: dict[str, Any], dataset_cfg: DictConfig | None = None
) -> torch.Tensor:
    """Reproduce training's 3-camera 384x320 RoboTwin mosaic exactly."""
    camera_root = observation.get("observation")
    if not isinstance(camera_root, dict):
        raise KeyError("RoboTwin observation is missing the 'observation' camera mapping.")
    cameras = []
    camera_sizes = _configured_camera_sizes(dataset_cfg)
    for name, camera_size in zip(CAMERA_NAMES, camera_sizes, strict=True):
        camera = camera_root.get(name)
        if not isinstance(camera, dict) or "rgb" not in camera:
            raise KeyError(f"RoboTwin observation is missing observation.{name}.rgb.")
        tensor = _as_rgb_tensor(camera["rgb"], camera_name=name)
        cameras.append(
            transforms_F.resize(tensor, list(camera_size), antialias=True)
        )

    top = transforms_F.resize(cameras[0], [256, 320], antialias=True)
    left = transforms_F.resize(cameras[1], [128, 160], antialias=True)
    right = transforms_F.resize(cameras[2], [128, 160], antialias=True)
    mosaic = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
    if tuple(mosaic.shape) != (3, 384, 320):
        raise ValueError(f"Unexpected RoboTwin mosaic shape: {tuple(mosaic.shape)}.")
    return mosaic.mul(2.0).sub(1.0).contiguous()


def extract_robotwin_qpos(observation: dict[str, Any]) -> np.ndarray:
    joint = observation.get("joint_action")
    if not isinstance(joint, dict) or "vector" not in joint:
        raise KeyError("RoboTwin observation is missing joint_action.vector.")
    state = np.asarray(joint["vector"], dtype=np.float32).reshape(-1)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"RoboTwin qpos must be finite [14], got {state.shape}.")
    return state


def _robot_dataset_cfg(cfg: DictConfig) -> DictConfig:
    node = cfg.data.train
    visited = set()
    while "base_dataset" in node:
        identity = id(node)
        if identity in visited:
            raise RuntimeError("Cyclic base_dataset configuration.")
        visited.add(identity)
        node = node.base_dataset
    if not str(node.get("_target_", "")).endswith("RobotVideoDataset"):
        raise ValueError("RoboTwin evaluation requires a RobotVideoDataset config.")
    return node


def _normalize_proprio_batch(
    states: np.ndarray, processor: WINGProcessor
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("RoboTwin adapter expects one merged state key.")
    key = str(state_meta[0]["key"])
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 14 or not np.isfinite(states).all():
        raise ValueError(
            f"RoboTwin batched qpos must be finite [B,14], got {states.shape}."
        )
    batch = {"state": {key: torch.as_tensor(states)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][key]


def _normalize_proprio(state: np.ndarray, processor: WINGProcessor) -> torch.Tensor:
    """Normalize one state while preserving the original [1,14] result."""
    return _normalize_proprio_batch(np.asarray(state).reshape(1, -1), processor)


def _denormalize_actions(
    action: torch.Tensor, processor: WINGProcessor
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}.")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("RoboTwin adapter expects one merged action key.")
    key = str(action_meta[0]["key"])
    normalizer = processor.normalizer.normalizers["action"][key]
    result = normalizer.backward(action.float().cpu()).numpy()
    if result.ndim != 3 or result.shape[2] != 14 or not np.isfinite(result).all():
        raise ValueError(
            f"WING RoboTwin action must be finite [B,T,14], got {result.shape}."
        )
    return result.astype(np.float32, copy=False)


def _denormalize_action(action: torch.Tensor, processor: WINGProcessor) -> np.ndarray:
    """Denormalize one action chunk while preserving the original [T,14] API."""
    return _denormalize_actions(action, processor)[0]


class WINGRoboTwinPolicy:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: WINGProcessor,
        cfg: DictConfig,
        dataset_cfg: DictConfig,
        device: str,
    ) -> None:
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.device = str(device)
        self.pending_actions: deque[np.ndarray] = deque()
        self.action_horizon = int(
            cfg.EVALUATION.get("action_horizon") or (int(dataset_cfg.num_frames) - 1)
        )
        self.replan_steps = int(cfg.EVALUATION.get("replan_steps", 24))
        if self.action_horizon <= 0 or self.replan_steps <= 0:
            raise ValueError("action_horizon and replan_steps must be positive.")
        self.num_video_frames = (
            (int(dataset_cfg.num_frames) - 1)
            // int(dataset_cfg.action_video_freq_ratio)
            + 1
        )

    def reset(self) -> None:
        self.pending_actions.clear()

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def predict_chunks(
        self,
        observations: Sequence[dict[str, Any]],
        instructions: Sequence[str],
    ) -> np.ndarray:
        """Stateless batched policy inference returning finite ``[B,T,14]``."""
        observations = list(observations)
        instructions = [str(instruction) for instruction in instructions]
        if not observations:
            raise ValueError("RoboTwin batched inference requires at least one observation.")
        if len(observations) != len(instructions):
            raise ValueError(
                "RoboTwin observations/instructions batch sizes differ: "
                f"{len(observations)} != {len(instructions)}."
            )

        image = torch.stack(
            [compose_robotwin_rgb(observation, self.dataset_cfg) for observation in observations]
        )
        expected_size = tuple(int(x) for x in self.dataset_cfg.video_size)
        if tuple(image.shape[-2:]) != expected_size:
            raise ValueError(
                f"RoboTwin mosaic {tuple(image.shape[-2:])} != configured {expected_size}."
            )
        image = image.to(device=self.device, dtype=self.model.torch_dtype)
        proprio = _normalize_proprio_batch(
            np.stack([extract_robotwin_qpos(observation) for observation in observations]),
            self.processor,
        )
        common_kwargs: dict[str, Any] = {
            "action_horizon": self.action_horizon,
            "num_video_frames": self.num_video_frames,
            "negative_prompt": str(self.cfg.EVALUATION.get("negative_prompt", "")),
            "text_cfg_scale": float(self.cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            "num_inference_steps": int(self.cfg.EVALUATION.get("num_inference_steps", 10)),
            "sigma_shift": self.cfg.EVALUATION.get("sigma_shift"),
            "seed": None if self.cfg.get("seed") is None else int(self.cfg.seed),
            "rand_device": str(self.cfg.EVALUATION.get("rand_device", "cpu")),
            "tiled": bool(self.cfg.EVALUATION.get("tiled", False)),
        }

        def run_infer(infer: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            with torch.no_grad():
                output = infer(**kwargs)
            if not isinstance(output, dict) or "action" not in output:
                raise TypeError(
                    "WING inference must return a dict containing 'action'."
                )
            return output

        infer_action_batch = getattr(self.model, "infer_action_batch", None)
        if callable(infer_action_batch):
            output = run_infer(
                infer_action_batch,
                {
                    **common_kwargs,
                    "prompt": [
                        DEFAULT_PROMPT.format(task=instruction)
                        for instruction in instructions
                    ],
                    "input_image": image,
                    "proprio": proprio,
                },
            )
            result = _denormalize_actions(output["action"], self.processor)
        else:
            infer_action = getattr(self.model, "infer_action", None)
            if not callable(infer_action):
                raise TypeError(
                    "WING model provides neither infer_action_batch nor infer_action."
                )
            chunks = []
            for index, instruction in enumerate(instructions):
                output = run_infer(
                    infer_action,
                    {
                        **common_kwargs,
                        "prompt": DEFAULT_PROMPT.format(task=instruction),
                        "input_image": image[index : index + 1],
                        "proprio": proprio[index : index + 1],
                    },
                )
                chunks.append(_denormalize_actions(output["action"], self.processor)[0])
            result = np.stack(chunks)

        if result.shape[0] != len(observations):
            raise ValueError(
                "WING inference returned the wrong batch size: "
                f"{result.shape[0]} != {len(observations)}."
            )
        return result

    def _predict_chunk(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        return self.predict_chunks([observation], [instruction])[0]

    def next_action(
        self, observation: dict[str, Any] | None, instruction: str
    ) -> np.ndarray:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("A new RoboTwin replan requires an observation.")
            chunk = self._predict_chunk(observation, instruction)
            count = min(self.replan_steps, len(chunk))
            if count <= 0:
                raise ValueError("WING returned an empty action chunk.")
            self.pending_actions.extend(chunk[:count])
        return self.pending_actions.popleft()


def _load_config(args: dict[str, Any]) -> DictConfig:
    task = str(args.get("wing_task", "robotwin"))
    extra = [str(x) for x in args.get("wing_overrides", [])]
    config_dir = _model_runtime_root(args) / "configs"
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={task}",
                *extra,
                "model.spectral_plan.routing_mode=fixed",
            ],
        )
    with open_dict(cfg):
        cfg.ckpt = str(args["ckpt"])
        cfg.seed = int(args.get("seed", 0))
        cfg.model.load_text_encoder = True
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        cfg.EVALUATION = OmegaConf.create(
            {
                "action_horizon": args.get("action_horizon"),
                "replan_steps": int(args.get("replan_steps", 24)),
                "num_inference_steps": int(args.get("num_inference_steps", 10)),
                "sigma_shift": args.get("sigma_shift"),
                "text_cfg_scale": float(args.get("text_cfg_scale", 1.0)),
                "negative_prompt": str(args.get("negative_prompt", "")),
                "rand_device": str(args.get("rand_device", "cpu")),
                "tiled": bool(args.get("tiled", False)),
            }
        )
    return cfg


def get_model(args: dict[str, Any]) -> WINGRoboTwinPolicy:
    cfg = _load_config(args)
    device = str(args.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    precision = str(args.get("mixed_precision", cfg.get("mixed_precision", "bf16"))).lower()
    dtype = {"no": torch.float32, "fp32": torch.float32, "fp16": torch.float16}.get(
        precision, torch.bfloat16
    )
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    model = model.to(device).eval()

    stats_path = Path(
        os.path.expanduser(os.path.expandvars(str(args["dataset_stats_path"])))
    )
    if not stats_path.is_file():
        raise FileNotFoundError(f"RoboTwin dataset stats not found: {stats_path}")
    processor: WINGProcessor = instantiate(_robot_dataset_cfg(cfg).processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    return WINGRoboTwinPolicy(
        model=model,
        processor=processor,
        cfg=cfg,
        dataset_cfg=_robot_dataset_cfg(cfg),
        device=device,
    )


def reset_model(model: WINGRoboTwinPolicy) -> None:
    model.reset()


def eval(task_env: Any, model: WINGRoboTwinPolicy, observation: dict | None) -> None:
    action = model.next_action(observation, str(task_env.get_instruction()))
    task_env.take_action(action, action_type="qpos")
