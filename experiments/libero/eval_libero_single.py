from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero.action_ensembler import ActionEnsembler
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from wing.datasets.lerobot.processors.wing_processor import WINGProcessor
from wing.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wing.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wing.utils.config_resolvers import register_default_resolvers
from wing.utils.pytorch_utils import set_global_seed


register_default_resolvers()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _robot_dataset_cfg(cfg: DictConfig) -> DictConfig:
    node = cfg.data.train
    visited: set[int] = set()
    while "base_dataset" in node:
        identity = id(node)
        if identity in visited:
            raise RuntimeError("Cyclic base_dataset configuration.")
        visited.add(identity)
        node = node.base_dataset
    if not str(node.get("_target_", "")).endswith("RobotVideoDataset"):
        raise ValueError("LIBERO evaluation requires a RobotVideoDataset config.")
    return node


def _model_dtype(value: str) -> torch.dtype:
    precision = str(value).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision: {value!r}.")


def _dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates = [] if explicit is None else [Path(os.path.expandvars(str(explicit))).expanduser()]
    checkpoint = Path(os.path.expandvars(str(cfg.ckpt))).expanduser()
    candidates.extend(parent / "dataset_stats.json" for parent in list(checkpoint.parents)[:4])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Dataset statistics were not found; set EVALUATION.dataset_stats_path."
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    source_width, source_height = pil_image.size
    scale = max(width / source_width, height / source_height)
    resized = pil_image.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.BILINEAR,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _normalize_proprio(proprio: np.ndarray, processor: WINGProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("LIBERO evaluation expects one merged state key.")
    state_key = state_meta[0]["key"]
    batch = {
        "state": {
            state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)
        }
    }
    batch = processor.action_state_transform(batch)
    return processor.normalizer.forward(batch)["state"][state_key]


def _model_input(
    obs: dict[str, Any],
    dataset_cfg: DictConfig,
    processor: WINGProcessor,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    images = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if int(processor.num_output_cameras) != 2 or len(image_meta) < 2:
        raise ValueError("The released LIBERO policy expects agent and wrist cameras.")
    primary_height, primary_width = map(int, image_meta[0]["shape"][1:])
    wrist_height, wrist_width = map(int, image_meta[1]["shape"][1:])
    primary = _center_crop_resize(images["image"], primary_width, primary_height)
    wrist = _center_crop_resize(images["wrist_image"], wrist_width, wrist_height)
    concatenation = str(dataset_cfg.get("concat_multi_camera", "horizontal"))
    if concatenation == "horizontal":
        rgb = np.concatenate([primary, wrist], axis=1)
    elif concatenation == "vertical":
        rgb = np.concatenate([primary, wrist], axis=0)
    else:
        raise ValueError(f"Unsupported camera concatenation: {concatenation!r}.")
    expected_height, expected_width = map(int, dataset_cfg.video_size)
    if rgb.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"LIBERO image shape {rgb.shape[:2]} does not match "
            f"configured {(expected_height, expected_width)}."
        )
    image = torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype).mul_(2.0 / 255.0).sub_(1.0)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return image, _normalize_proprio(state, processor), images


def _denormalize_action(action: torch.Tensor, processor: WINGProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if action.ndim != 3 or len(action_meta) != 1:
        raise ValueError("LIBERO action output must be [B,T,D] with one action key.")
    key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][key]
    return normalizer.backward(action.float().cpu()).numpy()


def _predict_action_chunk(
    *,
    obs: dict[str, Any],
    task_description: str,
    model: torch.nn.Module,
    processor: WINGProcessor,
    dataset_cfg: DictConfig,
    cfg: DictConfig,
    action_horizon: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image, proprio, images = _model_input(
        obs, dataset_cfg, processor, device, model.torch_dtype
    )
    kwargs = {
        "prompt": DEFAULT_PROMPT.format(task=task_description),
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "proprio": proprio,
        "sigma_shift": cfg.EVALUATION.get("sigma_shift"),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        kwargs["num_video_frames"] = (
            (int(dataset_cfg.num_frames) - 1)
            // int(dataset_cfg.action_video_freq_ratio)
            + 1
        )
    with torch.inference_mode():
        prediction = model.infer_action(**kwargs)
    action = _denormalize_action(prediction["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", True)):
        action[..., -1] = np.sign(action[..., -1])
    return action, images


def _maximum_steps(suite_name: str) -> int:
    return {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }[suite_name]


def _run_episode(
    *,
    env: Any,
    initial_state: Any,
    task_description: str,
    model: torch.nn.Module,
    processor: WINGProcessor,
    dataset_cfg: DictConfig,
    cfg: DictConfig,
    episode_index: int,
    action_horizon: int,
    device: str,
) -> tuple[bool, list[dict[str, np.ndarray]]]:
    maximum_steps = int(
        cfg.EVALUATION.get("max_steps") or _maximum_steps(cfg.EVALUATION.task_suite_name)
    )
    replan_steps = int(cfg.EVALUATION.replan_steps)
    wait_steps = int(cfg.EVALUATION.num_steps_wait)
    use_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    ensembler = ActionEnsembler() if use_ensembler else None
    if ensembler is not None:
        ensembler.reset()

    env.reset()
    obs = env.set_init_state(initial_state)
    pending_actions: list[list[float]] = []
    replay_images: list[dict[str, np.ndarray]] = []
    done = False
    timestep = 0
    progress = tqdm(total=maximum_steps + wait_steps, desc=f"Episode {episode_index + 1}")
    while timestep < maximum_steps + wait_steps:
        progress.update(1)
        if timestep < wait_steps:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            timestep += 1
            continue
        if not pending_actions:
            action_chunk, images = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                dataset_cfg=dataset_cfg,
                cfg=cfg,
                action_horizon=action_horizon,
                device=device,
            )
            if ensembler is None:
                pending_actions = action_chunk[:replan_steps].tolist()
            else:
                ensembler.add_actions(action_chunk, timestep)
                pending_actions = [
                    ensembler.get_action(index).tolist()
                    for index in range(timestep, timestep + replan_steps)
                ]
        else:
            images = get_libero_image(obs)
        replay_images.append(images)
        obs, _, done, _ = env.step(pending_actions.pop(0))
        timestep += 1
        if done:
            break
    progress.close()
    return bool(done), replay_images


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> dict[str, Any]:
    started = time.time()
    state = PartialState()
    state.config = cfg
    if cfg.ckpt is None:
        raise ValueError("ckpt is required.")
    if int(cfg.EVALUATION.get("env_num", 1)) != 1:
        raise ValueError("The LIBERO worker supports one environment per process.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    device = str(cfg.EVALUATION.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(cfg.get("mixed_precision", "bf16")),
        device=device,
    )
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()

    dataset_cfg = _robot_dataset_cfg(cfg)
    processor: WINGProcessor = instantiate(dataset_cfg.processor).eval()
    stats_path = _dataset_stats_path(cfg)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    logging.info("Using dataset statistics: %s", stats_path)

    action_horizon = int(
        cfg.EVALUATION.get("action_horizon") or (int(dataset_cfg.num_frames) - 1)
    )
    output_root = Path(str(cfg.EVALUATION.output_dir)).expanduser()
    video_dir = output_root / str(cfg.EVALUATION.task_suite_name) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[str(cfg.EVALUATION.task_suite_name)]()
    task = suite.get_task(int(cfg.EVALUATION.task_id))
    initial_states = list(suite.get_task_init_states(int(cfg.EVALUATION.task_id)))
    trials = int(cfg.EVALUATION.num_trials)
    while len(initial_states) < trials:
        initial_states.extend(initial_states[: trials - len(initial_states)])

    env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results = {
        "task_suite": str(cfg.EVALUATION.task_suite_name),
        "task_id": int(cfg.EVALUATION.task_id),
        "task_description": description,
        "successes": 0,
        "total_episodes": trials,
        "success_episodes": [],
        "failure_episodes": [],
    }
    for trial in range(trials):
        success, frames = _run_episode(
            env=env,
            initial_state=initial_states[trial],
            task_description=description,
            model=model,
            processor=processor,
            dataset_cfg=dataset_cfg,
            cfg=cfg,
            episode_index=trial,
            action_horizon=action_horizon,
            device=device,
        )
        key = "success_episodes" if success else "failure_episodes"
        results[key].append(trial)
        results["successes"] += int(success)
        save_rollout_video(
            video_dir,
            frames,
            f"task{cfg.EVALUATION.task_id}_trial{trial}",
            success,
            description,
        )
    env.close()
    results["duration"] = time.time() - started
    result_dir = output_root / str(cfg.EVALUATION.task_suite_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
    result_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print(f"Task {cfg.EVALUATION.task_id}: {results['successes']}/{trials} successes")
    return results


if __name__ == "__main__":
    main()
