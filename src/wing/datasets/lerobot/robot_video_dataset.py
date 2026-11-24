import hashlib
import json
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .packed_text_cache import PackedTextEmbeddingCache
from .vae_latent_cache import VaeLatentCache
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from wing.utils.logging_config import get_logger
from wing.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        stats_filename: str = "dataset_stats.json", # per-instance stats file (distinct per group for co-training)
        latent_action_sidecar_dirs: Optional[list[str]] = None,
        latent_action_key: str = "latent_action",
        vae_latent_cache_dir: Optional[str] = None,
        vae_latent_cache_required: bool = False,
        vae_latent_cache_max_open_shards: int = 8,
        vae_latent_cache_expected_vae_hash: Optional[str] = "e1de6c02cdac79f8b739f4d3698cd216",
        vae_latent_cache_allow_path_relocation: bool = False,
        validate_episode_files: bool = True,
        load_all_episodes_from_data_dir: bool = False,
        episode_indices_by_dataset: Optional[list[list[int]]] = None,
    ):
        cache_text = "" if vae_latent_cache_dir is None else str(vae_latent_cache_dir).strip()
        self.vae_latent_cache_dir = cache_text or None
        if vae_latent_cache_required and self.vae_latent_cache_dir is None:
            raise ValueError("vae_latent_cache_required=true but vae_latent_cache_dir is empty.")
        if self.vae_latent_cache_dir is not None and not is_training_set:
            raise ValueError("VAE latent cache is training-only; validation/evaluation must decode RGB.")
        self.vae_latent_cache = None
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            latent_action_sidecar_dirs=latent_action_sidecar_dirs,
            latent_action_key=latent_action_key,
            validate_episode_files=validate_episode_files,
            load_all_episodes_from_data_dir=load_all_episodes_from_data_dir,
            episode_indices_by_dataset=episode_indices_by_dataset,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(self.vae_latent_cache_dir is None)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.packed_text_cache = None
        if self.text_embedding_cache_dir is not None:
            text_cache_root = Path(str(self.text_embedding_cache_dir)).expanduser()
            if (text_cache_root / "packed_manifest.json").is_file():
                source_datasets = self.lerobot_dataset.multi_dataset._datasets
                if len(source_datasets) != 1:
                    raise ValueError(
                        "Task-index packed text cache currently requires exactly one "
                        f"dataset root, got {len(source_datasets)}."
                    )
                self.packed_text_cache = PackedTextEmbeddingCache(
                    text_cache_root,
                    expected_task_count=len(source_datasets[0].meta.tasks),
                    expected_context_len=self.context_len,
                )
                logger.info(
                    "Using verified packed text cache %s (tasks=%d, context=%dx%d).",
                    text_cache_root,
                    len(self.packed_text_cache),
                    self.packed_text_cache.context_len,
                    self.packed_text_cache.hidden_dim,
                )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, stats_filename))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, stats_filename))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

            if self.vae_latent_cache_dir is not None:
                if bool(getattr(processor, "image_augmentation", False)):
                    raise ValueError(
                        "VAE latent cache cannot be enabled with train-time image augmentation."
                    )
                signature = self.vae_latent_cache_signature
                self.vae_latent_cache = VaeLatentCache(
                    self.vae_latent_cache_dir,
                    expected_dataset_signature=signature,
                    expected_num_samples=len(self),
                    expected_vae_hash=vae_latent_cache_expected_vae_hash,
                    require_complete=True,
                    max_open_shards=vae_latent_cache_max_open_shards,
                    allow_path_relocation=vae_latent_cache_allow_path_relocation,
                )
                processor.set_image_processing_enabled(False)
                logger.info(
                    "Using verified VAE latent cache %s (shape=%s, samples=%d); RGB decode disabled.",
                    self.vae_latent_cache_dir,
                    self.vae_latent_cache.latent_shape,
                    len(self.vae_latent_cache),
                )
        elif self.vae_latent_cache_dir is not None:
            raise ValueError("VAE latent cache requires a WING processor.")

    @property
    def vae_latent_cache_signature(self) -> str:
        datasets = []
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            episodes = None if dataset.episodes is None else [int(x) for x in dataset.episodes]
            datasets.append({
                "root": os.path.realpath(os.fspath(dataset.root)),
                "episodes": episodes,
                "num_frames": int(dataset.num_frames),
            })
        processor = self.lerobot_dataset.processor
        payload = {
            "schema_version": 1,
            "datasets": datasets,
            "length": len(self),
            "num_frames": int(self.num_frames),
            "global_sample_stride": int(self.lerobot_dataset.global_sample_stride),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "video_sample_indices": self.video_sample_indices,
            "video_size": [int(x) for x in self.video_size],
            "concat_multi_camera": self.concat_multi_camera,
            "image_shape_meta": self.lerobot_dataset.shape_meta["images"],
            "train_transforms": repr(getattr(processor, "train_transforms", None)),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        image_is_pad = image_is_pad[self.video_sample_indices]
        video = None
        video_latents = None
        if self.vae_latent_cache is not None:
            video_latents = self.vae_latent_cache[int(sample["idx"])]
        else:
            video = sample["pixel_values"]  # [T,C,H,W] or [num_cameras,T,C,H,W]
            num_cameras = 1
            if video.ndim == 5:
                video = video[:, self.video_sample_indices, :, :, :]
                num_cameras, T_video, C, H, W = video.shape
            else:
                assert video.ndim == 4, f"Expected video to have shape [T,C,H,W], got {video.shape}"
                video = video[self.video_sample_indices, :, :, :]
                T_video, C, H, W = video.shape
            video = video.view(num_cameras, T_video, C, H, W)
            if self.concat_multi_camera == "robotwin":
                if num_cameras != 3:
                    raise ValueError(
                        f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                    )
                cam_top = transforms_F.resize(video[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
                cam_left = transforms_F.resize(video[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
                cam_right = transforms_F.resize(video[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
                video = torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
            elif num_cameras > 1:
                if self.concat_multi_camera == "horizontal":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
                elif self.concat_multi_camera == "vertical":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
                else:
                    raise ValueError(
                        f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                        "Expected one of: horizontal, vertical, robotwin."
                    )
            else:
                video = video.squeeze(0)
            video = self.normalize_transform(self.crop_transform(self.resize_transform(video)))
            video = video.permute(1, 0, 2, 3)

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        action_mask = sample.get("action_mask", None)
        proprio_mask = sample.get("proprio_mask", None)
        if proprio_mask is not None:
            proprio_mask = proprio_mask[:-1, :]
        video_num_frames = len(self.video_sample_indices)
        if video_num_frames <= 1:
            raise ValueError(f"Video must have at least 2 frames, got {video_num_frames}.")
        if action.shape[0] % (video_num_frames - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by video transitions, got {action.shape[0]} and {video_num_frames - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(
            instruction, task_index=sample.get("task_index")
        )
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if video_latents is not None:
            data["video_latents"] = video_latents
        else:
            data["video"] = video
        if action_mask is not None:
            data["action_mask"] = action_mask
        if proprio_mask is not None:
            data["proprio_mask"] = proprio_mask
        return data

    def _get_cached_text_context(self, prompt: str, *, task_index=None):
        packed_text_cache = getattr(self, "packed_text_cache", None)
        if packed_text_cache is not None and self.override_instruction is None:
            if task_index is None:
                raise ValueError(
                    "Packed text cache requires the sample's dataset-local task_index."
                )
            index = int(torch.as_tensor(task_index).reshape(-1)[0].item())
            return packed_text_cache[index]

        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            if packed_text_cache is not None and self.override_instruction is not None:
                raise FileNotFoundError(
                    "override_instruction cannot be resolved by the task-index packed "
                    f"text cache, and its legacy prompt cache is missing: {cache_path}. "
                    "Precompute the override prompt as a legacy hash .pt entry."
                )
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
