"""Pre-encode deterministic RobotVideoDataset clips into validated BF16 shards.

Example:
  LIBERO_VAE_CACHE=datasets/libero/libero_v2.1/vae_cache \
    torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
    task=libero_vae_preencode
"""

from __future__ import annotations

import json
import hashlib
import ctypes
import logging
import os
import random
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import ColorJitter, InterpolationMode, Resize
import torchvision.transforms.functional as transforms_F
from tqdm import tqdm

from wing.datasets.lerobot.vae_latent_cache import (
    VAE_LATENT_CACHE_SCHEMA_VERSION,
    VaeLatentCache,
    write_bfloat16_shard,
)
from wing.datasets.lerobot.vae_preencode_video_dataset import (
    RobotVaePreencodeVideoDataset,
)
from wing.datasets.ego_stage1_video import EgoStage1LogicalVideoDataset
from wing.models.wan22.helpers.loader import (
    WAN22_MODEL_REGISTRY,
    _load_registered_model,
    _resolve_configs,
)
from wing.utils import misc
from wing.utils.config_resolvers import register_default_resolvers
from wing.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)
ROBOT_DATASET_TARGET = "wing.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
SPECTRAL_DATASET_TARGET = "wing.datasets.spectral_targets.SpectralTargetDataset"
_LIBC: ctypes.CDLL | None = None
_MALLOC_TRIM: Any | None = None
ROBOTWIN_CUDA_PREPROCESS = "robotwin_cuda_v1"


def _color_jitter_contract(
    color_jitter: dict[str, float], *, execution: str | None = None
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "name": "color_jitter",
        **{
            name: float(color_jitter.get(name, 0.0))
            for name in ("brightness", "contrast", "saturation", "hue")
        },
        "application": "post_compose_pre_normalize",
        "shared_across_frames_and_cameras": True,
        "seed_strategy": "preencode_seed_plus_source_index",
    }
    if execution is not None:
        contract["execution"] = str(execution)
    return contract


def _validate_robotwin_cuda_preprocess(dataset: Any) -> dict[str, Any]:
    """Fail closed unless the deferred dataset has the exact CUDA-v1 recipe."""
    if not torch.cuda.is_available():
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires CUDA.")
    if getattr(dataset, "concat_multi_camera", None) != "robotwin":
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} only supports RoboTwin camera composition.")
    if tuple(getattr(dataset, "video_size", ())) != (384, 320):
        raise ValueError(
            f"{ROBOTWIN_CUDA_PREPROCESS} requires video_size=[384,320], got "
            f"{getattr(dataset, 'video_size', None)}."
        )
    if not bool(getattr(dataset, "defer_image_processing", False)):
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires deferred raw-camera samples.")
    if not bool(getattr(dataset, "batch_decode_required", False)):
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires strict batched union decode.")
    if len(tuple(getattr(dataset, "video_sample_indices", ()))) != 9:
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires exactly nine RGB endpoints.")
    image_meta = list(getattr(dataset, "image_meta", []))
    if len(image_meta) != 3:
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires exactly three cameras.")
    for meta in image_meta:
        if tuple(int(value) for value in meta.get("raw_shape", ())) != (3, 480, 640):
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} requires raw camera shape [3,480,640], "
                f"got {meta.get('raw_shape')} for {meta.get('key')}."
            )

    transforms = getattr(dataset, "train_transforms", None)
    transform_groups: list[Any]
    if isinstance(transforms, (dict, DictConfig)):
        transform_groups = [transforms[str(meta["key"])] for meta in image_meta]
    else:
        transform_groups = [transforms]
    for group in transform_groups:
        if group is None or len(group) != 2:
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} requires exactly ToTensor + Resize transforms."
            )
        to_tensor, resize = group
        if not (
            type(to_tensor).__module__ == "wing.datasets.lerobot.transforms.image"
            and type(to_tensor).__name__ == "ToTensor"
        ):
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} expected WING ToTensor, got {to_tensor!r}."
            )
        if not isinstance(resize, Resize):
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} expected torchvision Resize, got {resize!r}."
            )
        if tuple(int(value) for value in resize.size) != (240, 320):
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} expected camera Resize [240,320], got {resize.size}."
            )
        if resize.interpolation != InterpolationMode.BILINEAR or resize.antialias is not True:
            raise ValueError(
                f"{ROBOTWIN_CUDA_PREPROCESS} requires bilinear antialiased camera Resize."
            )

    return {
        "name": ROBOTWIN_CUDA_PREPROCESS,
        "execution": "rank_main_process_cuda",
        "batch_decode_required": True,
        "input": {"dtype": "uint8", "shape": [3, 9, 3, 480, 640]},
        "camera_transform": {
            "to_tensor": "float32_div_255",
            "resize": [240, 320],
            "interpolation": "bilinear",
            "antialias": True,
        },
        "composition": {
            "layout": "robotwin",
            "top_resize": [256, 320],
            "wrist_resize": [128, 160],
            "interpolation": "bilinear",
            "antialias": True,
        },
        "output": {
            "shape": [3, 9, 384, 320],
            "normalize_mean": 0.5,
            "normalize_std": 0.5,
            "dtype": "bfloat16",
        },
    }


def _sample_color_jitter_params(
    color_jitter: ColorJitter, *, seed: int, source_index: int
) -> tuple[torch.Tensor, float | None, float | None, float | None, float | None]:
    # ColorJitter parameters stay on CPU so this exactly preserves the existing
    # seed+source-index recipe independently of rank, worker, batch order, and resume.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((int(seed) + int(source_index)) % (2**63))
        return ColorJitter.get_params(
            color_jitter.brightness,
            color_jitter.contrast,
            color_jitter.saturation,
            color_jitter.hue,
        )


@torch.no_grad()
def _robotwin_cuda_preprocess_batch(
    raw_cameras: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    device: str,
    color_jitter: dict[str, float] | None,
    augmentation_seed: int,
) -> torch.Tensor:
    """Turn deferred RoboTwin uint8 cameras into Wan input entirely on CUDA."""
    def resize_frames(tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
        leading = tuple(tensor.shape[:-3])
        channels = int(tensor.shape[-3])
        flattened = tensor.reshape(-1, channels, *tensor.shape[-2:])
        resized = transforms_F.resize(
            flattened,
            size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return resized.reshape(*leading, channels, *size)

    raw = torch.as_tensor(raw_cameras)
    indices = torch.as_tensor(source_indices, dtype=torch.long).reshape(-1).cpu()
    if raw.dtype != torch.uint8 or raw.ndim != 6:
        raise ValueError(
            f"CUDA RoboTwin preprocessing expected uint8 [B,3,T,3,480,640], "
            f"got dtype={raw.dtype} shape={tuple(raw.shape)}."
        )
    batch_size, num_cameras, num_frames, channels, height, width = raw.shape
    if (num_cameras, num_frames, channels, height, width) != (3, 9, 3, 480, 640):
        raise ValueError(
            "CUDA RoboTwin preprocessing expected [B,3,9,3,480,640], "
            f"got {tuple(raw.shape)}."
        )
    if indices.numel() != batch_size:
        raise ValueError(
            f"CUDA RoboTwin source-index batch has {indices.numel()} entries for B={batch_size}."
        )
    if not str(device).startswith("cuda"):
        raise ValueError(f"{ROBOTWIN_CUDA_PREPROCESS} requires a CUDA device, got {device!r}.")

    # Match ToTensor followed by torchvision Resize([240,320]). Leading
    # B/camera/time dimensions are flattened explicitly for CUDA interpolate.
    cameras = raw.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    cameras = resize_frames(cameras, [240, 320])
    top = resize_frames(cameras[:, 0], [256, 320])
    left = resize_frames(cameras[:, 1], [128, 160])
    right = resize_frames(cameras[:, 2], [128, 160])
    videos = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
    if tuple(videos.shape[1:]) != (9, 3, 384, 320):
        raise ValueError(f"CUDA RoboTwin mosaic produced {tuple(videos.shape)}.")

    if color_jitter is not None:
        jitter = ColorJitter(**color_jitter)
        jittered: list[torch.Tensor] = []
        for video, source_index in zip(videos.unbind(0), indices.tolist(), strict=True):
            fn_idx, brightness, contrast, saturation, hue = _sample_color_jitter_params(
                jitter, seed=augmentation_seed, source_index=source_index
            )
            for fn_id in fn_idx.tolist():
                if fn_id == 0 and brightness is not None:
                    video = transforms_F.adjust_brightness(video, brightness)
                elif fn_id == 1 and contrast is not None:
                    video = transforms_F.adjust_contrast(video, contrast)
                elif fn_id == 2 and saturation is not None:
                    video = transforms_F.adjust_saturation(video, saturation)
                elif fn_id == 3 and hue is not None:
                    video = transforms_F.adjust_hue(video, hue)
            jittered.append(video)
        videos = torch.stack(jittered, dim=0)

    videos = videos.sub_(0.5).div_(0.5).permute(0, 2, 1, 3, 4).contiguous()
    expected = (batch_size, 3, 9, 384, 320)
    if tuple(videos.shape) != expected:
        raise ValueError(f"CUDA RoboTwin video shape {tuple(videos.shape)} != {expected}.")
    if not torch.isfinite(videos).all():
        raise ValueError("CUDA RoboTwin preprocessing produced non-finite video values.")
    return videos.to(dtype=torch.bfloat16)


def _collate_and_malloc_trim(batch: list[Any]) -> Any:
    """Collate a worker batch, then return unused glibc arenas to the OS."""
    result = torch.utils.data.default_collate(batch)
    global _LIBC, _MALLOC_TRIM
    if _MALLOC_TRIM is None:
        _LIBC = ctypes.CDLL(None)
        malloc_trim = _LIBC.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        _MALLOC_TRIM = malloc_trim
    _MALLOC_TRIM(0)
    return result


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA is available but no visible GPU devices were found.")
        if local_world_size > device_count or local_rank >= device_count:
            raise RuntimeError(
                "Batched VAE precompute requires exactly one rank per visible GPU: "
                f"LOCAL_WORLD_SIZE={local_world_size}, devices={device_count}."
            )
        local_device_rank = local_rank
        torch.cuda.set_device(local_device_rank)
    else:
        local_device_rank = 0
    if not dist.is_initialized():
        timeout_hours = float(os.environ.get("VAE_DIST_TIMEOUT_HOURS", "24"))
        if timeout_hours <= 0:
            raise ValueError("VAE_DIST_TIMEOUT_HOURS must be positive.")
        # Global world size may exceed the GPU count on any one node.  NCCL is
        # still correct as long as LOCAL_WORLD_SIZE is one rank per local GPU.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            # Shards are intentionally distributed round-robin. Heterogeneous
            # source codecs can leave the fastest rank waiting well over the
            # NCCL default 10 minutes at the final completion barrier.
            timeout=timedelta(hours=timeout_hours),
        )
    return True, dist.get_rank(), dist.get_world_size(), local_device_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _iter_robot_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        target = str(node.get("_target_", ""))
        if target == SPECTRAL_DATASET_TARGET and node.get("logical_index_manifest") is not None:
            # Phase-I precompute owns train logical positions. Val remains online.
            if str(node.get("logical_split", "train")) == "train":
                yield path, node
            return
        if target == ROBOT_DATASET_TARGET:
            yield path, node
            return
        for key, value in node.items():
            yield from _iter_robot_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for index, value in enumerate(node):
            yield from _iter_robot_dataset_nodes(value, f"{path}[{index}]")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _bounded_dataset_signature(signature: str, max_samples: int | None) -> str:
    if max_samples is None:
        return str(signature)
    payload = {
        "schema_version": 1,
        "full_dataset_signature": str(signature),
        "probe_prefix_records": int(max_samples),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dataset_and_cache_dir(
    node: DictConfig,
    fallback_cache_dir: str | None,
    *,
    color_jitter: dict[str, float] | None = None,
    augmentation_seed: int = 0,
    image_preprocess: str = "cpu",
):
    node_copy = OmegaConf.create(OmegaConf.to_container(node, resolve=True))
    configured = node_copy.get("vae_latent_cache_dir")
    cache_dir = str(configured).strip() if configured is not None else ""
    cache_dir = cache_dir or ("" if fallback_cache_dir is None else fallback_cache_dir.strip())
    if not cache_dir:
        raise ValueError(
            "Each RobotVideoDataset needs vae_latent_cache_dir (or +preencode_cache_dir=...)."
        )
    node_copy.vae_latent_cache_dir = None
    node_copy.vae_latent_cache_required = False
    if str(node_copy.get("_target_", "")) == SPECTRAL_DATASET_TARGET:
        if color_jitter is not None:
            raise ValueError("Pre-encode ColorJitter currently supports RobotVideoDataset only.")
        if image_preprocess != "cpu":
            raise ValueError("Deferred CUDA image preprocessing supports RobotVideoDataset only.")
        base = node_copy.base_dataset
        processor = instantiate(base.processor)
        processor.train()
        dataset = EgoStage1LogicalVideoDataset(
            logical_index_manifest=node_copy.logical_index_manifest,
            logical_split=node_copy.logical_split,
            dataset_roots=list(base.dataset_dirs),
            image_shape_meta=OmegaConf.to_container(base.shape_meta.images, resolve=True),
            train_transforms=processor.train_transforms,
            num_frames=int(base.num_frames),
            global_sample_stride=int(base.global_sample_stride),
            action_video_freq_ratio=int(base.action_video_freq_ratio),
            video_size=list(base.video_size),
            concat_multi_camera=base.concat_multi_camera,
            expected_fps=30.0,
            anchor_stride=int(node_copy.anchor_stride),
            window_span=int(node_copy.full_window_span),
        )
    else:
        if image_preprocess not in {"cpu", ROBOTWIN_CUDA_PREPROCESS}:
            raise ValueError(f"Unsupported VAE image preprocess mode {image_preprocess!r}.")
        cuda_preprocess = image_preprocess == ROBOTWIN_CUDA_PREPROCESS
        robot_dataset = instantiate(node_copy)
        processor = robot_dataset.lerobot_dataset.processor
        dataset = RobotVaePreencodeVideoDataset(
            robot_dataset,
            color_jitter=None if cuda_preprocess else color_jitter,
            augmentation_seed=augmentation_seed,
            defer_image_processing=cuda_preprocess,
        )
        if cuda_preprocess:
            dataset.vae_preencode_image_preprocess = _validate_robotwin_cuda_preprocess(dataset)
            if color_jitter is not None:
                dataset.vae_preencode_augmentation = _color_jitter_contract(
                    color_jitter, execution=ROBOTWIN_CUDA_PREPROCESS
                )
        else:
            dataset.vae_preencode_image_preprocess = None
    if processor is None:
        raise ValueError("RobotVideoDataset must have a processor for VAE pre-encoding.")
    if bool(getattr(processor, "image_augmentation", False)):
        raise ValueError("Refusing VAE pre-encode with stochastic train-time image augmentation.")
    return dataset, Path(cache_dir).expanduser().resolve()


@torch.no_grad()
def _encode(vae, videos: torch.Tensor, device: str) -> torch.Tensor:
    videos = videos.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    latents = vae.encode_batch(videos, device=device, tiled=False)
    if latents.ndim != 5 or latents.shape[0] != videos.shape[0]:
        raise ValueError(f"Unexpected VAE output shape {tuple(latents.shape)}.")
    if not torch.isfinite(latents).all():
        raise ValueError("VAE produced non-finite cached latents.")
    return latents.to(device="cpu", dtype=torch.bfloat16).contiguous()


class _ShardBatchSampler(Sampler[list[int]]):
    """Yield batches without crossing a cache-shard boundary."""

    def __init__(self, shards: list[dict[str, Any]], batch_size: int) -> None:
        self.shards = list(shards)
        self.batch_size = int(batch_size)

    def __iter__(self):
        for shard in self.shards:
            start, end = int(shard["start"]), int(shard["end"])
            for offset in range(start, end, self.batch_size):
                yield list(range(offset, min(offset + self.batch_size, end)))

    def __len__(self) -> int:
        return sum(
            (int(shard["end"]) - int(shard["start"]) + self.batch_size - 1)
            // self.batch_size
            for shard in self.shards
        )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def _process_rss_gib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024**2)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return 0.0


def _validate_source_identity(batch: dict[str, Any], expected: torch.Tensor) -> None:
    if "source_index" not in batch:
        raise RuntimeError("Strict VAE precompute batch lacks source_index.")
    actual = torch.as_tensor(batch["source_index"], dtype=torch.long).reshape(-1).cpu()
    expected = torch.as_tensor(expected, dtype=torch.long).reshape(-1).cpu()
    if not torch.equal(actual, expected):
        raise RuntimeError(
            "Strict VAE source identity differs from requested indices: "
            f"actual={actual.tolist()} expected={expected.tolist()}."
        )


def _prepare_preencode_videos(
    batch: dict[str, Any],
    dataset: Any,
    *,
    device: str,
    seed: int,
) -> torch.Tensor:
    preprocess = getattr(dataset, "vae_preencode_image_preprocess", None)
    if preprocess is None:
        if "video" not in batch:
            raise RuntimeError("VAE precompute batch lacks processed video.")
        return torch.as_tensor(batch["video"])
    if preprocess.get("name") != ROBOTWIN_CUDA_PREPROCESS:
        raise ValueError(f"Unsupported VAE image preprocess contract: {preprocess!r}.")
    if "raw_cameras" not in batch or "video" in batch:
        raise RuntimeError(
            "Deferred CUDA VAE batch must contain raw_cameras and must not contain video."
        )
    augmentation = getattr(dataset, "vae_preencode_augmentation", None)
    color_jitter = None
    if augmentation is not None:
        if (
            augmentation.get("name") != "color_jitter"
            or augmentation.get("execution") != ROBOTWIN_CUDA_PREPROCESS
        ):
            raise ValueError(f"Unsupported CUDA augmentation contract: {augmentation!r}.")
        color_jitter = {
            name: float(augmentation[name])
            for name in ("brightness", "contrast", "saturation", "hue")
        }
    return _robotwin_cuda_preprocess_batch(
        batch["raw_cameras"],
        batch["source_index"],
        device=device,
        color_jitter=color_jitter,
        augmentation_seed=seed,
    )


def _shared_filesystem_preflight(cache_dir: Path, rank: int, world_size: int) -> None:
    """Fail before encoding if all ranks do not observe one shared cache root."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if world_size <= 1 or not dist.is_initialized():
        return
    run_key = "|".join(
        [
            os.environ.get("MASTER_ADDR", ""),
            os.environ.get("MASTER_PORT", ""),
            str(world_size),
            os.environ.get("TORCHELASTIC_RUN_ID", "none"),
        ]
    )
    digest = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:20]
    sentinel = cache_dir / f".shared_fs_preflight_{digest}.json"
    expected = {"digest": digest, "world_size": int(world_size)}
    if rank == 0:
        _atomic_json(sentinel, expected)
    dist.barrier()
    try:
        if not sentinel.is_file():
            raise RuntimeError(
                f"Cache directory is not shared across ranks; missing sentinel {sentinel}."
            )
        actual = json.loads(sentinel.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"Shared-filesystem sentinel mismatch at {sentinel}.")
    finally:
        dist.barrier()
        if rank == 0:
            sentinel.unlink(missing_ok=True)
        dist.barrier()


def _validate_logical_preencode_batch(
    batch: dict[str, Any],
    *,
    logical_indices: dict[str, torch.Tensor],
    logical_positions: torch.Tensor,
) -> None:
    mappings = {
        "latent_source_index": "source_indices",
        "latent_source_dataset_index": "dataset_indices",
        "latent_source_episode_index": "episode_indices",
        "latent_source_frame_index": "frame_indices",
    }
    for sample_key, identity_key in mappings.items():
        if sample_key not in batch:
            raise RuntimeError(f"Logical VAE batch lacks {sample_key!r}.")
        actual = torch.as_tensor(batch[sample_key], dtype=torch.long).reshape(-1).cpu()
        expected = logical_indices[identity_key].index_select(0, logical_positions).cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(
                f"Logical VAE batch {sample_key} differs from requested logical positions."
            )
    latent_mask = batch.get("latent_target_mask")
    if latent_mask is None or not bool(torch.as_tensor(latent_mask, dtype=torch.bool).all()):
        raise RuntimeError("Logical VAE precompute requires an all-valid latent action H8 mask.")
    image_pad = batch.get("image_is_pad")
    if image_pad is None or bool(torch.as_tensor(image_pad, dtype=torch.bool).any()):
        raise RuntimeError("Logical VAE precompute refuses padded/missing RGB endpoints.")


def _precompute_one(
    dataset,
    cache_dir: Path,
    *,
    vae,
    vae_path: str,
    vae_hash: str,
    device: str,
    rank: int,
    world_size: int,
    batch_size: int,
    shard_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    seed: int = 20260817,
    log_every_batches: int = 20,
    overwrite: bool = False,
    max_samples: int | None = None,
) -> None:
    set_preencode_seed = getattr(dataset, "set_preencode_seed", None)
    if set_preencode_seed is not None:
        set_preencode_seed(seed)
    _shared_filesystem_preflight(cache_dir, rank, world_size)
    full_num_samples = len(dataset)
    if full_num_samples <= 0:
        raise ValueError("VAE precompute requires a non-empty dataset.")
    if int(shard_size) % int(batch_size) != 0:
        raise ValueError(
            "preencode_shard_size must be divisible by preencode_batch_size so "
            "source_index % batch_size has a canonical batch position."
        )
    if max_samples is not None and not 0 < int(max_samples) <= full_num_samples:
        raise ValueError(
            f"preencode_max_samples must be in [1,{full_num_samples}], got {max_samples}."
        )
    num_samples = full_num_samples if max_samples is None else int(max_samples)
    signature = _bounded_dataset_signature(
        dataset.vae_latent_cache_signature,
        None if num_samples == full_num_samples else num_samples,
    )
    encoding_contract = {
        "version": 2,
        "encoder_mode": "true_batch",
        "source_reader": type(dataset).__name__,
        "video_sample_indices": [
            int(value) for value in getattr(dataset, "video_sample_indices", [])
        ],
        "video_backend": str(getattr(dataset, "video_backend", "pyav")),
        "strict_source_identity": True,
        "batch_size": int(batch_size),
        "seed": int(seed),
    }
    augmentation_contract = getattr(dataset, "vae_preencode_augmentation", None)
    if augmentation_contract is not None:
        encoding_contract["version"] = 3
        encoding_contract["augmentation"] = dict(augmentation_contract)
    image_preprocess_contract = getattr(dataset, "vae_preencode_image_preprocess", None)
    if image_preprocess_contract is not None:
        encoding_contract["version"] = 4
        encoding_contract["image_preprocess"] = dict(image_preprocess_contract)
    logical_metadata = getattr(dataset, "vae_logical_metadata", None)
    logical_indices = getattr(dataset, "vae_logical_indices", None)
    reader_kwargs = {}
    if logical_metadata is not None or logical_indices is not None:
        if logical_metadata is None or logical_indices is None:
            raise ValueError("Logical VAE precompute needs both metadata and identity tensors.")
        logical_metadata = dict(logical_metadata)
        logical_indices = {
            name: torch.as_tensor(value, dtype=torch.long)[:num_samples].clone()
            for name, value in logical_indices.items()
        }
        if num_samples != full_num_samples:
            logical_metadata["num_samples"] = num_samples
            logical_metadata["probe_prefix_records"] = num_samples
        reader_kwargs = {
            "expected_logical_metadata": logical_metadata,
            "expected_logical_indices": logical_indices,
        }
    manifest_path = cache_dir / "manifest.json"
    if overwrite:
        stale = manifest_path.is_file() or any(cache_dir.glob("shard_*.bf16"))
        if stale:
            raise ValueError(
                "preencode_overwrite=true is fail-closed for an existing cache. "
                "Use a new empty cache directory (or explicitly remove the old generation)."
            )
    elif not manifest_path.is_file() and any(cache_dir.glob("shard_*.bf16")):
        raise ValueError("Found VAE shard files without a manifest; refusing ambiguous resume.")
    resume_incomplete = False
    if manifest_path.is_file() and not overwrite:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "complete":
            if (
                augmentation_contract is not None or image_preprocess_contract is not None
            ) and existing_manifest.get("preencode") != encoding_contract:
                raise ValueError("Complete VAE cache preencode contract mismatch.")
            VaeLatentCache(
                cache_dir,
                expected_dataset_signature=signature,
                expected_num_samples=num_samples,
                expected_vae_hash=vae_hash,
                **reader_kwargs,
            ).close()
            if rank == 0:
                logger.info("Complete VAE cache already exists at %s; skipping.", cache_dir)
            return
        if existing_manifest.get("status") != "incomplete":
            raise ValueError(f"Existing VAE manifest has invalid status at {manifest_path}.")
        required = {
            "schema_version": VAE_LATENT_CACHE_SCHEMA_VERSION,
            "dataset_signature": signature,
            "num_samples": num_samples,
            "dtype": "bfloat16",
            "shard_size": shard_size,
        }
        mismatched = {
            key: (existing_manifest.get(key), value)
            for key, value in required.items()
            if existing_manifest.get(key) != value
        }
        if existing_manifest.get("vae", {}).get("model_hash") != vae_hash:
            mismatched["vae.model_hash"] = (
                existing_manifest.get("vae", {}).get("model_hash"), vae_hash
            )
        if existing_manifest.get("preencode") != encoding_contract:
            mismatched["preencode"] = (
                existing_manifest.get("preencode"), encoding_contract
            )
        if mismatched:
            raise ValueError(f"Incomplete VAE cache contract mismatch: {mismatched}.")
        if logical_metadata is not None:
            logical = existing_manifest.get("logical_index")
            if not isinstance(logical, dict) or logical.get("metadata") != logical_metadata:
                raise ValueError("Incomplete VAE cache logical metadata mismatch.")
            identity_path = cache_dir / str(logical.get("identity_file", ""))
            payload = torch.load(identity_path, map_location="cpu", weights_only=True)
            for name, expected in logical_indices.items():
                actual = torch.as_tensor(payload[name], dtype=torch.long).reshape(-1)
                if not torch.equal(actual, expected):
                    raise ValueError(
                        f"Incomplete VAE cache logical identity mismatch for {name!r}."
                    )
        manifest = existing_manifest
        latent_shape = tuple(int(x) for x in manifest.get("latent_shape", []))
        shards = list(manifest.get("shards", []))
        num_shards = (num_samples + shard_size - 1) // shard_size
        if len(latent_shape) != 4 or min(latent_shape) <= 0 or len(shards) != num_shards:
            raise ValueError("Incomplete VAE cache has invalid latent shape or shard table.")
        resume_incomplete = True
        if rank == 0:
            completed = sum(
                (cache_dir / str(shard["file"])).is_file() for shard in shards
            )
            logger.info(
                "Resuming incomplete VAE cache at %s: existing_shards=%d/%d",
                cache_dir, completed, num_shards,
            )

    if not resume_incomplete:
        num_shards = (num_samples + shard_size - 1) // shard_size
        shards = [
            {
                "file": f"shard_{shard_index:06d}.bf16",
                "start": shard_index * shard_size,
                "end": min((shard_index + 1) * shard_size, num_samples),
            }
            for shard_index in range(num_shards)
        ]

    assigned = list(range(rank, num_shards, world_size))
    pending: list[tuple[int, dict[str, Any]]] = []
    for shard_index in assigned:
        shard = shards[shard_index]
        shard_path = cache_dir / str(shard["file"])
        if shard_path.is_file():
            if not resume_incomplete:
                raise ValueError(f"Unexpected stale VAE shard without resumable manifest: {shard_path}.")
            expected_bytes = (
                (int(shard["end"]) - int(shard["start"]))
                * int(torch.tensor(latent_shape).prod().item())
                * 2
            )
            actual_bytes = shard_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"Existing VAE shard has wrong byte size: {shard_path} "
                    f"actual={actual_bytes} expected={expected_bytes}."
                )
            continue
        pending.append((shard_index, shard))

    sampler = _ShardBatchSampler([item[1] for item in pending], batch_size)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank) * 1_000_003)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(prefetch_factor),
        )
        if _to_bool(os.environ.get("WING_DATALOADER_MALLOC_TRIM", "0")):
            loader_kwargs["collate_fn"] = _collate_and_malloc_trim
    loader = DataLoader(**loader_kwargs)
    loader_iter = iter(loader)
    prefetched: tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None = None
    prefetched_load_seconds = 0.0
    prefetched_preprocess_seconds = 0.0
    prefetched_encode_seconds = 0.0

    if not resume_incomplete:
        # Shape discovery must use the same long-lived DataLoader used below.
        # Forking workers after a parent-process PyAV decode can segfault.
        if rank == 0:
            first_shard = pending[0][1]
            prefetch_load_start = time.perf_counter()
            batch = next(loader_iter)
            prefetched_load_seconds = time.perf_counter() - prefetch_load_start
            actual_batch_size = int(
                torch.as_tensor(batch["source_index"], dtype=torch.long).numel()
            )
            expected = torch.arange(
                int(first_shard["start"]),
                int(first_shard["start"]) + actual_batch_size,
                dtype=torch.long,
            )
            if logical_indices is not None:
                _validate_logical_preencode_batch(
                    batch, logical_indices=logical_indices, logical_positions=expected
                )
            else:
                _validate_source_identity(batch, expected)
            preprocess_start = time.perf_counter()
            prepared_videos = _prepare_preencode_videos(
                batch, dataset, device=device, seed=seed
            )
            prefetched_preprocess_seconds = time.perf_counter() - preprocess_start
            prefetch_encode_start = time.perf_counter()
            first_latents = _encode(vae, prepared_videos, device)
            prefetched_encode_seconds = time.perf_counter() - prefetch_encode_start
            prefetched = (batch, expected, first_latents)
            latent_shape_tensor = torch.tensor(
                list(first_latents.shape[1:]), dtype=torch.long, device=device
            )
        else:
            latent_shape_tensor = torch.empty(4, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(latent_shape_tensor, src=0)
        latent_shape = tuple(int(x) for x in latent_shape_tensor.cpu().tolist())
        manifest = {
            "schema_version": VAE_LATENT_CACHE_SCHEMA_VERSION,
            "status": "incomplete",
            "dataset_signature": signature,
            "num_samples": num_samples,
            "latent_shape": list(latent_shape),
            "dtype": "bfloat16",
            "shard_size": shard_size,
            "shards": shards,
            "vae": {"path": str(vae_path), "model_hash": str(vae_hash)},
            "preencode": encoding_contract,
        }
        if num_samples != full_num_samples:
            manifest["probe_prefix_records"] = num_samples
            manifest["full_num_samples"] = full_num_samples
        if rank == 0:
            if logical_metadata is not None:
                identity_file = "logical_identity.pt"
                _atomic_torch_save(cache_dir / identity_file, logical_indices)
                manifest["logical_index"] = {
                    "metadata": logical_metadata,
                    "identity_file": identity_file,
                }
            _atomic_json(manifest_path, manifest)
        if dist.is_initialized():
            dist.barrier()

    progress = tqdm(pending, desc=f"VAE shards rank {rank}/{world_size}", disable=rank != 0)
    load_seconds = prefetched_load_seconds
    preprocess_seconds = prefetched_preprocess_seconds
    encode_seconds = prefetched_encode_seconds
    write_seconds = 0.0
    total_records = total_batches = 0
    run_start = time.perf_counter() - load_seconds - preprocess_seconds - encode_seconds
    last_log_time = run_start
    last_log_records = 0
    for shard_index, shard in progress:
        encoded: list[torch.Tensor] = []
        consumed = 0
        start, end = int(shard["start"]), int(shard["end"])
        while consumed < end - start:
            expected_count = min(batch_size, end - start - consumed)
            expected = torch.arange(start + consumed, start + consumed + expected_count)
            if prefetched is not None:
                batch, prefetched_expected, latents = prefetched
                prefetched = None
                if not torch.equal(expected, prefetched_expected):
                    raise RuntimeError("Prefetched VAE batch does not match shard position.")
            else:
                load_start = time.perf_counter()
                batch = next(loader_iter)
                load_seconds += time.perf_counter() - load_start
                if logical_indices is not None:
                    _validate_logical_preencode_batch(
                        batch, logical_indices=logical_indices, logical_positions=expected
                    )
                else:
                    _validate_source_identity(batch, expected)
                preprocess_start = time.perf_counter()
                prepared_videos = _prepare_preencode_videos(
                    batch, dataset, device=device, seed=seed
                )
                preprocess_seconds += time.perf_counter() - preprocess_start
                encode_start = time.perf_counter()
                latents = _encode(vae, prepared_videos, device)
                encode_seconds += time.perf_counter() - encode_start
            if latents.shape[0] != expected_count:
                raise RuntimeError(
                    f"VAE short batch: got {latents.shape[0]}, expected {expected_count}."
                )
            if tuple(latents.shape[1:]) != latent_shape:
                raise ValueError(
                    f"VAE latent shape changed: {tuple(latents.shape[1:])} vs {latent_shape}."
                )
            encoded.append(latents)
            consumed += expected_count
            total_records += expected_count
            total_batches += 1
            if log_every_batches > 0 and total_batches % log_every_batches == 0:
                now = time.perf_counter()
                window_seconds = max(now - last_log_time, 1e-9)
                total_seconds = max(now - run_start, 1e-9)
                window_rate = (total_records - last_log_records) / window_seconds
                total_rate = total_records / total_seconds
                peak_gib = (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if str(device).startswith("cuda") else 0.0
                )
                logger.info(
                    "VAE perf rank=%d batches=%d records=%d clips_s_window=%.2f "
                    "clips_s_total=%.2f load_s=%.2f cuda_preprocess_enqueue_s=%.2f "
                    "encode_h2d_compute_d2h_s=%.2f "
                    "write_s=%.2f peak_cuda_gib=%.2f",
                    rank, total_batches, total_records, window_rate, total_rate,
                    load_seconds, preprocess_seconds, encode_seconds, write_seconds, peak_gib,
                )
                last_log_time = now
                last_log_records = total_records
        shard_latents = torch.cat(encoded, dim=0)
        write_start = time.perf_counter()
        write_bfloat16_shard(
            cache_dir / str(shard["file"]),
            shard_latents,
            temporary_tag=f"rank{rank}",
        )
        write_seconds += time.perf_counter() - write_start

    elapsed = max(time.perf_counter() - run_start, 1e-9)
    worker_processes = list(getattr(loader_iter, "_workers", []) or [])
    worker_rss_gib = sum(
        _process_rss_gib(worker.pid) for worker in worker_processes if worker.is_alive()
    )
    main_rss_gib = _process_rss_gib(os.getpid())
    logger.info(
        "VAE perf final rank=%d records=%d batches=%d clips_s=%.2f load_s=%.2f "
        "cuda_preprocess_enqueue_s=%.2f encode_h2d_compute_d2h_s=%.2f "
        "write_s=%.2f main_rss_gib=%.2f "
        "worker_rss_total_gib=%.2f",
        rank, total_records, total_batches, total_records / elapsed,
        load_seconds, preprocess_seconds, encode_seconds, write_seconds,
        main_rss_gib, worker_rss_gib,
    )
    # Persistent workers otherwise survive until interpreter finalization; on
    # NFS their multiprocessing temp handles can turn into noisy busy .nfs files.
    shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        # The same reader used by training performs complete range and byte-size validation.
        VaeLatentCache(
            cache_dir,
            expected_dataset_signature=signature,
            expected_num_samples=num_samples,
            require_complete=False,
            **reader_kwargs,
        ).close()
        manifest["status"] = "complete"
        _atomic_json(manifest_path, manifest)
        VaeLatentCache(
            cache_dir,
            expected_dataset_signature=signature,
            expected_num_samples=num_samples,
            **reader_kwargs,
        ).close()
        logger.info(
            "Completed VAE cache: path=%s samples=%d latent_shape=%s shards=%d",
            cache_dir, num_samples, latent_shape, num_shards,
        )
    if dist.is_initialized():
        dist.barrier()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    misc.register_work_dir(str(cfg.get("output_dir", "./runs/precompute_vae_latents")))
    # OmegaConf interpolations may expose the same RobotVideoDataset node under
    # a reusable config alias and under data.train. Process that object once.
    nodes = []
    seen_node_ids = set()
    for node_path, node in _iter_robot_dataset_nodes(cfg.data):
        node_id = id(node)
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)
        nodes.append((node_path, node))
    if not nodes:
        raise ValueError("No RobotVideoDataset nodes found under cfg.data.")
    fallback_cache_dir = cfg.get("preencode_cache_dir")
    if fallback_cache_dir is not None and len(nodes) != 1:
        raise ValueError("+preencode_cache_dir can only be used when exactly one dataset node exists.")

    model_cfg = cfg.model
    model_id = str(model_cfg.model_id)
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=str(model_cfg.tokenizer_model_id),
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    vae_hash = next(
        item["model_hash"] for item in WAN22_MODEL_REGISTRY
        if item["model_name"] == "wan_video_vae"
    )

    overwrite = _to_bool(cfg.get("preencode_overwrite", False))
    batch_size = int(cfg.get("preencode_batch_size", 4))
    shard_size = int(cfg.get("preencode_shard_size", 256))
    num_workers = int(cfg.get("preencode_num_workers", cfg.get("num_workers", 4)))
    prefetch_factor = int(cfg.get("preencode_prefetch_factor", 2))
    seed = int(cfg.get("preencode_seed", 20260817))
    log_every_batches = int(cfg.get("preencode_log_every_batches", 20))
    max_samples_value = cfg.get("preencode_max_samples")
    max_samples = None if max_samples_value is None else int(max_samples_value)
    image_preprocess = str(cfg.get("preencode_image_preprocess", "cpu")).strip().lower()
    if image_preprocess not in {"cpu", ROBOTWIN_CUDA_PREPROCESS}:
        raise ValueError(f"Unsupported preencode_image_preprocess={image_preprocess!r}.")
    color_jitter = {
        name: float(cfg.get(f"preencode_color_jitter_{name}", 0.0))
        for name in ("brightness", "contrast", "saturation", "hue")
    }
    if any(value < 0 for value in color_jitter.values()):
        raise ValueError("Pre-encode ColorJitter strengths must be non-negative.")
    if color_jitter["hue"] > 0.5:
        raise ValueError("preencode_color_jitter_hue must be <= 0.5.")
    if not any(color_jitter.values()):
        color_jitter = None
    if min(batch_size, shard_size, prefetch_factor) <= 0 or num_workers < 0:
        raise ValueError("Invalid pre-encode batch/shard/worker configuration.")
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)

    for node_path, node in nodes:
        dataset, cache_dir = _dataset_and_cache_dir(
            node,
            None if fallback_cache_dir is None else str(fallback_cache_dir),
            color_jitter=color_jitter,
            augmentation_seed=seed,
            image_preprocess=image_preprocess,
        )
        if rank == 0:
            logger.info("Pre-encoding %s into %s", node_path, cache_dir)
        _precompute_one(
            dataset,
            cache_dir,
            vae=vae,
            vae_path=str(vae_config.path),
            vae_hash=vae_hash,
            device=device,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            shard_size=shard_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            seed=seed,
            log_every_batches=log_every_batches,
            overwrite=overwrite,
            max_samples=max_samples,
        )
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
