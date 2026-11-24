"""Video-only Ego Stage-I dataset for Wan VAE pre-encoding.

Unlike ``RobotVideoDataset``, this reader never materializes the 21.6M-row
parquet/Arrow dataset.  The authoritative logical index already identifies the
eligible train anchors, so VAE precompute only needs to decode their nine RGB
endpoints from the source videos.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .ego_stage1_index import EgoStage1LogicalIndex
from .spectral_targets import SpectralEpisodeCache, SpectralSampleKey
from .lerobot.lerobot.datasets.video_utils import decode_video_frames
from .lerobot.vae_latent_cache import VaeLatentCache


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_ego_base_video_signature(
    *,
    logical_index: EgoStage1LogicalIndex,
    num_frames: int,
    global_sample_stride: int,
    action_video_freq_ratio: int,
    video_size: Sequence[int],
    concat_multi_camera: str | None,
    image_shape_meta: Sequence[dict[str, Any]],
    train_transforms: Any,
) -> str:
    """Reproduce ``RobotVideoDataset.vae_latent_cache_signature`` without Arrow."""
    datasets = []
    manifest_datasets = logical_index.manifest.get("datasets", [])
    for item in manifest_datasets:
        datasets.append(
            {
                "root": os.path.realpath(item["root"]),
                "episodes": None,
                "num_frames": int(item["total_frames"]),
            }
        )
    video_sample_indices = list(range(0, int(num_frames), int(action_video_freq_ratio)))
    # BaseLerobotDataset adds this derived key to shape_meta in-place before
    # RobotVideoDataset computes its cache signature.  Mirror that canonical
    # representation here so the lightweight precompute reader and the real
    # training reader describe the same RGB stream.
    canonical_image_shape_meta = []
    for item in image_shape_meta:
        canonical_item = dict(item)
        key = str(canonical_item["key"])
        canonical_item.setdefault(
            "lerobot_key",
            f"observation.images.{key}" if key != "default" else "observation.images",
        )
        canonical_image_shape_meta.append(canonical_item)
    payload = {
        "schema_version": 1,
        "datasets": datasets,
        "length": sum(int(item["total_frames"]) for item in manifest_datasets),
        "num_frames": int(num_frames),
        "global_sample_stride": int(global_sample_stride),
        "action_video_freq_ratio": int(action_video_freq_ratio),
        "video_sample_indices": video_sample_indices,
        "video_size": [int(value) for value in video_size],
        "concat_multi_camera": concat_multi_camera,
        "image_shape_meta": canonical_image_shape_meta,
        "train_transforms": repr(train_transforms),
    }
    return _canonical_sha256(payload)


class EgoStage1LogicalVideoDataset(Dataset):
    """Decode exactly the Stage-I logical train endpoints and nothing else."""

    def __init__(
        self,
        *,
        logical_index_manifest: str | Path,
        logical_split: str,
        dataset_roots: Sequence[str],
        image_shape_meta: Sequence[dict[str, Any]],
        train_transforms: Any,
        num_frames: int,
        global_sample_stride: int,
        action_video_freq_ratio: int,
        video_size: Sequence[int],
        concat_multi_camera: str | None,
        expected_fps: float = 30.0,
        anchor_stride: int = 24,
        window_span: int = 48,
    ) -> None:
        self.logical_index = EgoStage1LogicalIndex(
            logical_index_manifest,
            split=logical_split,
            expected_dataset_roots=dataset_roots,
            expected_anchor_stride=anchor_stride,
            expected_window_span=window_span,
        )
        self.num_frames = int(num_frames)
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = tuple(int(value) for value in video_size)
        self.concat_multi_camera = concat_multi_camera
        self.image_shape_meta = [dict(item) for item in image_shape_meta]
        self.train_transforms = train_transforms
        self.expected_fps = float(expected_fps)
        if self.num_frames != window_span + 1 or self.global_sample_stride != 1:
            raise ValueError("Ego Stage-I direct video reader requires 49 raw slots and stride=1.")
        if tuple(range(0, self.num_frames, self.action_video_freq_ratio)) != self.logical_index.frame_offsets:
            raise ValueError("Direct video endpoint offsets differ from the logical index.")
        if len(self.image_shape_meta) != 1 or self.concat_multi_camera is not None:
            raise ValueError("Ego Stage-I direct VAE reader requires exactly one non-concatenated camera.")
        self.camera_key = str(self.image_shape_meta[0]["key"])
        if self.camera_key != "cam_high":
            raise ValueError(f"Unexpected Ego Stage-I camera key: {self.camera_key!r}.")

        for item in self.logical_index.manifest.get("datasets", []):
            info_path = Path(item["root"]) / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if float(info["fps"]) != self.expected_fps:
                raise ValueError(f"Ego source fps mismatch at {info_path}.")
            if int(info["total_frames"]) != int(item["total_frames"]):
                raise ValueError(f"Ego source frame count mismatch at {info_path}.")
            feature = info.get("features", {}).get(f"observation.images.{self.camera_key}")
            if not isinstance(feature, dict) or feature.get("dtype") != "video":
                raise ValueError(f"Missing Ego video feature in {info_path}.")

        self._resize = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self._crop = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self._normalize = Normalize(args={"mean": 0.5, "std": 0.5})
        base_signature = build_ego_base_video_signature(
            logical_index=self.logical_index,
            num_frames=self.num_frames,
            global_sample_stride=self.global_sample_stride,
            action_video_freq_ratio=self.action_video_freq_ratio,
            video_size=self.video_size,
            concat_multi_camera=self.concat_multi_camera,
            image_shape_meta=self.image_shape_meta,
            train_transforms=self.train_transforms,
        )
        self.vae_latent_cache_signature = self.logical_index.vae_dataset_signature(base_signature)

    @property
    def vae_logical_metadata(self) -> dict[str, Any]:
        return self.logical_index.vae_logical_metadata

    @property
    def vae_logical_indices(self) -> dict[str, torch.Tensor]:
        return self.logical_index.vae_identity_tensors

    def __len__(self) -> int:
        return len(self.logical_index)

    def _decode(self, source_video: str, absolute_frames: Sequence[int]) -> torch.Tensor:
        timestamps = [int(frame) / self.expected_fps for frame in absolute_frames]
        return decode_video_frames(
            source_video,
            timestamps=timestamps,
            tolerance_s=1.0e-4,
            backend="pyav",
        )

    def _sample_identity(
        self, index: int
    ) -> tuple[int, int, int, int, Any, list[int]]:
        position = int(index)
        dataset_index = int(self.logical_index.dataset_indices[position])
        episode_index = int(self.logical_index.episode_indices[position])
        anchor = int(self.logical_index.frame_indices[position])
        source_index = int(self.logical_index.source_indices[position])
        identity = self.logical_index.episode_identity(dataset_index, episode_index)
        local_frames = [anchor + value for value in self.logical_index.frame_offsets]
        if local_frames[-1] >= identity.episode_length:
            raise RuntimeError("Logical VAE anchor is not a full episode-local window.")
        absolute_frames = [identity.frame_start + value for value in local_frames]
        return position, dataset_index, episode_index, source_index, identity, absolute_frames

    def _make_sample(
        self,
        *,
        dataset_index: int,
        episode_index: int,
        source_index: int,
        anchor: int,
        video: torch.Tensor,
    ) -> dict[str, Any]:
        local_frames = self.logical_index.frame_offsets
        if video.shape[:2] != (len(local_frames), 3):
            raise ValueError(f"Unexpected decoded Ego video shape: {tuple(video.shape)}.")
        # LeRobot's decoder returns float32 in [0,1], while the existing
        # WING processor intentionally starts with a uint8 ToTensor op.
        # Match BaseLerobotDataset's established conversion exactly.
        if video.dtype != torch.uint8:
            video = (video * 255).to(torch.uint8)
        transforms = self.train_transforms
        if isinstance(transforms, dict):
            transforms = transforms[self.camera_key]
        for transform in transforms:
            video = transform(video)
        expected = (len(local_frames), 3, self.video_size[0], self.video_size[1])
        if tuple(video.shape) != expected:
            raise ValueError(f"Ego train transforms produced {tuple(video.shape)}, expected {expected}.")
        video = self._normalize(self._crop(self._resize(video))).permute(1, 0, 2, 3).contiguous()
        if tuple(video.shape) != (3, len(local_frames), *self.video_size):
            raise ValueError(f"Unexpected normalized Ego video shape: {tuple(video.shape)}.")
        if not torch.isfinite(video).all():
            raise ValueError("Decoded Ego video contains non-finite values.")
        return {
            "video": video,
            "latent_source_index": torch.tensor(source_index, dtype=torch.long),
            "latent_source_dataset_index": torch.tensor(dataset_index, dtype=torch.long),
            "latent_source_episode_index": torch.tensor(episode_index, dtype=torch.long),
            "latent_source_frame_index": torch.tensor(anchor, dtype=torch.long),
            "latent_target_mask": torch.ones(len(self.logical_index.frame_offsets) - 1, dtype=torch.bool),
            "image_is_pad": torch.zeros(len(self.logical_index.frame_offsets), dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        position, dataset_index, episode_index, source_index, identity, absolute_frames = (
            self._sample_identity(index)
        )
        video = self._decode(identity.source_video, absolute_frames)
        return self._make_sample(
            dataset_index=dataset_index,
            episode_index=episode_index,
            source_index=source_index,
            anchor=int(self.logical_index.frame_indices[position]),
            video=video,
        )

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Decode each source video once for a DataLoader batch.

        Consecutive Stage-I anchors overlap heavily (anchor stride 24 versus a
        48-frame window).  PyTorch's batched Dataset API lets us merge their
        endpoint requests without changing sample identities or ordering.
        """
        requests = [self._sample_identity(index) for index in indices]
        grouped: dict[str, list[tuple[int, tuple[Any, ...]]]] = defaultdict(list)
        for output_position, request in enumerate(requests):
            grouped[request[4].source_video].append((output_position, request))
        outputs: list[dict[str, Any] | None] = [None] * len(requests)
        for source_video, source_requests in grouped.items():
            union_frames = sorted(
                {frame for _, request in source_requests for frame in request[5]}
            )
            decoded = self._decode(source_video, union_frames)
            if decoded.shape[:2] != (len(union_frames), 3):
                raise ValueError(
                    f"Unexpected batched Ego video shape: {tuple(decoded.shape)}."
                )
            frame_to_offset = {frame: offset for offset, frame in enumerate(union_frames)}
            for output_position, request in source_requests:
                position, dataset_index, episode_index, source_index, _, absolute_frames = request
                offsets = torch.tensor(
                    [frame_to_offset[frame] for frame in absolute_frames], dtype=torch.long
                )
                outputs[output_position] = self._make_sample(
                    dataset_index=dataset_index,
                    episode_index=episode_index,
                    source_index=source_index,
                    anchor=int(self.logical_index.frame_indices[position]),
                    video=decoded.index_select(0, offsets),
                )
        if any(sample is None for sample in outputs):
            raise RuntimeError("Batched Ego VAE decode did not produce every requested sample.")
        return [sample for sample in outputs if sample is not None]


class EgoStage1CachedTrainingDataset(Dataset):
    """Read completed Stage-I VAE/latent action/text caches without materializing Parquet."""

    def __init__(
        self,
        *,
        logical_index_manifest: str | Path,
        logical_split: str,
        dataset_roots: Sequence[str],
        latent_cache_root: str | Path,
        vae_cache_dir: str | Path,
        text_embedding_cache_dir: str | Path,
        context_len: int = 128,
        horizon: int = 8,
        target_dim: int = 32,
        anchor_stride: int = 24,
        window_span: int = 48,
        max_cached_episodes: int = 32,
        max_cached_text_embeddings: int = 32,
        vae_latent_cache_max_open_shards: int = 8,
        vae_latent_cache_expected_vae_hash: str | None = "e1de6c02cdac79f8b739f4d3698cd216",
        vae_latent_cache_required: bool = True,
        allow_path_relocation: bool = False,
    ) -> None:
        if not vae_latent_cache_required:
            raise ValueError("Stage-I cache-only training requires the completed VAE cache.")
        self.dataset_roots = [Path(root).resolve() for root in dataset_roots]
        self.logical_index = EgoStage1LogicalIndex(
            logical_index_manifest,
            split=logical_split,
            expected_dataset_roots=self.dataset_roots,
            expected_anchor_stride=anchor_stride,
            expected_window_span=window_span,
            allow_path_relocation=allow_path_relocation,
        )
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = Path(text_embedding_cache_dir).resolve()
        self._text_embeddings: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.max_cached_text_embeddings = max(int(max_cached_text_embeddings), 1)

        self.episode_tasks: dict[tuple[int, int], str] = {}
        for dataset_index, root in enumerate(self.dataset_roots):
            episodes_path = root / "meta" / "episodes.jsonl"
            with episodes_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    tasks = item.get("tasks", [])
                    if len(tasks) != 1:
                        raise ValueError(
                            f"Stage-I cache-only reader requires one task per episode: {episodes_path}"
                        )
                    self.episode_tasks[(dataset_index, int(item["episode_index"]))] = str(tasks[0])

        self.latent_cache = SpectralEpisodeCache(
            latent_cache_root,
            expected_dataset_roots=self.dataset_roots,
            horizon=horizon,
            target_dim=target_dim,
            max_cached_episodes=max_cached_episodes,
            require_complete=True,
            allow_path_relocation=allow_path_relocation,
        )
        vae_manifest = json.loads(
            (Path(vae_cache_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        self.vae_cache = VaeLatentCache(
            vae_cache_dir,
            expected_dataset_signature=str(vae_manifest.get("dataset_signature", "")),
            expected_num_samples=len(self.logical_index),
            expected_vae_hash=vae_latent_cache_expected_vae_hash,
            require_complete=True,
            max_open_shards=vae_latent_cache_max_open_shards,
            expected_logical_metadata=self.logical_index.vae_logical_metadata,
            expected_logical_indices=self.logical_index.vae_identity_tensors,
            allow_path_relocation=allow_path_relocation,
        )

    def __len__(self) -> int:
        return len(self.logical_index)

    def _text_context(self, task: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        prompt = DEFAULT_PROMPT.format(task=task)
        cached = self._text_embeddings.get(prompt)
        if cached is None:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing text embedding cache: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context = torch.as_tensor(payload["context"])
            mask = torch.as_tensor(payload["mask"], dtype=torch.bool)
            if context.ndim != 2 or context.shape[0] != self.context_len:
                raise ValueError(f"Invalid cached context shape in {path}: {tuple(context.shape)}")
            if mask.shape != (self.context_len,):
                raise ValueError(f"Invalid cached context mask shape in {path}: {tuple(mask.shape)}")
            context = context.clone()
            context[~mask] = 0
            cached = (context, torch.ones_like(mask))
            self._text_embeddings[prompt] = cached
            while len(self._text_embeddings) > self.max_cached_text_embeddings:
                self._text_embeddings.popitem(last=False)
        else:
            self._text_embeddings.move_to_end(prompt)
        return cached[0], cached[1], prompt

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        dataset_index = int(self.logical_index.dataset_indices[position])
        episode_index = int(self.logical_index.episode_indices[position])
        frame_index = int(self.logical_index.frame_indices[position])
        source_index = int(self.logical_index.source_indices[position])
        identity = self.logical_index.episode_identity(dataset_index, episode_index)
        task = self.episode_tasks[(dataset_index, episode_index)]
        context, context_mask, prompt = self._text_context(task)
        key = SpectralSampleKey(
            dataset_index=dataset_index,
            dataset_root=str(self.dataset_roots[dataset_index]),
            episode_index=episode_index,
            frame_index=frame_index,
            episode_length=identity.episode_length,
        )
        latent_target, latent_target_mask = self.latent_cache.get_window(
            key, expected_source=identity
        )
        if not bool(latent_target_mask.all()):
            raise RuntimeError("Stage-I eligible anchor produced an invalid latent action target window.")
        return {
            "video_latents": self.vae_cache[position],
            "context": context,
            "context_mask": context_mask,
            "prompt": prompt,
            "image_is_pad": torch.zeros(
                len(self.logical_index.frame_offsets), dtype=torch.bool
            ),
            "proprio_available": torch.tensor(False, dtype=torch.bool),
            "latent_target": latent_target,
            "latent_target_mask": latent_target_mask,
            "latent_source_index": torch.tensor(source_index, dtype=torch.long),
            "latent_source_dataset_index": torch.tensor(dataset_index, dtype=torch.long),
            "latent_source_episode_index": torch.tensor(episode_index, dtype=torch.long),
            "latent_source_frame_index": torch.tensor(frame_index, dtype=torch.long),
        }
