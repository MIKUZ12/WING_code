"""Strict, video-only reader used by offline Wan VAE pre-encoding.

The training dataset asks LeRobot for the full action/state horizon and 33 RGB
timestamps before selecting the nine frames consumed by Wan.  This view keeps
the exact training-time visual transforms and cache signature, but constructs a
second LeRobot reader whose only temporal queries are those nine RGB endpoints.
It deliberately has no retry/random-sample fallback: record ``i`` either
decodes record ``i`` or raises.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
import warnings

import torch
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as transforms_F
from omegaconf import DictConfig

from wing.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)

from .lerobot.lerobot_dataset import MultiLeRobotDataset


class RobotVaePreencodeVideoDataset(torch.utils.data.Dataset):
    """Sparse RGB-only view of an initialized ``RobotVideoDataset``."""

    def __init__(
        self,
        robot_dataset: Any,
        *,
        color_jitter: dict[str, float] | None = None,
        augmentation_seed: int = 0,
        defer_image_processing: bool = False,
    ) -> None:
        base = robot_dataset.lerobot_dataset
        processor = base.processor
        if processor is None:
            raise ValueError("VAE pre-encoding requires an initialized image processor.")
        if bool(getattr(processor, "image_augmentation", False)):
            raise ValueError("VAE pre-encoding refuses stochastic image augmentation.")

        self.vae_latent_cache_signature = robot_dataset.vae_latent_cache_signature
        self.video_size = tuple(int(value) for value in robot_dataset.video_size)
        self.concat_multi_camera = robot_dataset.concat_multi_camera
        self.video_sample_indices = tuple(int(value) for value in robot_dataset.video_sample_indices)
        self.image_meta = [dict(item) for item in base.image_meta]
        self.train_transforms = processor.train_transforms
        self.global_sample_stride = int(base.global_sample_stride)
        self.color_jitter = None
        self.vae_preencode_augmentation = None
        self.augmentation_seed = int(augmentation_seed)
        # The default path remains the established CPU transform recipe.  The
        # deferred form is intentionally just a transport contract for the
        # offline pre-encoder: [camera,time,channel,height,width] uint8 pixels,
        # with all source identity/padding fields left unchanged.
        self.defer_image_processing = bool(defer_image_processing)
        # Deferred CUDA preprocessing is only worthwhile with the batched raw
        # decoder.  Treat loss of that optimization as a contract failure
        # instead of silently returning to the multi-hour scalar path.
        self.batch_decode_required = self.defer_image_processing
        self._batch_decode_disabled = False
        if color_jitter is not None:
            values = {
                name: float(color_jitter.get(name, 0.0))
                for name in ("brightness", "contrast", "saturation", "hue")
            }
            self.color_jitter = ColorJitter(**values)
            self.vae_preencode_augmentation = {
                "name": "color_jitter",
                **values,
                "application": "post_compose_pre_normalize",
                "shared_across_frames_and_cameras": True,
                "seed_strategy": "preencode_seed_plus_source_index",
            }
        self.preencode_seed: int | None = None

        source_datasets = base.multi_dataset._datasets
        if not source_datasets:
            raise ValueError("Robot VAE reader received no LeRobot datasets.")
        fps = int(source_datasets[0].fps)
        self.fps = fps
        image_delta_timestamps: dict[str, list[float]] = {}
        for meta in self.image_meta:
            lerobot_key = str(meta["lerobot_key"])
            image_delta_timestamps[lerobot_key] = [
                (offset * self.global_sample_stride) / fps
                for offset in self.video_sample_indices
            ]

        episodes: dict[str, list[int]] | None = {}
        for source in source_datasets:
            if source.episodes is None:
                episodes = None
                break
            episodes[str(source.repo_id)] = [int(value) for value in source.episodes]

        self.reader = MultiLeRobotDataset(
            dataset_dirs=list(base.dataset_dirs),
            episodes=episodes,
            delta_timestamps=image_delta_timestamps,
            validate_episode_files=False,
        )
        self.reader.set_during_training(True)
        backends = {str(source.video_backend) for source in self.reader._datasets}
        if len(backends) != 1:
            raise ValueError(f"Sparse VAE reader has mixed video backends: {sorted(backends)}.")
        self.video_backend = next(iter(backends))
        if len(self.reader) != len(robot_dataset):
            raise ValueError(
                "Sparse VAE reader changed dataset length: "
                f"{len(self.reader)} vs {len(robot_dataset)}."
            )

        self._dataset_ends: list[int] = []
        self._dataset_starts: list[int] = []
        running = 0
        for source in self.reader._datasets:
            self._dataset_starts.append(running)
            running += int(source.num_frames)
            self._dataset_ends.append(running)

        self._resize = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self._crop = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self._normalize = Normalize(args={"mean": 0.5, "std": 0.5})

    def __len__(self) -> int:
        return len(self.reader)

    def set_preencode_seed(self, seed: int) -> None:
        self.preencode_seed = int(seed)

    def _expected_dataset_index(self, index: int) -> int:
        return bisect_right(self._dataset_ends, int(index))

    def _expected_identity(self, index: int) -> tuple[int, int, int]:
        dataset_index = self._expected_dataset_index(index)
        source = self.reader._datasets[dataset_index]
        local_index = int(index) - self._dataset_starts[dataset_index]
        episode_to = torch.as_tensor(source.episode_data_index["to"], dtype=torch.long)
        episode_from = torch.as_tensor(source.episode_data_index["from"], dtype=torch.long)
        episode_position = int(
            torch.searchsorted(episode_to, torch.tensor(local_index), right=True).item()
        )
        if episode_position >= len(episode_to):
            raise RuntimeError(f"Sparse VAE index {index} is outside child episode ranges.")
        episodes = source.episodes
        episode_index = episode_position if episodes is None else int(episodes[episode_position])
        frame_index = local_index - int(episode_from[episode_position].item())
        return dataset_index, episode_index, frame_index

    def _prepare_raw_camera(self, meta: dict[str, Any], raw: torch.Tensor) -> torch.Tensor:
        video = raw
        if video.ndim == 3:
            video = video.unsqueeze(0)
        expected_frames = len(self.video_sample_indices)
        if video.ndim != 4 or video.shape[0] != expected_frames or video.shape[1] != 3:
            raise ValueError(
                f"Sparse VAE camera {meta['key']!r} produced {tuple(video.shape)}, "
                f"expected [{expected_frames},3,H,W]."
            )
        if video.dtype != torch.uint8:
            video = (video * 255).to(torch.uint8)
        return video

    def _transform_camera(
        self, meta: dict[str, Any], raw: torch.Tensor, *, source_index: int
    ) -> torch.Tensor:
        video = self._prepare_raw_camera(meta, raw)
        transforms = self.train_transforms
        if isinstance(transforms, (dict, DictConfig)):
            transforms = transforms[str(meta["key"])]
        if self.preencode_seed is None:
            for transform in transforms:
                video = transform(video)
        else:
            # fork_rng restores the worker RNG afterwards. Reusing this seed for
            # each camera also gives geometrically aligned multi-camera clips.
            sample_seed = (
                int(self.preencode_seed) + int(source_index) * 1_000_003
            ) % (2**63 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(sample_seed)
                for transform in transforms:
                    video = transform(video)
        return video

    def _compose_cameras(
        self, cameras: Sequence[torch.Tensor], *, source_index: int
    ) -> torch.Tensor:
        num_cameras = len(cameras)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError("RoboTwin VAE composition requires exactly three cameras.")
            top = transforms_F.resize(
                cameras[0], [256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            left = transforms_F.resize(
                cameras[1], [128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            right = transforms_F.resize(
                cameras[2], [128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
        elif num_cameras == 1:
            video = cameras[0]
        elif self.concat_multi_camera == "horizontal":
            video = torch.cat(list(cameras), dim=-1)
        elif self.concat_multi_camera == "vertical":
            video = torch.cat(list(cameras), dim=-2)
        else:
            raise ValueError(f"Invalid multi-camera VAE composition: {self.concat_multi_camera!r}.")
        video = self._crop(self._resize(video))
        if self.color_jitter is not None:
            # ColorJitter samples one transform order and one set of factors for
            # the entire [T,C,H,W] tensor, keeping all frames/cameras coherent.
            # Isolating and deriving the RNG from source_index makes a record
            # invariant to worker/rank ownership and interrupted-cache resume.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.augmentation_seed + int(source_index))
                video = self.color_jitter(video)
        video = self._normalize(video)
        video = video.permute(1, 0, 2, 3).contiguous()
        expected = (3, len(self.video_sample_indices), *self.video_size)
        if tuple(video.shape) != expected:
            raise ValueError(f"Sparse VAE video shape {tuple(video.shape)} != {expected}.")
        if not torch.isfinite(video).all():
            raise ValueError("Sparse VAE video contains non-finite values.")
        return video

    @staticmethod
    def _decode_torchcodec_requests(
        video_path: Path,
        request_timestamps: Sequence[Sequence[float]],
        tolerance_s: float,
    ) -> list[torch.Tensor]:
        """Decode the union of nearby requests while preserving per-request selection.

        The normal reader constructs one ``VideoDecoder`` for each sample and
        camera.  A cache-shard batch consists of adjacent samples, whose nine
        requested frames overlap heavily.  Here one decoder reads each unique
        frame index once.  Crucially, closest-PTS selection is still performed
        independently against each sample's original candidate list; frames
        introduced by a neighbouring sample can therefore never alter it.
        """
        from torchcodec.decoders import VideoDecoder

        # Keep these operations identical to decode_video_frames_torchcodec:
        # Python round on metadata.average_fps, float32 cdist, and strict '<'
        # tolerance.  Only decoder construction and duplicate frame reads are
        # coalesced.
        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        average_fps = decoder.metadata.average_fps
        per_request_indices = [
            [round(float(timestamp) * average_fps) for timestamp in timestamps]
            for timestamps in request_timestamps
        ]
        unique_indices = sorted({
            index for indices in per_request_indices for index in indices
        })
        frames_batch = decoder.get_frames_at(indices=unique_indices)
        if len(frames_batch.data) != len(unique_indices):
            raise RuntimeError(
                f"torchcodec returned {len(frames_batch.data)} frames for "
                f"{len(unique_indices)} indices from {video_path}."
            )
        index_to_position = {index: position for position, index in enumerate(unique_indices)}

        # Importing the local policy avoids changing the historical float32
        # behaviour when the uint8 VAE fast path is explicitly disabled.
        from .lerobot.datasets.video_utils import _return_uint8_video_frames

        return_uint8 = _return_uint8_video_frames()
        decoded: list[torch.Tensor] = []
        for timestamps, frame_indices in zip(
            request_timestamps, per_request_indices, strict=True
        ):
            positions = [index_to_position[index] for index in frame_indices]
            loaded_frames = [frames_batch.data[position] for position in positions]
            loaded_ts = torch.tensor(
                [frames_batch.pts_seconds[position].item() for position in positions],
                dtype=torch.float32,
            )
            query_ts = torch.tensor(timestamps, dtype=torch.float32)
            distances = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
            minimum, closest = distances.min(1)
            is_within_tolerance = minimum < float(tolerance_s)
            if not bool(is_within_tolerance.all()):
                raise AssertionError(
                    "One or several batched VAE timestamps violate the decode "
                    f"tolerance ({minimum[~is_within_tolerance]} > tolerance_s={tolerance_s}). "
                    f"queried={query_ts} loaded={loaded_ts} video={video_path}"
                )
            closest_frames = torch.stack(
                [loaded_frames[int(position)] for position in closest]
            )
            if return_uint8:
                if closest_frames.dtype != torch.uint8:
                    raise TypeError(
                        "uint8 video output requested, but batched torchcodec "
                        f"returned {closest_frames.dtype}."
                    )
            else:
                closest_frames = closest_frames.to(torch.float32) / 255
            decoded.append(closest_frames)
        return decoded

    def _make_output(
        self,
        *,
        requested: int,
        actual_identity: tuple[int, int, int],
        source_episode_index: Any,
        source_frame_index: Any,
        image_is_pad: torch.Tensor,
        cameras: Sequence[torch.Tensor],
    ) -> dict[str, Any]:
        output = {
            "source_index": torch.tensor(requested, dtype=torch.long),
            "source_dataset_index": torch.tensor(actual_identity[0], dtype=torch.long),
            "source_episode_index": torch.as_tensor(source_episode_index, dtype=torch.long),
            "source_frame_index": torch.as_tensor(source_frame_index, dtype=torch.long),
            "image_is_pad": torch.as_tensor(image_is_pad, dtype=torch.bool).reshape(-1),
        }
        expected_pad_shape = (len(self.video_sample_indices),)
        if tuple(output["image_is_pad"].shape) != expected_pad_shape:
            raise ValueError(
                f"Invalid sparse VAE padding shape: {tuple(output['image_is_pad'].shape)}."
            )
        if self.defer_image_processing:
            raw_cameras = [
                self._prepare_raw_camera(meta, raw)
                for meta, raw in zip(self.image_meta, cameras, strict=True)
            ]
            shapes = {tuple(camera.shape) for camera in raw_cameras}
            if len(shapes) != 1:
                raise ValueError(
                    "Deferred VAE image processing requires uniform camera shapes; "
                    f"got {sorted(shapes)}."
                )
            output["raw_cameras"] = torch.stack(raw_cameras, dim=0).contiguous()
        else:
            transformed = [
                self._transform_camera(meta, raw, source_index=requested)
                for meta, raw in zip(self.image_meta, cameras, strict=True)
            ]
            output["video"] = self._compose_cameras(
                transformed, source_index=requested
            )
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested = int(index)
        if requested < 0 or requested >= len(self):
            raise IndexError(f"VAE source index {requested} is out of range.")
        # MultiLeRobotDataset and LeRobotDataset have no retry path. Any decode
        # exception is intentionally propagated so a wrong sample can never be
        # cached under the requested record number.
        raw = self.reader[requested]
        if "frame_index" in raw:
            actual_frame_index = int(torch.as_tensor(raw["frame_index"]).item())
        else:
            # LeRobot v2.0 RoboCasa parquet files omit frame_index. Their
            # timestamp is frame_index / fps, so it is an independent identity
            # check against the episode bounds used by _expected_identity().
            timestamp = torch.as_tensor(raw["timestamp"]).reshape(-1)
            if timestamp.numel() != 1:
                raise RuntimeError(
                    f"Sparse VAE source timestamp is not scalar: {tuple(timestamp.shape)}."
                )
            actual_frame_index = int(round(float(timestamp.item()) * self.fps))
        actual_identity = (
            int(torch.as_tensor(raw["dataset_index"]).item()),
            int(torch.as_tensor(raw["episode_index"]).item()),
            actual_frame_index,
        )
        expected_identity = self._expected_identity(requested)
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"Sparse VAE source identity mismatch for {requested}: "
                f"actual={actual_identity} expected={expected_identity}."
            )
        cameras = [raw[str(meta["lerobot_key"])] for meta in self.image_meta]
        first_pad_key = f"{self.image_meta[0]['lerobot_key']}_is_pad"
        image_is_pad = torch.as_tensor(raw[first_pad_key], dtype=torch.bool).reshape(-1)
        if image_is_pad.shape != (len(self.video_sample_indices),):
            raise ValueError(f"Invalid sparse VAE padding shape: {tuple(image_is_pad.shape)}.")
        return self._make_output(
            requested=requested,
            actual_identity=actual_identity,
            source_episode_index=raw["episode_index"],
            source_frame_index=actual_frame_index,
            image_is_pad=image_is_pad,
            cameras=cameras,
        )

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Load a DataLoader batch with one union decode per episode/camera.

        PyTorch's map-style DataLoader automatically calls ``__getitems__``
        when a batch sampler supplies a list of indices.  Non-torchcodec
        backends and any batched-decode failure take the canonical scalar path,
        preserving the strict no-random-retry behaviour of ``__getitem__``.
        """
        requested_indices = [int(index) for index in indices]
        if self.video_backend != "torchcodec":
            if self.batch_decode_required:
                raise RuntimeError(
                    "Deferred VAE image processing requires batched torchcodec decode; "
                    f"got video_backend={self.video_backend!r}."
                )
            return [self[index] for index in requested_indices]
        if self._batch_decode_disabled:
            if self.batch_decode_required:
                raise RuntimeError(
                    "Required batched VAE decode was previously disabled in this worker."
                )
            return [self[index] for index in requested_indices]
        if not requested_indices:
            return []

        records: list[dict[str, Any]] = []
        groups: dict[tuple[int, int, str], dict[str, Any]] = {}
        first_pad_key = f"{self.image_meta[0]['lerobot_key']}_is_pad"
        try:
            for output_position, requested in enumerate(requested_indices):
                if requested < 0 or requested >= len(self):
                    raise IndexError(f"VAE source index {requested} is out of range.")
                dataset_index = self._expected_dataset_index(requested)
                source = self.reader._datasets[dataset_index]
                local_index = requested - self._dataset_starts[dataset_index]
                episode_to = torch.as_tensor(source.episode_data_index["to"], dtype=torch.long)
                episode_position = int(
                    torch.searchsorted(
                        episode_to, torch.tensor(local_index), right=True
                    ).item()
                )
                if episode_position >= len(episode_to):
                    raise RuntimeError(
                        f"Sparse VAE index {requested} is outside child episode ranges."
                    )

                item = source.hf_dataset[local_index]
                actual_identity = (
                    dataset_index,
                    int(torch.as_tensor(item["episode_index"]).item()),
                    int(torch.as_tensor(item["frame_index"]).item()),
                )
                expected_identity = self._expected_identity(requested)
                if actual_identity != expected_identity:
                    raise RuntimeError(
                        f"Sparse VAE source identity mismatch for {requested}: "
                        f"actual={actual_identity} expected={expected_identity}."
                    )

                query_indices, padding = source._get_query_indices(
                    local_index, episode_position
                )
                # This reader is constructed with image-only delta timestamps.
                # Fail back to the canonical reader if that invariant changes.
                if any(key not in source.meta.video_keys for key in query_indices):
                    raise RuntimeError(
                        "Sparse VAE batched reader received non-video delta queries."
                    )
                query_timestamps = source._get_query_timestamps(
                    float(torch.as_tensor(item["timestamp"]).item()), query_indices
                )
                record = {
                    "requested": requested,
                    "actual_identity": actual_identity,
                    "source_episode_index": item["episode_index"],
                    "source_frame_index": item["frame_index"],
                    "image_is_pad": padding[first_pad_key],
                    "cameras": [None] * len(self.image_meta),
                }
                records.append(record)

                for camera_position, meta in enumerate(self.image_meta):
                    lerobot_key = str(meta["lerobot_key"])
                    group_key = (dataset_index, actual_identity[1], lerobot_key)
                    group = groups.get(group_key)
                    if group is None:
                        video_path = source.root / source.meta.get_video_file_path(
                            actual_identity[1], lerobot_key
                        )
                        group = {
                            "video_path": video_path,
                            "tolerance_s": float(source.tolerance_s),
                            "requests": [],
                        }
                        groups[group_key] = group
                    group["requests"].append(
                        (
                            output_position,
                            camera_position,
                            list(query_timestamps[lerobot_key]),
                        )
                    )

            def decode_group(group: dict[str, Any]):
                timestamps = [request[2] for request in group["requests"]]
                decoded = self._decode_torchcodec_requests(
                    group["video_path"], timestamps, group["tolerance_s"]
                )
                return group["requests"], decoded

            # Three independent RoboTwin camera streams leave CPU headroom in
            # the current pre-encode job.  Bound concurrency by camera count so
            # an episode-boundary batch cannot create six decoder threads.
            max_workers = min(len(groups), max(1, len(self.image_meta)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(decode_group, group) for group in groups.values()]
                for future in futures:
                    requests, decoded = future.result()
                    for request, camera in zip(requests, decoded, strict=True):
                        output_position, camera_position, _ = request
                        records[output_position]["cameras"][camera_position] = camera

            outputs: list[dict[str, Any]] = []
            for record in records:
                if any(camera is None for camera in record["cameras"]):
                    raise RuntimeError("Batched VAE decode left a camera request unresolved.")
                outputs.append(
                    self._make_output(
                        requested=record["requested"],
                        actual_identity=record["actual_identity"],
                        source_episode_index=record["source_episode_index"],
                        source_frame_index=record["source_frame_index"],
                        image_is_pad=record["image_is_pad"],
                        cameras=record["cameras"],
                    )
                )
            return outputs
        except Exception as error:
            if self.batch_decode_required:
                raise RuntimeError(
                    "Required batched RoboTwin VAE decode failed; refusing scalar fallback."
                ) from error
            # The scalar path already implements torchcodec->PyAV fallback and
            # carries the established error messages.  Disable this optional
            # optimization for the lifetime of the worker after its first
            # failure, then retry exactly the requested records through it.
            self._batch_decode_disabled = True
            warnings.warn(
                "Disabling batched RoboTwin VAE decode in this worker after "
                f"{type(error).__name__}: {error}"
            )
            return [self[index] for index in requested_indices]
