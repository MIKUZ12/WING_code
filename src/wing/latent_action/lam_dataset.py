"""Dataset for WING-LAM pair caches and anchor-preserving warps."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .cache import PairSupervision, load_pair_supervision
from .camera_motion_bank import CameraMotionBank, MotionSegment
from .lam import transform_points


def _crop_four_three(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_ratio = 4.0 / 3.0
    if width / height > target_ratio:
        crop_width = int(height * target_ratio)
        left = (width - crop_width) // 2
        return frame[:, left : left + crop_width]
    if width / height < target_ratio:
        crop_height = int(width / target_ratio)
        top = (height - crop_height) // 2
        return frame[top : top + crop_height]
    return frame


def _decode_video(video_path: Path, output_hw: tuple[int, int]) -> Tensor:
    frames: list[Tensor] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            rgb = _crop_four_three(frame.to_ndarray(format="rgb24"))
            image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
            image = F.interpolate(
                image[None],
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )[0]
            frames.append(image.permute(1, 2, 0).contiguous())
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    return torch.stack(frames)


def sample_camera_augmentation(
    *,
    translation_fraction: float = 0.04,
    translation_pixels: float | None = None,
    image_hw: tuple[int, int] | None = None,
    rotation_degrees: float = 4.0,
    scale_fraction: float = 0.04,
) -> Tensor:
    """Sample an affine transform in normalized coordinates around image center."""
    if translation_pixels is not None:
        if translation_pixels < 0 or image_hw is None or min(image_hw) < 2:
            raise ValueError("pixel translation requires non-negative magnitude and valid image_hw")
        direction = torch.empty(()).uniform_(0.0, 2.0 * torch.pi)
        radius = translation_pixels * torch.sqrt(torch.rand(()))
        translation = torch.stack(
            [
                radius * torch.cos(direction) / (image_hw[1] - 1),
                radius * torch.sin(direction) / (image_hw[0] - 1),
            ]
        )
    else:
        translation = torch.empty(2).uniform_(-translation_fraction, translation_fraction)
    angle = torch.empty(()).uniform_(-rotation_degrees, rotation_degrees) * torch.pi / 180.0
    scale = 1.0 + torch.empty(()).uniform_(-scale_fraction, scale_fraction)
    cosine, sine = torch.cos(angle) * scale, torch.sin(angle) * scale
    linear = torch.stack([cosine, -sine, sine, cosine]).reshape(2, 2)
    center = torch.tensor([0.5, 0.5])
    offset = center + translation - linear @ center
    matrix = torch.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = offset
    return matrix


def camera_replacement_target_matrix(
    global_matrix: Tensor,
    replacement_matrix: Tensor,
) -> Tensor:
    """Map the original target view into an independent replacement camera view.

    With source frame fixed as the canonical anchor, the original global
    motion is removed by ``inv(global_matrix)`` and replaced by the sampled
    ``replacement_matrix``.
    """
    if global_matrix.shape != (3, 3) or replacement_matrix.shape != (3, 3):
        raise ValueError("global and replacement matrices must have shape [3,3]")
    return replacement_matrix @ torch.linalg.inv(global_matrix)


def warp_images(
    images: Tensor,
    matrices: Tensor,
    *,
    padding_mode: str = "zeros",
) -> Tensor:
    """Warp batched HWC images so content point ``p`` moves to ``matrix(p)``."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape [B,H,W,3]")
    if matrices.shape != (len(images), 3, 3):
        raise ValueError("matrices must have shape [B,3,3]")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"unsupported padding_mode {padding_mode!r}")
    height, width = images.shape[1:3]
    x = torch.linspace(0.0, 1.0, width, device=images.device, dtype=images.dtype)
    y = torch.linspace(0.0, 1.0, height, device=images.device, dtype=images.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    output_points = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
    output_points = output_points.expand(len(images), -1, -1)
    input_points = transform_points(output_points, torch.linalg.inv(matrices)).reshape(
        len(images), height, width, 2
    )
    grid = input_points.mul(2.0).sub(1.0)
    warped = F.grid_sample(
        images.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped.permute(0, 2, 3, 1).contiguous()


def warp_target_image(
    image: Tensor,
    matrix: Tensor,
    *,
    padding_mode: str = "zeros",
) -> Tensor:
    """Warp one HWC image so content point ``p`` moves to ``matrix(p)``."""
    if image.ndim != 3 or image.shape[-1] != 3 or matrix.shape != (3, 3):
        raise ValueError("image/matrix must have shapes [H,W,3] and [3,3]")
    return warp_images(image[None], matrix[None], padding_mode=padding_mode)[0]


def discover_pair_records(paths: Iterable[str | Path]) -> list[Path]:
    records: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            records.extend(sorted(path.glob("pair_*.npz")))
        elif path.is_file() and path.suffix in (".txt", ".list"):
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                listed = Path(line)
                if not listed.is_absolute():
                    listed = path.parent / listed
                if not listed.is_file():
                    raise FileNotFoundError(listed)
                records.append(listed)
        elif path.is_file():
            records.append(path)
        else:
            raise FileNotFoundError(path)
    if not records:
        raise ValueError("no WING-LAM pair cache records found")
    return records


class WINGLAMPairsDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        records: Iterable[str | Path],
        *,
        output_hw: tuple[int, int] = (240, 320),
        augment_camera: bool = True,
        max_cached_videos: int = 2,
        camera_translation_pixels: float | None = None,
        camera_rotation_degrees: float = 4.0,
        camera_scale_fraction: float = 0.04,
        camera_padding_mode: str = "zeros",
        camera_augmentation_mode: str = "addition",
        replacement_min_valid_fraction: float = 0.5,
        training_view_mode: str | None = None,
        camera_motion_bank: str | Path | None = None,
        view_assignment_seed: int = 20260715,
    ) -> None:
        self.records = discover_pair_records(records)
        self.output_hw = output_hw
        self.augment_camera = augment_camera
        if camera_translation_pixels is not None and camera_translation_pixels < 0:
            raise ValueError("camera_translation_pixels must be non-negative")
        if camera_rotation_degrees < 0 or camera_scale_fraction < 0:
            raise ValueError("camera rotation/scale magnitudes must be non-negative")
        if camera_padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(f"unsupported camera_padding_mode {camera_padding_mode!r}")
        if camera_augmentation_mode not in ("addition", "replacement"):
            raise ValueError(
                f"unsupported camera_augmentation_mode {camera_augmentation_mode!r}"
            )
        if not 0.0 <= replacement_min_valid_fraction <= 1.0:
            raise ValueError("replacement_min_valid_fraction must be in [0,1]")
        allowed_view_modes = (None, "addition_pair", "empirical_pair", "natural_empirical_triplet")
        if training_view_mode not in allowed_view_modes:
            raise ValueError(f"unsupported training_view_mode {training_view_mode!r}")
        if training_view_mode is not None and not augment_camera:
            raise ValueError("training_view_mode requires augment_camera=True")
        self.camera_translation_pixels = camera_translation_pixels
        self.camera_rotation_degrees = camera_rotation_degrees
        self.camera_scale_fraction = camera_scale_fraction
        self.camera_padding_mode = camera_padding_mode
        self.camera_augmentation_mode = camera_augmentation_mode
        self.replacement_min_valid_fraction = replacement_min_valid_fraction
        self.training_view_mode = training_view_mode
        self.view_assignment_seed = view_assignment_seed
        self.camera_motion_bank = (
            CameraMotionBank.load(camera_motion_bank) if camera_motion_bank is not None else None
        )
        if training_view_mode in ("empirical_pair", "natural_empirical_triplet"):
            if self.camera_motion_bank is None:
                raise ValueError(f"{training_view_mode} requires camera_motion_bank")
            self._empirical_assignments, self.assignment_summary = self._build_assignments()
        else:
            if self.camera_motion_bank is not None:
                raise ValueError("camera_motion_bank is only valid for empirical training views")
            self._empirical_assignments = None
            self.assignment_summary = {"full_bucket": 0, "displacement_bucket": 0, "gap_only": 0}
        if max_cached_videos <= 0:
            raise ValueError("max_cached_videos must be positive")
        self.max_cached_videos = max_cached_videos
        self._video_cache: OrderedDict[Path, Tensor] = OrderedDict()

    def _build_assignments(self) -> tuple[list[list[MotionSegment]], dict[str, int]]:
        assert self.camera_motion_bank is not None
        assignments = []
        summary = {"full_bucket": 0, "displacement_bucket": 0, "gap_only": 0}
        for path in self.records:
            record = load_pair_supervision(path)
            selected, fallbacks = self.camera_motion_bank.assignments(
                source_video=str(Path(record.metadata["video"]).resolve()),
                gap=int(record.metadata["gap"]),
                key=str(path.resolve()),
                seed=self.view_assignment_seed,
                count=2,
            )
            assignments.append(selected)
            for fallback in fallbacks:
                summary[fallback] += 1
        return assignments, summary

    def __len__(self) -> int:
        return len(self.records)

    def _frames(self, path: Path) -> Tensor:
        path = path.resolve()
        frames = self._video_cache.pop(path, None)
        if frames is None:
            frames = _decode_video(path, self.output_hw)
        self._video_cache[path] = frames
        while len(self._video_cache) > self.max_cached_videos:
            self._video_cache.popitem(last=False)
        return frames

    @staticmethod
    def _tensor_record(record: PairSupervision) -> dict[str, Tensor]:
        return {
            "points0": torch.from_numpy(record.points0).float(),
            "points1": torch.from_numpy(record.points1).float(),
            "canonical_residual": torch.from_numpy(record.canonical_residual).float(),
            "valid": torch.from_numpy(record.valid).bool(),
            "background": torch.from_numpy(record.background).bool(),
            "interaction": torch.from_numpy(record.interaction).bool(),
            "visibility": torch.from_numpy(record.visibility).float(),
            "confidence": torch.from_numpy(record.confidence).float(),
            "fit_confidence": torch.tensor(record.fit_confidence, dtype=torch.float32),
            "global_matrix": torch.from_numpy(record.global_matrix).float(),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = load_pair_supervision(self.records[index])
        frame_start = int(record.metadata["frame_start"])
        frame_end = int(record.metadata["frame_end"])
        frames = self._frames(Path(record.metadata["video"]))
        if not 0 <= frame_start < len(frames) or not 0 <= frame_end < len(frames):
            raise IndexError(
                f"pair frames {frame_start}/{frame_end} outside decoded video length {len(frames)}"
            )
        result = self._tensor_record(record)
        if record.hand_masks is not None:
            result["hand_masks"] = torch.from_numpy(record.hand_masks).bool()
        result["videos"] = frames[[frame_start, frame_end]].clone()
        if self.augment_camera:
            if self.training_view_mode is not None:
                self._add_training_views(result, index)
            elif self.camera_augmentation_mode == "addition":
                augmentation = sample_camera_augmentation(
                    translation_pixels=self.camera_translation_pixels,
                    image_hw=self.output_hw,
                    rotation_degrees=self.camera_rotation_degrees,
                    scale_fraction=self.camera_scale_fraction,
                )
                points1_augmented = transform_points(
                    result["points1"][None],
                    augmentation[None],
                )[0]
                inside = torch.all(
                    (points1_augmented >= 0.0) & (points1_augmented <= 1.0),
                    dim=-1,
                )
                videos_augmented = result["videos"].clone()
                videos_augmented[1] = warp_target_image(
                    videos_augmented[1],
                    augmentation,
                    padding_mode=self.camera_padding_mode,
                )
                result.update(
                    {
                        "videos_augmented": videos_augmented,
                        "points1_augmented": points1_augmented,
                        "valid_augmented": result["valid"] & inside,
                        "global_matrix_augmented": augmentation @ result["global_matrix"],
                        "augmentation_matrix": augmentation,
                        "orbit_weight": torch.ones((), dtype=torch.float32),
                    }
                )
            else:
                source_global_matrix = result["global_matrix"].clone()
                replacements = torch.stack(
                    [
                        sample_camera_augmentation(
                            translation_pixels=self.camera_translation_pixels,
                            image_hw=self.output_hw,
                            rotation_degrees=self.camera_rotation_degrees,
                            scale_fraction=self.camera_scale_fraction,
                        )
                        for _ in range(2)
                    ]
                )
                target_matrices = torch.stack(
                    [
                        camera_replacement_target_matrix(
                            source_global_matrix,
                            replacement,
                        )
                        for replacement in replacements
                    ]
                )
                target_points = transform_points(
                    result["points1"][None].expand(2, -1, -1),
                    target_matrices,
                )
                inside = torch.all(
                    (target_points >= 0.0) & (target_points <= 1.0),
                    dim=-1,
                )
                common_valid = result["valid"] & inside[0] & inside[1]
                valid_count = result["valid"].sum().clamp_min(1)
                common_valid_fraction = common_valid.sum().float() / valid_count.float()
                target_images = warp_images(
                    result["videos"][1][None].expand(2, -1, -1, -1),
                    target_matrices,
                    padding_mode=self.camera_padding_mode,
                )
                replacement_videos = result["videos"][None].expand(2, -1, -1, -1, -1).clone()
                replacement_videos[:, 1] = target_images
                orbit_eligible = common_valid_fraction >= self.replacement_min_valid_fraction
                result.update(
                    {
                        "videos": replacement_videos[0],
                        "videos_augmented": replacement_videos[1],
                        "points1": target_points[0],
                        "points1_augmented": target_points[1],
                        "valid": common_valid,
                        "valid_augmented": common_valid.clone(),
                        "global_matrix": replacements[0],
                        "global_matrix_augmented": replacements[1],
                        "augmentation_matrix": replacements[1],
                        "replacement_matrix_original": source_global_matrix,
                        "replacement_target_matrices": target_matrices,
                        "replacement_common_valid_fraction": common_valid_fraction,
                        "orbit_weight": (
                            result["fit_confidence"] * orbit_eligible.to(torch.float32)
                        ),
                    }
                )
        return result

    def _add_training_views(self, result: dict[str, Tensor], index: int) -> None:
        natural_videos = result["videos"]
        natural_points1 = result["points1"]
        natural_matrix = result["global_matrix"]
        matrices: list[Tensor]
        target_matrices: list[Tensor | None]
        view_confidence: list[Tensor]
        view_kind: list[int]
        if self.training_view_mode == "addition_pair":
            augmentation = sample_camera_augmentation(
                translation_pixels=self.camera_translation_pixels,
                image_hw=self.output_hw,
                rotation_degrees=self.camera_rotation_degrees,
                scale_fraction=self.camera_scale_fraction,
            )
            matrices = [natural_matrix, augmentation @ natural_matrix]
            target_matrices = [None, augmentation]
            view_confidence = [result["fit_confidence"], result["fit_confidence"]]
            view_kind = [0, 1]
        else:
            assert self._empirical_assignments is not None
            assignments = self._empirical_assignments[index]
            empirical = [torch.from_numpy(segment.matrix).float() for segment in assignments]
            empirical_targets = [
                camera_replacement_target_matrix(natural_matrix, matrix) for matrix in empirical
            ]
            empirical_confidence = [
                torch.minimum(
                    result["fit_confidence"],
                    torch.tensor(segment.confidence, dtype=torch.float32),
                )
                for segment in assignments
            ]
            if self.training_view_mode == "empirical_pair":
                matrices = empirical
                target_matrices = empirical_targets
                view_confidence = empirical_confidence
                view_kind = [2, 2]
            elif self.training_view_mode == "natural_empirical_triplet":
                matrices = [natural_matrix, *empirical]
                target_matrices = [None, *empirical_targets]
                view_confidence = [result["fit_confidence"], *empirical_confidence]
                view_kind = [0, 2, 2]
            else:
                raise AssertionError(f"unexpected training view mode {self.training_view_mode}")

        view_points = []
        view_videos = []
        inside_masks = []
        for target_matrix in target_matrices:
            if target_matrix is None:
                points = natural_points1
                videos = natural_videos
            else:
                points = transform_points(natural_points1[None], target_matrix[None])[0]
                videos = natural_videos.clone()
                videos[1] = warp_target_image(
                    videos[1],
                    target_matrix,
                    padding_mode=self.camera_padding_mode,
                )
            view_points.append(points)
            view_videos.append(videos)
            inside_masks.append(torch.all((points >= 0.0) & (points <= 1.0), dim=-1))
        common_valid = result["valid"].clone()
        for inside in inside_masks:
            common_valid &= inside
        common_fraction = common_valid.sum().float() / result["valid"].sum().clamp_min(1).float()
        eligible = common_fraction >= self.replacement_min_valid_fraction
        result.update(
            {
                "view_videos": torch.stack(view_videos),
                "view_points1": torch.stack(view_points),
                "view_valid": common_valid[None].expand(len(view_points), -1).clone(),
                "view_global_matrix": torch.stack(matrices),
                "view_fit_confidence": torch.stack(view_confidence),
                "view_kind": torch.tensor(view_kind, dtype=torch.int64),
                "view_common_valid_fraction": common_fraction,
                "view_orbit_weight": torch.stack(view_confidence).min()
                * eligible.to(torch.float32),
            }
        )

