"""Datasets and schemas for RGB-only WING-LAM distillation."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _crop_four_three(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target = 4.0 / 3.0
    if width / height > target:
        crop_width = int(height * target)
        left = (width - crop_width) // 2
        return frame[:, left : left + crop_width]
    if width / height < target:
        crop_height = int(width / target)
        top = (height - crop_height) // 2
        return frame[top : top + crop_height]
    return frame


def preprocess_rgb_frame(
    frame: np.ndarray,
    output_hw: tuple[int, int] = (240, 320),
) -> Tensor:
    """Preprocess one uint8 RGB frame for the student model."""
    rgb = _crop_four_three(np.asarray(frame))
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB frame must have shape [H,W,3], got {rgb.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    tensor = F.interpolate(
        tensor[None],
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )[0]
    return tensor.permute(1, 2, 0).contiguous()


def preprocess_rgb_frames(
    frames: list[np.ndarray],
    output_hw: tuple[int, int] = (240, 320),
) -> Tensor:
    """Batch the exact CPU preprocessing used by ``preprocess_rgb_frame``."""
    if not frames:
        raise ValueError("frames must not be empty")
    cropped = [np.ascontiguousarray(_crop_four_three(np.asarray(frame))) for frame in frames]
    shapes = {frame.shape for frame in cropped}
    if len(shapes) != 1:
        raise ValueError(f"all frames in a batch must have the same shape, got {sorted(shapes)}")
    if cropped[0].ndim != 3 or cropped[0].shape[-1] != 3:
        raise ValueError(f"RGB frames must have shape [H,W,3], got {cropped[0].shape}")
    tensor = torch.from_numpy(np.stack(cropped)).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = F.interpolate(
        tensor,
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )
    return tensor.permute(0, 2, 3, 1).contiguous()


def decode_rgb_video(path: str | Path, output_hw: tuple[int, int] = (240, 320)) -> Tensor:
    """Decode, center-crop to 4:3, and resize exactly like WING-LAM training."""
    chunks: list[Tensor] = []
    pending: list[Tensor] = []

    def flush() -> None:
        if not pending:
            return
        tensor = torch.stack(pending).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = F.interpolate(tensor, size=output_hw, mode="bilinear", align_corners=False)
        chunks.append(tensor.permute(0, 2, 3, 1).contiguous())
        pending.clear()

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = _crop_four_three(frame.to_ndarray(format="rgb24"))
            pending.append(torch.from_numpy(np.ascontiguousarray(rgb)))
            if len(pending) == 32:
                flush()
    flush()
    if not chunks:
        raise RuntimeError(f"video has no decodable frames: {path}")
    return torch.cat(chunks, dim=0)


@dataclass(frozen=True, order=True)
class PairKey:
    video: str
    frame_start: int
    frame_end: int
    gap: int

    @classmethod
    def create(
        cls,
        video: str | Path,
        frame_start: int,
        frame_end: int,
        gap: int,
    ) -> "PairKey":
        start, end, gap = int(frame_start), int(frame_end), int(gap)
        if start < 0 or end <= start or gap != end - start:
            raise ValueError(f"invalid frame pair {start}->{end} with gap={gap}")
        return cls(os.path.realpath(os.fspath(video)), start, end, gap)

    @property
    def pair_id(self) -> str:
        return f"{quote(self.video, safe='')}:{self.frame_start}:{self.frame_end}:{self.gap}"


def load_student_index(path: str | Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty RGB student index: {path}")
    required = {
        "pair_id",
        "video",
        "frame_start",
        "frame_end",
        "gap",
        "target_npz",
        "target_index",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"index row {index} is missing {sorted(missing)}")
        PairKey.create(row["video"], row["frame_start"], row["frame_end"], row["gap"])
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("RGB student index contains duplicate pair_id values")
    return rows


def validate_target_archive(
    path: str | Path,
    *,
    pair_list: str | Path | None = None,
    teacher_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Validate target arrays and optional source paths without loading pickle."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "z_int",
            "interaction_available",
            "video",
            "frame_start",
            "frame_end",
            "gap",
            "split",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"teacher target archive is missing {sorted(missing)}: {path}")
        targets = np.asarray(data["z_int"])
        if targets.ndim != 2 or targets.shape[1] != 32 or not np.isfinite(targets).all():
            raise ValueError(f"z_int must be finite [N,32], got {targets.shape}: {path}")
        if any(len(data[name]) != len(targets) for name in required - {"metadata_json"}):
            raise ValueError(f"teacher target arrays have inconsistent lengths: {path}")
        metadata = json.loads(str(data["metadata_json"].item()))
    if pair_list is not None:
        pair_list = Path(pair_list).resolve()
        if Path(metadata["pair_list"]).resolve() != pair_list:
            raise ValueError("teacher target pair-list path does not match")
    if teacher_checkpoint is not None:
        teacher_checkpoint = Path(teacher_checkpoint).resolve()
        recorded_teacher = metadata.get("wing_lam_checkpoint")
        if recorded_teacher is None or Path(recorded_teacher).resolve() != teacher_checkpoint:
            raise ValueError("teacher target checkpoint path does not match")
    return metadata | {"pairs": int(targets.shape[0])}


class RGBStudentDataset(Dataset[dict[str, Tensor]]):
    """Dataset whose model-facing input is exactly a two-frame RGB tensor."""

    def __init__(
        self,
        index: str | Path,
        *,
        output_hw: tuple[int, int] = (240, 320),
        max_cached_videos: int = 8,
    ) -> None:
        self.rows = load_student_index(index)
        self.output_hw = output_hw
        if max_cached_videos <= 0:
            raise ValueError("max_cached_videos must be positive")
        self.max_cached_videos = max_cached_videos
        self._videos: OrderedDict[str, Tensor] = OrderedDict()
        self._targets: dict[str, np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _video(self, path: str) -> Tensor:
        frames = self._videos.pop(path, None)
        if frames is None:
            frames = decode_rgb_video(path, self.output_hw)
        self._videos[path] = frames
        while len(self._videos) > self.max_cached_videos:
            self._videos.popitem(last=False)
        return frames

    def _target(self, row: dict[str, object]) -> Tensor:
        path = str(row["target_npz"])
        data = self._targets.get(path)
        if data is None:
            data = np.load(path, allow_pickle=False)
            self._targets[path] = data
        target = np.asarray(data["z_int"][int(row["target_index"])], dtype=np.float32)
        if target.shape != (32,) or not np.isfinite(target).all():
            raise ValueError(f"invalid teacher target for {row['pair_id']}")
        return torch.from_numpy(target.copy())

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        frames = self._video(str(row["video"]))
        start, end = int(row["frame_start"]), int(row["frame_end"])
        if not 0 <= start < len(frames) or not 0 <= end < len(frames):
            raise IndexError(f"frames {start}/{end} outside {len(frames)} for {row['video']}")
        return {
            "videos": frames[[start, end]].clone(),
            "target": self._target(row),
            "pair_index": torch.tensor(index, dtype=torch.int64),
        }

    def close(self) -> None:
        for data in self._targets.values():
            data.close()
        self._targets.clear()


def read_pair_list(path: str | Path) -> list[Path]:
    path = Path(path)
    records = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        record = Path(value)
        records.append((record if record.is_absolute() else path.parent / record).resolve())
    if not records:
        raise ValueError(f"empty pair list: {path}")
    return records


def source_video_id(pair_cache: Path, key: PairKey) -> str:
    if Path(key.video).name == "clip.mp4" and pair_cache.parent.parent.parent.name == "windows":
        window_id = pair_cache.parent.parent.name
        return window_id.rsplit("_w", 1)[0] if "_w" in window_id else window_id
    return key.video

