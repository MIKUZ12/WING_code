"""Shared video and cache helpers for WING-LAM data preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


def four_three_crop_box(height: int, width: int) -> tuple[int, int, int, int]:
    """Return ``top, left, crop_height, crop_width`` for a centered 4:3 crop."""

    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got {height}/{width}")
    target_ratio = 4.0 / 3.0
    if width / height > target_ratio:
        crop_height = height
        crop_width = int(height * target_ratio)
    elif width / height < target_ratio:
        crop_height = int(width / target_ratio)
        crop_width = width
    else:
        crop_height, crop_width = height, width
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width


def decode_video_clip(
    video_path: str | Path,
    *,
    start_frame: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode a contiguous RGB uint8 clip with OpenCV."""

    if start_frame < 0 or num_frames <= 0:
        raise ValueError("start_frame must be non-negative and num_frames positive")
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    for offset in range(num_frames):
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        indices.append(start_frame + offset)
    capture.release()
    if len(frames) != num_frames:
        raise ValueError(
            f"{video_path}: requested {num_frames} frames from {start_frame}, decoded {len(frames)}"
        )
    return np.stack(frames), np.asarray(indices, dtype=np.int64), fps


def crop_and_resize_four_three(
    frames_rgb: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center-crop RGB frames to 4:3 and resize with bilinear interpolation."""

    frames = np.asarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames_rgb must have shape [T,H,W,3], got {frames.shape}")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output size must be positive")
    top, left, crop_height, crop_width = four_three_crop_box(frames.shape[1], frames.shape[2])
    cropped = frames[:, top : top + crop_height, left : left + crop_width]
    resized = np.stack(
        [
            cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            for frame in cropped
        ]
    )
    return resized, (top, left, crop_height, crop_width)


def save_npz_atomic(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def metadata_array(payload: dict) -> np.ndarray:
    return np.asarray(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))

