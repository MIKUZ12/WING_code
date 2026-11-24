"""Clip-relative affine trajectory bank used by WING-LAM training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def relative_segment(trajectory: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return ``H_end @ inv(H_start)`` for an anchor-relative trajectory."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (3, 3):
        raise ValueError("trajectory must have shape [T,3,3]")
    if not 0 <= start < end < len(trajectory):
        raise ValueError("segment indices must satisfy 0 <= start < end < T")
    return trajectory[end] @ np.linalg.inv(trajectory[start])


def _stable_index(seed: int, key: str, size: int) -> int:
    if size <= 0:
        raise ValueError("stable index requires a non-empty pool")
    return random.Random(f"{seed}:{key}").randrange(size)


def _center_path(trajectory: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    center = np.asarray([0.5, 0.5, 1.0], dtype=np.float64)
    transformed = np.einsum("tij,j->ti", trajectory, center)
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed * np.asarray([image_hw[1] - 1, image_hw[0] - 1], dtype=np.float64)


@dataclass(frozen=True)
class MotionSegment:
    matrix: np.ndarray
    confidence: float
    video: str
    gap: int
    start: int
    end: int
    displacement_px: float
    speed_px: float
    acceleration_px: float
    bins: tuple[int, int, int]


class CameraMotionBank:
    """Validated bank plus deterministic, different-clip segment assignment."""

    def __init__(
        self,
        *,
        trajectories: np.ndarray,
        confidence: np.ndarray,
        videos: np.ndarray,
        frame_indices: np.ndarray,
        image_hw: tuple[int, int],
        metadata: dict,
    ) -> None:
        trajectories = np.asarray(trajectories, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        videos = np.asarray(videos).astype(str)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if trajectories.ndim != 4 or trajectories.shape[2:] != (3, 3):
            raise ValueError("trajectories must have shape [C,T,3,3]")
        if confidence.shape != trajectories.shape[:2]:
            raise ValueError("confidence must have shape [C,T]")
        if frame_indices.shape != trajectories.shape[:2] or videos.shape != (len(trajectories),):
            raise ValueError("bank video/frame metadata shapes do not match trajectories")
        if not np.all(np.isfinite(trajectories)) or not np.all(np.isfinite(confidence)):
            raise ValueError("motion bank must be finite")
        identity = np.eye(3, dtype=np.float64)
        if not np.allclose(trajectories[:, 0], identity, atol=1e-8):
            raise ValueError("every bank trajectory must start at identity")
        self.trajectories = trajectories
        self.confidence = confidence
        self.videos = videos
        self.frame_indices = frame_indices
        self.image_hw = image_hw
        self.metadata = metadata
        self._segments = self._build_segments()

    @classmethod
    def load(cls, path: str | Path) -> "CameraMotionBank":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            image_hw = tuple(int(value) for value in metadata["image_hw"])
            return cls(
                trajectories=archive["trajectories"].copy(),
                confidence=archive["fit_confidence"].copy(),
                videos=archive["video"].copy(),
                frame_indices=archive["frame_indices"].copy(),
                image_hw=image_hw,
                metadata=metadata,
            )

    def _build_segments(self) -> list[MotionSegment]:
        provisional = []
        for clip_index, trajectory in enumerate(self.trajectories):
            center = _center_path(trajectory, self.image_hw)
            for gap in (1, 4):
                for start in range(len(trajectory) - gap):
                    end = start + gap
                    path = center[start : end + 1]
                    steps = np.diff(path, axis=0)
                    displacement = float(np.linalg.norm(path[-1] - path[0]))
                    speed = float(np.linalg.norm(steps, axis=1).mean())
                    acceleration = (
                        float(np.linalg.norm(np.diff(steps, axis=0), axis=1).mean())
                        if len(steps) > 1
                        else 0.0
                    )
                    provisional.append(
                        {
                            "matrix": relative_segment(trajectory, start, end),
                            "confidence": float(
                                min(self.confidence[clip_index, start], self.confidence[clip_index, end])
                            ),
                            "video": self.videos[clip_index],
                            "gap": gap,
                            "start": start,
                            "end": end,
                            "displacement_px": displacement,
                            "speed_px": speed,
                            "acceleration_px": acceleration,
                        }
                    )
        segments: list[MotionSegment] = []
        for gap in (1, 4):
            selected = [item for item in provisional if item["gap"] == gap]
            values = np.asarray(
                [
                    [item["displacement_px"], item["speed_px"], item["acceleration_px"]]
                    for item in selected
                ],
                dtype=np.float64,
            )
            thresholds = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], axis=0)
            for item, value in zip(selected, values, strict=True):
                bins = tuple(int(np.searchsorted(thresholds[:, index], value[index], side="right")) for index in range(3))
                segments.append(MotionSegment(**item, bins=bins))
        return segments

    def assignments(
        self,
        *,
        source_video: str,
        gap: int,
        key: str,
        seed: int,
        count: int = 2,
    ) -> tuple[list[MotionSegment], list[str]]:
        candidates = [segment for segment in self._segments if segment.gap == gap and segment.video != source_video]
        if not candidates:
            raise ValueError(f"no different-clip motion-bank candidates for gap {gap}")
        full_buckets = sorted({segment.bins for segment in candidates})
        selected: list[MotionSegment] = []
        fallbacks: list[str] = []
        for draw in range(count):
            desired = full_buckets[_stable_index(seed, f"{key}:bucket:{draw}", len(full_buckets))]
            pool = [segment for segment in candidates if segment.bins == desired]
            fallback = "full_bucket"
            if not pool:
                pool = [segment for segment in candidates if segment.bins[0] == desired[0]]
                fallback = "displacement_bucket"
            if not pool:
                pool = candidates
                fallback = "gap_only"
            selected_videos = {segment.video for segment in selected}
            if selected_videos:
                distinct = [segment for segment in pool if segment.video not in selected_videos]
                if distinct:
                    pool = distinct
                else:
                    pool = [segment for segment in candidates if segment.video not in selected_videos]
                    fallback = "gap_only_distinct"
            if not pool:
                raise ValueError(
                    f"need {count} different-clip candidates for gap {gap}, "
                    f"but only found {len({segment.video for segment in candidates})} unique clips"
                )
            selected.append(pool[_stable_index(seed, f"{key}:segment:{draw}", len(pool))])
            fallbacks.append(fallback)
        return selected, fallbacks
