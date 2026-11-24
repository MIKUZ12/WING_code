#!/usr/bin/env python3
"""Combine AllTracker tracks and EgoHOS masks into WING-LAM pair supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wing.latent_action import (  # noqa: E402
    PairSupervision,
    build_interaction_region,
    decompose_pair_motion,
    fit_global_motion,
    save_pair_supervision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--hand-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap", type=int, action="append", default=None)
    parser.add_argument("--global-model", choices=("affine", "homography"), default="affine")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--ransac-threshold", type=float, default=0.01)
    parser.add_argument("--neighbor-radius", type=float, default=0.06)
    parser.add_argument("--minimum-residual", type=float, default=0.005)
    parser.add_argument("--store-pair-hand-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata(archive: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(archive["metadata_json"].item()))


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x = np.rint(points[:, 0] * max(width - 1, 1)).astype(np.int64)
    y = np.rint(points[:, 1] * max(height - 1, 1)).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.zeros(len(points), dtype=bool)
    result[inside] = mask[y[inside], x[inside]]
    return result


def main() -> int:
    args = parse_args()
    gaps = sorted(set(args.gap or [1, 4]))
    if not gaps or gaps[0] <= 0:
        raise ValueError("gaps must be positive")
    with np.load(args.tracks, allow_pickle=False) as archive:
        tracks = archive["tracks"].astype(np.float64)
        visibility = archive["visibility"].astype(np.float64)
        confidence = archive["confidence"].astype(np.float64)
        frame_indices = archive["frame_indices"].astype(np.int64)
        track_metadata = _metadata(archive)
    with np.load(args.hand_masks, allow_pickle=False) as archive:
        hand_masks = archive["hand_masks"].astype(bool)
        mask_frame_indices = archive["frame_indices"].astype(np.int64)
        mask_metadata = _metadata(archive)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError(f"tracks must have shape [T,K,2], got {tracks.shape}")
    if visibility.shape != tracks.shape[:2] or confidence.shape != tracks.shape[:2]:
        raise ValueError("visibility/confidence shapes must match tracks[:2]")
    if hand_masks.ndim != 3 or hand_masks.shape[0] != tracks.shape[0]:
        raise ValueError("hand_masks must have shape [T,H,W] with the same T as tracks")
    if not np.array_equal(frame_indices, mask_frame_indices):
        raise ValueError("AllTracker and EgoHOS caches refer to different frame indices")
    if track_metadata.get("input_video") != mask_metadata.get("input_video"):
        raise ValueError("AllTracker and EgoHOS caches refer to different videos")
    if track_metadata.get("crop_box_tl_hw") != mask_metadata.get("crop_box_tl_hw"):
        raise ValueError("AllTracker and EgoHOS used different crop boxes")
    if track_metadata.get("tracking_hw") != mask_metadata.get("mask_hw"):
        raise ValueError("AllTracker and EgoHOS used different processing resolutions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failures: list[dict] = []
    for gap in gaps:
        for start in range(len(tracks) - gap):
            end = start + gap
            output = args.output_dir / f"pair_{int(frame_indices[start]):08d}_{int(frame_indices[end]):08d}_{args.global_model}.npz"
            if output.exists() and not args.overwrite:
                continue
            points0, points1 = tracks[start], tracks[end]
            finite = np.all(np.isfinite(points0), axis=1) & np.all(np.isfinite(points1), axis=1)
            inside = np.all((points0 >= 0) & (points0 <= 1), axis=1) & np.all((points1 >= 0) & (points1 <= 1), axis=1)
            valid = (
                finite
                & inside
                & (visibility[start] >= args.visibility_threshold)
                & (visibility[end] >= args.visibility_threshold)
                & (confidence[start] >= args.confidence_threshold)
                & (confidence[end] >= args.confidence_threshold)
            )
            hand_seed = valid & (
                _sample_mask(hand_masks[start], points0) | _sample_mask(hand_masks[end], points1)
            )
            background = valid & ~hand_seed
            point_confidence = np.sqrt(np.clip(confidence[start] * confidence[end], 0.0, 1.0))
            try:
                fit = fit_global_motion(
                    points0,
                    points1,
                    valid=valid,
                    background=background,
                    point_confidence=point_confidence,
                    model_type=args.global_model,
                    ransac_threshold=args.ransac_threshold,
                    random_seed=int(frame_indices[start]),
                )
                decomposition = decompose_pair_motion(points0, points1, fit.matrix)
                region = build_interaction_region(
                    points0,
                    decomposition.canonical_residual,
                    valid=valid,
                    seed_mask=hand_seed,
                    background_reference=fit.inliers,
                    neighbor_radius=args.neighbor_radius,
                    minimum_residual=args.minimum_residual,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                failures.append({"start": int(frame_indices[start]), "end": int(frame_indices[end]), "error": str(error)})
                continue
            metadata = {
                "video": track_metadata["input_video"],
                "frame_start": int(frame_indices[start]),
                "frame_end": int(frame_indices[end]),
                "gap": gap,
                "global_model": args.global_model,
                "coordinate_system": "normalized_xy_0_1",
                "processing_hw": track_metadata["tracking_hw"],
                "canonical_residual": "inv(W)(p1)-p0",
                "track_cache": str(args.tracks.resolve()),
                "mask_cache": str(args.hand_masks.resolve()),
                "fit": {
                    "num_candidates": fit.num_candidates,
                    "num_inliers": fit.num_inliers,
                    "inlier_ratio": fit.inlier_ratio,
                    "median_inlier_error": fit.median_inlier_error,
                    "rmse_inlier_error": fit.rmse_inlier_error,
                    "spatial_coverage": fit.spatial_coverage,
                    "condition_number": fit.condition_number,
                    "determinant": fit.determinant,
                },
                "interaction": {
                    "residual_threshold": region.residual_threshold,
                    "num_seeds": region.num_seeds,
                    "num_candidates": region.num_candidates,
                    "num_selected": region.num_selected,
                    "truncated": region.truncated,
                },
            }
            save_pair_supervision(
                output,
                PairSupervision(
                    points0=points0,
                    points1=points1,
                    visibility=np.minimum(visibility[start], visibility[end]),
                    confidence=point_confidence,
                    valid=valid,
                    hand_seed=hand_seed,
                    background=background,
                    interaction=region.mask,
                    global_matrix=fit.matrix,
                    camera_displacement=decomposition.camera_displacement,
                    destination_residual=decomposition.destination_residual,
                    canonical_residual=decomposition.canonical_residual,
                    fit_confidence=fit.confidence,
                    metadata=metadata,
                    hand_masks=hand_masks[[start, end]] if args.store_pair_hand_masks else None,
                ),
            )
            written += 1
    summary = {"written": written, "failures": len(failures), "failure_examples": failures[:20]}
    (args.output_dir / f"summary_{args.global_model}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

