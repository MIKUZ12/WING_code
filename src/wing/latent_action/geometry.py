"""Geometry primitives for WING-LAM weak supervision.

All point coordinates are expected in the normalized image coordinate system
``[0, 1] x [0, 1]``.  The module deliberately uses NumPy only: tracker and
segmentation models run in isolated environments, while their cache is
consumed by the WING environment without OpenCV/mmcv dependencies.

Convention
----------
``W`` maps a source-frame point to its camera-dominant destination:

    p_camera = W(p0)

The destination residual is useful for visualization, but cross-composition
uses the source-canonical residual:

    r_int = inv(W)(p1) - p0
    p1 = W(p0 + r_int)

With an anchor-preserving augmentation applied only to the destination frame,
``p1_tilde = A(p1)`` and ``W_tilde = A @ W``.  The canonical residual is then
exactly invariant (up to floating-point error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GlobalModel = Literal["affine", "homography"]


class GlobalMotionFitError(ValueError):
    """Raised when a valid global motion model cannot be estimated."""


@dataclass(frozen=True)
class GlobalMotionFit:
    matrix: np.ndarray
    model_type: GlobalModel
    inliers: np.ndarray
    reprojection_errors: np.ndarray
    confidence: float
    num_candidates: int
    num_inliers: int
    inlier_ratio: float
    median_inlier_error: float
    rmse_inlier_error: float
    spatial_coverage: float
    condition_number: float
    determinant: float


@dataclass(frozen=True)
class PairMotionDecomposition:
    camera_displacement: np.ndarray
    destination_residual: np.ndarray
    canonical_residual: np.ndarray


@dataclass(frozen=True)
class InteractionRegion:
    mask: np.ndarray
    residual_threshold: float
    num_seeds: int
    num_candidates: int
    num_selected: int
    truncated: bool


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {result.shape}")
    return result


def _as_mask(mask: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {result.shape}")
    return result


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"matrix must be finite with shape [3, 3], got {matrix.shape}")
    scale = matrix[2, 2]
    if abs(scale) > 1e-12:
        matrix = matrix / scale
    else:
        norm = np.linalg.norm(matrix)
        if norm <= 1e-12:
            raise ValueError("matrix is numerically zero")
        matrix = matrix / norm
    return matrix


def apply_homography(
    points: np.ndarray,
    matrix: np.ndarray,
    *,
    min_denominator: float = 1e-10,
) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to ``[N,2]`` points."""

    points = _as_points(points, "points")
    matrix = _normalize_matrix(matrix)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed = homogeneous @ matrix.T
    denominator = transformed[:, 2]
    if np.any(np.abs(denominator) <= min_denominator):
        raise ValueError("homography maps one or more points to infinity")
    return transformed[:, :2] / denominator[:, None]


def _fit_affine(points0: np.ndarray, points1: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(points0) < 3:
        raise GlobalMotionFitError("affine fitting requires at least 3 points")
    design = np.zeros((2 * len(points0), 6), dtype=np.float64)
    target = points1.reshape(-1)
    design[0::2, 0:2] = points0
    design[0::2, 2] = 1.0
    design[1::2, 3:5] = points0
    design[1::2, 5] = 1.0
    row_weights = np.repeat(np.sqrt(np.maximum(weights, 0.0)), 2)
    weighted_design = design * row_weights[:, None]
    weighted_target = target * row_weights
    parameters, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    if rank < 6:
        raise GlobalMotionFitError("degenerate affine point configuration")
    matrix = np.array(
        [
            [parameters[0], parameters[1], parameters[2]],
            [parameters[3], parameters[4], parameters[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if abs(np.linalg.det(matrix[:2, :2])) <= 1e-10:
        raise GlobalMotionFitError("estimated affine matrix is singular")
    return matrix


def _normalization_transform(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise GlobalMotionFitError("point weights must have positive sum")
    center = np.sum(points * weights[:, None], axis=0) / total_weight
    distances = np.linalg.norm(points - center, axis=1)
    mean_distance = float(np.sum(distances * weights) / total_weight)
    if mean_distance <= 1e-10:
        raise GlobalMotionFitError("degenerate homography point configuration")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = apply_homography(points, transform)
    return normalized, transform


def _fit_homography(points0: np.ndarray, points1: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(points0) < 4:
        raise GlobalMotionFitError("homography fitting requires at least 4 points")
    normalized0, transform0 = _normalization_transform(points0, weights)
    normalized1, transform1 = _normalization_transform(points1, weights)
    design = np.zeros((2 * len(points0), 9), dtype=np.float64)
    x, y = normalized0[:, 0], normalized0[:, 1]
    u, v = normalized1[:, 0], normalized1[:, 1]
    design[0::2] = np.stack([-x, -y, -np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), u * x, u * y, u], axis=1)
    design[1::2] = np.stack([np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), -x, -y, -np.ones_like(x), v * x, v * y, v], axis=1)
    design *= np.repeat(np.sqrt(np.maximum(weights, 0.0)), 2)[:, None]
    _, singular_values, vh = np.linalg.svd(design, full_matrices=True)
    if len(points0) > 4 and singular_values[-1] <= 0:
        raise GlobalMotionFitError("degenerate homography design matrix")
    normalized_matrix = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(transform1) @ normalized_matrix @ transform0
    matrix = _normalize_matrix(matrix)
    if abs(np.linalg.det(matrix)) <= 1e-10:
        raise GlobalMotionFitError("estimated homography is singular")
    return matrix


def _fit_model(
    points0: np.ndarray,
    points1: np.ndarray,
    weights: np.ndarray,
    model_type: GlobalModel,
) -> np.ndarray:
    if model_type == "affine":
        return _fit_affine(points0, points1, weights)
    if model_type == "homography":
        return _fit_homography(points0, points1, weights)
    raise ValueError(f"unsupported global motion model: {model_type!r}")


def _convex_hull_area(points: np.ndarray) -> float:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        area += point[0] * next_point[1] - point[1] * next_point[0]
    return abs(area) * 0.5


def fit_global_motion(
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    valid: np.ndarray,
    background: np.ndarray,
    point_confidence: np.ndarray | None = None,
    model_type: GlobalModel = "affine",
    ransac_threshold: float = 0.01,
    max_iterations: int = 1000,
    random_seed: int = 0,
    minimum_spatial_coverage: float = 0.15,
    maximum_condition_number: float = 1e8,
) -> GlobalMotionFit:
    """Fit an affine/homography background warp with deterministic RANSAC.

    ``ransac_threshold`` is measured in normalized image coordinates.  A value
    of ``0.01`` corresponds to 3.2 pixels at width 320.
    """

    points0 = _as_points(points0, "points0")
    points1 = _as_points(points1, "points1")
    if points1.shape != points0.shape:
        raise ValueError(f"points1 shape {points1.shape} != points0 shape {points0.shape}")
    length = len(points0)
    valid = _as_mask(valid, length, "valid")
    background = _as_mask(background, length, "background")
    if ransac_threshold <= 0 or max_iterations <= 0:
        raise ValueError("ransac_threshold and max_iterations must be positive")
    if point_confidence is None:
        weights = np.ones(length, dtype=np.float64)
    else:
        weights = np.asarray(point_confidence, dtype=np.float64)
        if weights.shape != (length,):
            raise ValueError(f"point_confidence must have shape [{length}], got {weights.shape}")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("point_confidence must be finite and non-negative")

    finite = np.all(np.isfinite(points0), axis=1) & np.all(np.isfinite(points1), axis=1)
    candidate_mask = valid & background & finite & (weights > 0)
    candidate_indices = np.flatnonzero(candidate_mask)
    sample_size = 3 if model_type == "affine" else 4
    if len(candidate_indices) < sample_size:
        raise GlobalMotionFitError(
            f"{model_type} requires at least {sample_size} valid background points; "
            f"got {len(candidate_indices)}"
        )
    candidate_weights = weights[candidate_indices]
    probabilities = candidate_weights / candidate_weights.sum()
    rng = np.random.default_rng(random_seed)

    best_matrix: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_score = -np.inf
    best_median = np.inf
    for _ in range(max_iterations):
        sample_local = rng.choice(len(candidate_indices), size=sample_size, replace=False, p=probabilities)
        sample_indices = candidate_indices[sample_local]
        try:
            matrix = _fit_model(
                points0[sample_indices],
                points1[sample_indices],
                weights[sample_indices],
                model_type,
            )
            predicted = apply_homography(points0[candidate_indices], matrix)
        except (GlobalMotionFitError, ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm(predicted - points1[candidate_indices], axis=1)
        inliers = errors <= ransac_threshold
        if int(inliers.sum()) < sample_size:
            continue
        score = float(candidate_weights[inliers].sum())
        median = float(np.median(errors[inliers]))
        if score > best_score or (np.isclose(score, best_score) and median < best_median):
            best_matrix, best_inliers = matrix, inliers
            best_score, best_median = score, median

    if best_matrix is None or best_inliers is None:
        raise GlobalMotionFitError(f"RANSAC failed to estimate a {model_type}")

    inlier_indices = candidate_indices[best_inliers]
    try:
        refined = _fit_model(points0[inlier_indices], points1[inlier_indices], weights[inlier_indices], model_type)
        candidate_predictions = apply_homography(points0[candidate_indices], refined)
    except (GlobalMotionFitError, ValueError, np.linalg.LinAlgError) as error:
        raise GlobalMotionFitError(f"{model_type} refinement failed: {error}") from error
    candidate_errors = np.linalg.norm(candidate_predictions - points1[candidate_indices], axis=1)
    refined_inliers = candidate_errors <= ransac_threshold
    if int(refined_inliers.sum()) < sample_size:
        raise GlobalMotionFitError(f"refined {model_type} has too few inliers")

    full_errors = np.full(length, np.nan, dtype=np.float64)
    all_finite_indices = np.flatnonzero(valid & finite)
    full_errors[all_finite_indices] = np.linalg.norm(
        apply_homography(points0[all_finite_indices], refined) - points1[all_finite_indices],
        axis=1,
    )
    full_inliers = np.zeros(length, dtype=bool)
    full_inliers[candidate_indices[refined_inliers]] = True
    inlier_errors = candidate_errors[refined_inliers]
    inlier_weight = float(candidate_weights[refined_inliers].sum())
    inlier_ratio = inlier_weight / float(candidate_weights.sum())
    median_error = float(np.median(inlier_errors))
    rmse_error = float(np.sqrt(np.mean(inlier_errors**2)))
    spatial_coverage = _convex_hull_area(points0[candidate_indices[refined_inliers]])
    condition_number = float(np.linalg.cond(refined))
    determinant = float(np.linalg.det(refined))
    if not np.isfinite(condition_number) or condition_number > maximum_condition_number:
        raise GlobalMotionFitError(
            f"estimated {model_type} is ill-conditioned: condition={condition_number:.3g}"
        )

    coverage_score = min(spatial_coverage / max(minimum_spatial_coverage, 1e-12), 1.0)
    error_score = float(np.exp(-median_error / ransac_threshold))
    condition_score = min(1.0, 1e4 / max(condition_number, 1.0))
    fit_confidence = float(np.clip(inlier_ratio * error_score * np.sqrt(coverage_score) * condition_score, 0.0, 1.0))
    return GlobalMotionFit(
        matrix=refined,
        model_type=model_type,
        inliers=full_inliers,
        reprojection_errors=full_errors,
        confidence=fit_confidence,
        num_candidates=len(candidate_indices),
        num_inliers=int(refined_inliers.sum()),
        inlier_ratio=inlier_ratio,
        median_inlier_error=median_error,
        rmse_inlier_error=rmse_error,
        spatial_coverage=spatial_coverage,
        condition_number=condition_number,
        determinant=determinant,
    )


def decompose_pair_motion(
    points0: np.ndarray,
    points1: np.ndarray,
    matrix: np.ndarray,
) -> PairMotionDecomposition:
    """Compute camera, destination-residual and canonical-residual motion."""

    points0 = _as_points(points0, "points0")
    points1 = _as_points(points1, "points1")
    if points1.shape != points0.shape:
        raise ValueError(f"points1 shape {points1.shape} != points0 shape {points0.shape}")
    matrix = _normalize_matrix(matrix)
    condition = np.linalg.cond(matrix)
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError(f"matrix is not safely invertible: condition={condition:.3g}")
    camera_endpoint = apply_homography(points0, matrix)
    canonical_endpoint = apply_homography(points1, np.linalg.inv(matrix))
    return PairMotionDecomposition(
        camera_displacement=camera_endpoint - points0,
        destination_residual=points1 - camera_endpoint,
        canonical_residual=canonical_endpoint - points0,
    )


def compose_endpoint(
    points0: np.ndarray,
    matrix: np.ndarray,
    canonical_residual: np.ndarray,
    interaction_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compose a global warp and source-canonical interaction residual."""

    points0 = _as_points(points0, "points0")
    canonical_residual = _as_points(canonical_residual, "canonical_residual")
    if canonical_residual.shape != points0.shape:
        raise ValueError(
            f"canonical_residual shape {canonical_residual.shape} != points0 shape {points0.shape}"
        )
    if interaction_mask is not None:
        mask = _as_mask(interaction_mask, len(points0), "interaction_mask")
        canonical_residual = canonical_residual * mask[:, None]
    return apply_homography(points0 + canonical_residual, matrix)


def _robust_residual_threshold(
    residual_norm: np.ndarray,
    reference_mask: np.ndarray,
    *,
    minimum_threshold: float,
    mad_scale: float,
) -> float:
    reference = residual_norm[reference_mask & np.isfinite(residual_norm)]
    if len(reference) == 0:
        return float(minimum_threshold)
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    robust_sigma = 1.4826 * mad
    return max(float(minimum_threshold), median + mad_scale * robust_sigma)


def build_interaction_region(
    points0: np.ndarray,
    canonical_residual: np.ndarray,
    *,
    valid: np.ndarray,
    seed_mask: np.ndarray,
    background_reference: np.ndarray,
    neighbor_radius: float = 0.06,
    minimum_residual: float = 0.005,
    mad_scale: float = 3.0,
    max_hops: int = 4,
    max_fraction: float = 0.35,
) -> InteractionRegion:
    """Grow a hand-seeded interaction region through high-residual tracks.

    The graph is a deterministic radius graph implemented with spatial hash
    bins.  Seeds are always retained.  Expansion stops at ``max_fraction`` of
    valid tracks and reports ``truncated=True`` instead of silently labeling
    most of the image as interaction.
    """

    points0 = _as_points(points0, "points0")
    canonical_residual = _as_points(canonical_residual, "canonical_residual")
    if canonical_residual.shape != points0.shape:
        raise ValueError("canonical_residual must match points0")
    length = len(points0)
    valid = _as_mask(valid, length, "valid")
    seed_mask = _as_mask(seed_mask, length, "seed_mask") & valid
    background_reference = _as_mask(background_reference, length, "background_reference") & valid
    if neighbor_radius <= 0 or minimum_residual < 0 or max_hops < 0:
        raise ValueError("neighbor_radius must be positive; residual/hops must be non-negative")
    if not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0, 1]")

    residual_norm = np.linalg.norm(canonical_residual, axis=1)
    threshold = _robust_residual_threshold(
        residual_norm,
        background_reference,
        minimum_threshold=minimum_residual,
        mad_scale=mad_scale,
    )
    finite = np.all(np.isfinite(points0), axis=1) & np.isfinite(residual_norm)
    eligible = valid & finite & ((residual_norm >= threshold) | seed_mask)
    eligible_indices = np.flatnonzero(eligible)
    selected = seed_mask.copy()
    seed_indices = np.flatnonzero(seed_mask)
    if len(seed_indices) == 0 or len(eligible_indices) == 0:
        return InteractionRegion(selected, threshold, len(seed_indices), len(eligible_indices), int(selected.sum()), False)

    cells: dict[tuple[int, int], list[int]] = {}
    for index in eligible_indices:
        cell = tuple(np.floor(points0[index] / neighbor_radius).astype(np.int64))
        cells.setdefault(cell, []).append(int(index))
    for members in cells.values():
        members.sort()

    valid_count = int(valid.sum())
    max_selected = max(len(seed_indices), int(np.ceil(valid_count * max_fraction)))
    queue = [(int(index), 0) for index in seed_indices]
    cursor = 0
    truncated = False
    while cursor < len(queue):
        current, hops = queue[cursor]
        cursor += 1
        if hops >= max_hops:
            continue
        base_cell = np.floor(points0[current] / neighbor_radius).astype(np.int64)
        neighbors: list[int] = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                neighbors.extend(cells.get((int(base_cell[0] + offset_x), int(base_cell[1] + offset_y)), ()))
        for neighbor in sorted(set(neighbors)):
            if selected[neighbor]:
                continue
            if np.linalg.norm(points0[neighbor] - points0[current]) > neighbor_radius:
                continue
            if int(selected.sum()) >= max_selected:
                truncated = True
                break
            selected[neighbor] = True
            queue.append((neighbor, hops + 1))
        if truncated:
            break

    return InteractionRegion(
        mask=selected,
        residual_threshold=threshold,
        num_seeds=len(seed_indices),
        num_candidates=len(eligible_indices),
        num_selected=int(selected.sum()),
        truncated=truncated,
    )
