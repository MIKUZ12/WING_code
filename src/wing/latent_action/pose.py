"""Pose geometry helpers used alongside WING-LAM."""

from __future__ import annotations

import numpy as np


def rotation_homography_normalized(
    intrinsic: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Return the normalized-image homography induced by camera rotation only.

    Poses are world-from-camera ``xyz + wxyz``. Translation and scene depth
    are omitted, so the result represents rotation rather than full camera
    compensation.
    """
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    source_pose = np.asarray(source_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    height, width = image_hw
    if intrinsic.shape != (3, 3) or source_pose.shape != (7,) or target_pose.shape != (7,):
        raise ValueError("intrinsic/source_pose/target_pose must have shapes [3,3]/[7]/[7]")
    if min(height, width) < 2 or not np.isfinite(intrinsic).all():
        raise ValueError("image size and intrinsic must be finite and valid")
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
    pixels_to_normalized = np.asarray(
        [
            [1.0 / (crop_width - 1), 0.0, -left / (crop_width - 1)],
            [0.0, 1.0 / (crop_height - 1), -top / (crop_height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized_intrinsic = pixels_to_normalized @ intrinsic
    source_rotation = quaternion_to_matrix(source_pose[3:7])
    target_rotation = quaternion_to_matrix(target_pose[3:7])
    relative_rotation = target_rotation.T @ source_rotation
    homography = normalized_intrinsic @ relative_rotation @ np.linalg.inv(normalized_intrinsic)
    return homography / homography[2, 2]


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    result = quaternion.copy()
    result[..., 1:] *= -1
    return result / np.maximum(np.sum(quaternion**2, axis=-1, keepdims=True), 1e-12)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    quaternion = np.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[..., :1], 0.0, 1.0))
    scale = np.where(vector_norm > 1e-8, angle / np.maximum(vector_norm, 1e-12), 2.0)
    return vector * scale


def camera_twist(pose: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return source-camera-frame translation and rotation-vector deltas."""
    pose = np.asarray(pose, dtype=np.float64)
    first, second = pose[starts], pose[ends]
    rotation = quaternion_to_matrix(first[:, 3:7])
    translation = np.einsum(
        "nij,nj->ni",
        rotation.transpose(0, 2, 1),
        second[:, :3] - first[:, :3],
    )
    relative_quaternion = quaternion_multiply(
        quaternion_inverse(first[:, 3:7]),
        second[:, 3:7],
    )
    return np.concatenate([translation, quaternion_to_rotvec(relative_quaternion)], axis=1)


def hand_pose_in_head(hand_pose: np.ndarray, head_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express world hand poses in their same-frame head coordinate systems."""
    hand_pose = np.asarray(hand_pose, dtype=np.float64)
    head_pose = np.asarray(head_pose, dtype=np.float64)
    head_rotation = quaternion_to_matrix(head_pose[:, 3:7])
    translation = np.einsum(
        "nij,nj->ni",
        head_rotation.transpose(0, 2, 1),
        hand_pose[:, :3] - head_pose[:, :3],
    )
    quaternion = quaternion_multiply(
        quaternion_inverse(head_pose[:, 3:7]),
        hand_pose[:, 3:7],
    )
    return translation, quaternion


def compensated_hand_twist(
    hand_pose: np.ndarray,
    head_pose: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return hand motion after removing the same-frame head rigid transform."""
    translation, quaternion = hand_pose_in_head(hand_pose, head_pose)
    translation_delta = translation[ends] - translation[starts]
    rotation_delta = quaternion_multiply(
        quaternion_inverse(quaternion[starts]),
        quaternion[ends],
    )
    return np.concatenate([translation_delta, quaternion_to_rotvec(rotation_delta)], axis=1)
