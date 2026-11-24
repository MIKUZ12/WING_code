from typing import List, Dict

import torch
from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


# ---------------------------------------------------------------------------
# Shared pose6 (xyz + XYZ Euler radians) <-> SE(3) helpers.
#
# These are module-level so newer transforms (RelativeUnified32DTransform,
# RelativeEef14Transform) can reuse them. The legacy RelativeUnified28DTransform
# keeps its own equivalent staticmethods untouched (the 28D comparison line
# still trains with it); the unit test asserts these module helpers produce
# identical results to that class, so the two copies cannot silently diverge.
# ---------------------------------------------------------------------------
def pose6_to_matrix(pose: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    position = pose[..., 0:3]
    euler = pose[..., 3:6]
    rotation = euler_angles_to_matrix(euler, convention)
    matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
    matrix[..., 0:3, 0:3] = rotation
    matrix[..., 0:3, 3] = position
    matrix[..., 3, 3] = 1
    return matrix


def matrix_to_pose6(matrix: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    position = matrix[..., 0:3, 3] / matrix[..., 3, 3][..., None]
    euler = matrix_to_euler_angles(matrix[..., 0:3, 0:3], convention)
    return torch.cat([position, euler], dim=-1)


def pose6_absolute_to_relative(pose: torch.Tensor, base_pose: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Express `pose` in the coordinate frame of `base_pose`: T_rel = inv(T_base) @ T_pose."""
    pose_matrix = pose6_to_matrix(pose, convention)
    base_pose_matrix = pose6_to_matrix(base_pose, convention)
    relative = torch.linalg.inv(base_pose_matrix) @ pose_matrix
    return matrix_to_pose6(relative, convention)


def pose6_relative_to_absolute(pose: torch.Tensor, base_pose: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Inverse of `pose6_absolute_to_relative`: T_abs = T_base @ T_rel."""
    pose_matrix = pose6_to_matrix(pose, convention)
    base_pose_matrix = pose6_to_matrix(base_pose, convention)
    absolute = base_pose_matrix @ pose_matrix
    return matrix_to_pose6(absolute, convention)


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch


class RelativeUnified28DTransform:
    """Convert unified 28D absolute targets into relative joint/EEF-pose actions.

    LEGACY (28D comparison line): used only by the unified28 / worstconf /
    28dpartial comparison configs. New EgoDex mainline work uses the native
    eef14 datasets with RelativeEef14Transform below.

    Joint arm dims and EEF pose dims are relative to the first state frame.
    EEF pose uses xyz + XYZ Euler angles in radians.
    Gripper dims remain absolute targets, matching GR00T/OpenPI-style configs.
    """

    joint_slices = (slice(0, 6), slice(7, 13))
    pose_slices = (slice(14, 20), slice(21, 27))
    euler_convention = "XYZ"

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch

        for k in self.keys:
            action = batch["action"][k].clone()
            state0 = batch["state"][k][..., :1, :]
            self._check_shape(k, action, state0)

            for sl in self.joint_slices:
                action[..., sl] = action[..., sl] - state0[..., sl]
            for sl in self.pose_slices:
                action[..., sl] = self._pose6_absolute_to_relative(action[..., sl], state0[..., sl])

            batch["action"][k] = action
        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k].clone()
            state0 = batch["state"][k][..., :1, :]
            self._check_shape(k, action, state0)

            for sl in self.joint_slices:
                action[..., sl] = action[..., sl] + state0[..., sl]
            for sl in self.pose_slices:
                action[..., sl] = self._pose6_relative_to_absolute(action[..., sl], state0[..., sl])

            batch["action"][k] = action
        return batch

    @staticmethod
    def _check_shape(key: str, action: torch.Tensor, state0: torch.Tensor):
        if action.shape[-1] != 28:
            raise ValueError(f"Unified 28D action key {key!r} must have 28 dims, got {action.shape[-1]}")
        if state0.shape[-1] != 28:
            raise ValueError(f"Unified 28D state key {key!r} must have 28 dims, got {state0.shape[-1]}")

    @staticmethod
    def _pose6_to_matrix(pose: torch.Tensor):
        position = pose[..., 0:3]
        euler = pose[..., 3:6]
        rotation = euler_angles_to_matrix(euler, RelativeUnified28DTransform.euler_convention)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0:3, 0:3] = rotation
        matrix[..., 0:3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose6(matrix: torch.Tensor):
        position = matrix[..., 0:3, 3] / matrix[..., 3, 3][..., None]
        euler = matrix_to_euler_angles(matrix[..., 0:3, 0:3], RelativeUnified28DTransform.euler_convention)
        return torch.cat([position, euler], dim=-1)

    @classmethod
    def _pose6_absolute_to_relative(cls, pose: torch.Tensor, base_pose: torch.Tensor):
        pose_matrix = cls._pose6_to_matrix(pose)
        base_pose_matrix = cls._pose6_to_matrix(base_pose)
        relative = torch.linalg.inv(base_pose_matrix) @ pose_matrix
        return cls._matrix_to_pose6(relative)

    @classmethod
    def _pose6_relative_to_absolute(cls, pose: torch.Tensor, base_pose: torch.Tensor):
        pose_matrix = cls._pose6_to_matrix(pose)
        base_pose_matrix = cls._pose6_to_matrix(base_pose)
        absolute = base_pose_matrix @ pose_matrix
        return cls._matrix_to_pose6(absolute)


class _RelativeEefPoseTransform:
    """Shared engine for fixed-layout eef-pose-relative action transforms.

    Subclasses pin the layout via two class attributes:
      pose_slices : eef pose (xyz + XYZ Euler rad) slices that become RELATIVE
                    to the first state frame via SE(3) frame change
                    T_rel = inv(T0) @ T_t (module-level pose6 helpers);
      DIM         : required last dim of action/state.
    Every other dim (grippers, joints, base) is ABSOLUTE passthrough.
    """

    pose_slices: tuple = ()
    euler_convention = "XYZ"
    DIM: int = -1

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch

        for k in self.keys:
            action = batch["action"][k].clone()
            state0 = batch["state"][k][..., :1, :]
            self._check_shape(k, action, state0)

            for sl in self.pose_slices:
                action[..., sl] = pose6_absolute_to_relative(
                    action[..., sl], state0[..., sl], self.euler_convention
                )
            # grippers / joints / base stay absolute (passthrough)

            batch["action"][k] = action
        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k].clone()
            state0 = batch["state"][k][..., :1, :]
            self._check_shape(k, action, state0)

            for sl in self.pose_slices:
                action[..., sl] = pose6_relative_to_absolute(
                    action[..., sl], state0[..., sl], self.euler_convention
                )

            batch["action"][k] = action
        return batch

    @classmethod
    def _check_shape(cls, key: str, action: torch.Tensor, state0: torch.Tensor):
        if action.shape[-1] != cls.DIM:
            raise ValueError(f"{cls.__name__}: action key {key!r} must have {cls.DIM} dims, got {action.shape[-1]}")
        if state0.shape[-1] != cls.DIM:
            raise ValueError(f"{cls.__name__}: state key {key!r} must have {cls.DIM} dims, got {state0.shape[-1]}")


class RelativeUnified32DTransform(_RelativeEefPoseTransform):
    """Convert unified 32D absolute targets into EEF-relative actions.

    Legacy experimental layout (eef-first):
      [0:14]  eef block  : [0:6] L eef pose, [6] L grip, [7:13] R eef pose, [13] R grip
      [14:30] joint block: [14:21] L joint(7), [21] L grip, [22:29] R joint(7), [29] R grip
      [30:32] base reserved

    Representation (locked with user):
      - EEF pose ([0:6], [7:13]) -> RELATIVE to the first state frame via SE(3)
        frame change T_rel = inv(T0) @ T_t (position re-expressed in the first
        frame's local axes, rotation as the relative rotation), re-encoded as
        xyz + XYZ Euler(rad). NOT a per-component subtraction.
      - Everything else (eef grippers, the whole joint block, base) is ABSOLUTE
        (passthrough). Absolute joints match the existing real-robot raw-joint
        recipe and keep deployment unchanged.

    Because only the EEF pose slices are touched and SE(3) frame change of a zero
    pose is the identity (euler[0,0,0] -> I, inv(I)@I = I), this transform is a
    no-op on joint-only samples (whose eef slots are masked to 0) and is therefore
    safe to apply uniformly to eef-only / joint-only / both groups.
    """

    pose_slices = (slice(0, 6), slice(7, 13))
    DIM = 32


class RelativeEef14Transform(_RelativeEefPoseTransform):
    """Convert native 14D eef absolute targets into EEF-relative actions.

    The "natural transform" of the mainline EgoDex eef-14 datasets
    (data/egodex_pnp_15tasks_eef14, layout = EEF_14_NAMES of
    scripts/data_process/slice_unified28_to_eef14.py):
      [0:6]  L eef pose (xyz + XYZ Euler rad) -> relative to first state frame
      [6]    L gripper                        -> absolute
      [7:13] R eef pose                       -> relative to first state frame
      [13]   R gripper                        -> absolute

    Dim-for-dim this is exactly the 32D layout's eef block [0:14], so the pose
    slices (and the SE(3) math) coincide with RelativeUnified32DTransform; only
    the total dim differs. Used by per_embodiment configs as the per-source
    transform (no slot placement, no dim-mask).
    """

    pose_slices = (slice(0, 6), slice(7, 13))
    DIM = 14
