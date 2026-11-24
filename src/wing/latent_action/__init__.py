"""WING-LAM models, datasets, and preprocessing utilities."""

from .cache import PairSupervision, load_pair_supervision, save_pair_supervision
from .geometry import (
    GlobalMotionFit,
    GlobalMotionFitError,
    InteractionRegion,
    PairMotionDecomposition,
    apply_homography,
    build_interaction_region,
    compose_endpoint,
    decompose_pair_motion,
    fit_global_motion,
)
from .lam import (
    WINGLAM,
    WINGLAMLossWeights,
    compose_points,
    lam_loss,
    transform_points,
)
from .lam_dataset import WINGLAMPairsDataset, discover_pair_records
from .pose import camera_twist, compensated_hand_twist, rotation_homography_normalized
from .rgb_student import (
    DreamDojoFramePairBackbone,
    RGBLatentActionStudent,
    build_rgb_student,
    load_rgb_student_checkpoint,
)
from .rgb_student_data import (
    PairKey as RGBStudentPairKey,
    RGBStudentDataset,
    decode_rgb_video,
    load_student_index,
    preprocess_rgb_frame,
    preprocess_rgb_frames,
    read_pair_list,
    source_video_id,
    validate_target_archive,
)


__all__ = [
    "DreamDojoFramePairBackbone",
    "GlobalMotionFit",
    "GlobalMotionFitError",
    "InteractionRegion",
    "PairMotionDecomposition",
    "PairSupervision",
    "RGBLatentActionStudent",
    "RGBStudentDataset",
    "RGBStudentPairKey",
    "WINGLAM",
    "WINGLAMLossWeights",
    "WINGLAMPairsDataset",
    "apply_homography",
    "build_interaction_region",
    "build_rgb_student",
    "camera_twist",
    "compose_endpoint",
    "compose_points",
    "compensated_hand_twist",
    "decode_rgb_video",
    "decompose_pair_motion",
    "discover_pair_records",
    "fit_global_motion",
    "lam_loss",
    "load_pair_supervision",
    "load_rgb_student_checkpoint",
    "load_student_index",
    "preprocess_rgb_frame",
    "preprocess_rgb_frames",
    "read_pair_list",
    "rotation_homography_normalized",
    "save_pair_supervision",
    "source_video_id",
    "transform_points",
    "validate_target_archive",
]
