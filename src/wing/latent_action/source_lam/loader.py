"""Load and preprocess the source latent-action checkpoint used by WING-LAM."""

import torch
import torch.nn.functional as F
from einops import rearrange

from .model import LatentActionModel

# Architecture of the released LAM_400k.ckpt. Must match how DreamDojo
# instantiates the LAM (cosmos_predict2/_src/predict2/models/
# text2world_model_rectified_flow.py:179-188). The upstream LAM wrapper's
# defaults (model_dim=512, 8/8 blocks, 8 heads) do NOT match the checkpoint.
LAM_400K_KWARGS = dict(
    in_dim=3,
    model_dim=1024,
    latent_dim=32,
    patch_size=16,
    enc_blocks=24,
    dec_blocks=24,
    num_heads=16,
)

# Input resolution the LAM was trained on (H, W).
LAM_INPUT_HW = (240, 320)


def load_lam(ckpt_path: str, device: str = "cpu") -> LatentActionModel:
    """Build a LatentActionModel and load a Lightning checkpoint (LAM_400k.ckpt
    or any ``lam_project`` continue-training product).

    Lightning stores the weights under ``["state_dict"]`` with a ``lam.`` key
    prefix. Loads float32 weights and returns the model in eval mode with
    grads disabled.
    """
    model = LatentActionModel(**LAM_400K_KWARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k[len("lam."):]: v for k, v in state.items() if k.startswith("lam.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """uint8 RGB frames ``(T, H, W, C)`` -> float32 ``(T, 240, 320, C)`` in [0,1].

    Mirrors the LAM training preprocessing (DreamDojo
    external/lam_project/lam/dataset.py:221-239): /255, center-crop to 4:3,
    bilinear resize to 240x320. Frames must be RGB (upstream decodes with
    cv2 + an explicit BGR->RGB conversion).
    """
    video = frames.float() / 255.0
    target_ratio = 640 / 480
    if video.shape[2] / video.shape[1] > target_ratio:
        target_height = video.shape[1]
        target_width = int(video.shape[1] * target_ratio)
    elif video.shape[2] / video.shape[1] < target_ratio:
        target_height = int(video.shape[2] / target_ratio)
        target_width = video.shape[2]
    else:
        target_height = video.shape[1]
        target_width = video.shape[2]
    h_crop = (video.shape[1] - target_height) // 2
    w_crop = (video.shape[2] - target_width) // 2
    video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
    video = rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, LAM_INPUT_HW, mode="bilinear")
    video = rearrange(video, "c t h w -> t h w c")
    return video
