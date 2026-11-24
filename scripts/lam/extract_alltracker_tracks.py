#!/usr/bin/env python3
"""Extract a fixed-grid dense trajectory cache with the frozen AllTracker.

Run this script in the existing AllTracker environment.  It writes a compact,
pickle-free NPZ and never modifies the source video or AllTracker repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from preprocessing import (
    crop_and_resize_four_three,
    decode_video_clip,
    metadata_array,
    save_npz_atomic,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alltracker-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--grid-stride", type=int, default=16)
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--inference-iters", type=int, default=4)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate decoding/cropping and write frames metadata without loading the model.",
    )
    return parser.parse_args()


def _load_model(repo: Path, checkpoint: Path, device: torch.device, window_len: int):
    repo = repo.resolve()
    if not (repo / "nets" / "alltracker.py").is_file():
        raise FileNotFoundError(f"invalid AllTracker repository: {repo}")
    sys.path.insert(0, str(repo))
    from nets.alltracker import Net  # noqa: PLC0415

    model = Net(window_len, init_weights=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _sample_tracks(
    frames_rgb: np.ndarray,
    *,
    model,
    device: torch.device,
    grid_stride: int,
    inference_iters: int,
    window_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(frames_rgb) < 3:
        raise ValueError("AllTracker clip extraction requires at least 3 frames")
    height, width = frames_rgb.shape[1:3]
    if height % 8 or width % 8:
        raise ValueError(f"tracking height/width must be divisible by 8, got {height}x{width}")
    if grid_stride <= 0:
        raise ValueError("grid_stride must be positive")
    images = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        flows, visconf, _, _ = model.forward_sliding(
            images,
            iters=inference_iters,
            sw=None,
            is_training=False,
            window_len=window_len,
        )
    flows = flows[0].float().cpu()
    visconf = visconf[0].float().cpu()
    if flows.ndim != 4 or flows.shape[:2] != (len(frames_rgb), 2):
        raise RuntimeError(f"unexpected AllTracker flow shape: {tuple(flows.shape)}")
    y = torch.arange(0, height, grid_stride)
    x = torch.arange(0, width, grid_stride)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    anchor = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float()
    sampled_flow = flows[:, :, ::grid_stride, ::grid_stride].permute(0, 2, 3, 1).reshape(len(frames_rgb), -1, 2)
    sampled_visconf = visconf[:, :, ::grid_stride, ::grid_stride].permute(0, 2, 3, 1).reshape(len(frames_rgb), -1, 2)
    tracks = anchor[None] + sampled_flow
    tracks[0] = anchor
    visibility = sampled_visconf[..., 0].clamp(0, 1)
    confidence = sampled_visconf[..., 1].clamp(0, 1)
    visibility[0] = 1.0
    confidence[0] = 1.0
    tracks[..., 0] /= max(width - 1, 1)
    tracks[..., 1] /= max(height - 1, 1)
    return tracks.numpy(), visibility.numpy(), confidence.numpy()


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("height/width must be positive and divisible by 8")
    frames, frame_indices, fps = decode_video_clip(
        args.input_video,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
    )
    original_hw = frames.shape[1:3]
    frames, crop_box = crop_and_resize_four_three(
        frames,
        output_height=args.height,
        output_width=args.width,
    )
    metadata = {
        "backend": "AllTracker",
        "input_video": str(args.input_video.resolve()),
        "fps": fps,
        "original_hw": list(original_hw),
        "crop_box_tl_hw": list(crop_box),
        "tracking_hw": [args.height, args.width],
        "grid_stride": args.grid_stride,
        "query_frame_in_clip": 0,
        "alltracker_repo": str(args.alltracker_repo.resolve()),
    }
    if args.preprocess_only:
        metadata["preprocess_only"] = True
        save_npz_atomic(
            args.output,
            frames_rgb=frames,
            frame_indices=frame_indices,
            metadata_json=metadata_array(metadata),
        )
        print(json.dumps(metadata, indent=2))
        return 0

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"AllTracker checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; run the command in the GPU-enabled environment")
    model = _load_model(args.alltracker_repo, args.checkpoint, device, args.window_len)
    tracks, visibility, confidence = _sample_tracks(
        frames,
        model=model,
        device=device,
        grid_stride=args.grid_stride,
        inference_iters=args.inference_iters,
        window_len=args.window_len,
    )
    metadata.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "inference_iters": args.inference_iters,
            "window_len": args.window_len,
        }
    )
    save_npz_atomic(
        args.output,
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(np.float32),
        confidence=confidence.astype(np.float32),
        frame_indices=frame_indices,
        metadata_json=metadata_array(metadata),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tracks_shape": list(tracks.shape),
                "visibility_mean": float(visibility.mean()),
                "confidence_mean": float(confidence.mean()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

