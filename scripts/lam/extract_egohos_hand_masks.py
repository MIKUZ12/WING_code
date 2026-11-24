#!/usr/bin/env python3
"""Extract required left/right hand masks with the frozen EgoHOS model.

This script belongs in a dedicated EgoHOS/mmseg environment.  It uses only the
three-class two-hand model: label 0 is background and labels 1/2 are left/right
hand.  Interacted-object modules are intentionally not required by the main
WING-LAM supervision path.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from preprocessing import (
    crop_and_resize_four_three,
    decode_video_clip,
    metadata_array,
    save_npz_atomic,
)



def install_mmcv_ops_import_shim() -> None:
    """Allow this ops-free EgoHOS model to run with pure-Python MMCV.

    MMSeg 0.24 imports every registered loss/head at module import time.  A few
    unused modules therefore import ``mmcv.ops`` even though the Swin/UPerNet
    two-hand inference graph needs no compiled MMCV operators.  Official
    ``mmcv-full==1.6`` wheels do not exist for the current H100/PyTorch stack,
    so provide inert symbols solely to let those unused registrations import.
    Any accidental execution still fails loudly instead of silently changing
    model behavior.
    """
    if "mmcv.ops" in sys.modules:
        return

    def unavailable_op(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("this EgoHOS path does not provide compiled mmcv.ops")

    shim = types.ModuleType("mmcv.ops")
    shim.__path__ = []  # type: ignore[attr-defined]
    shim.sigmoid_focal_loss = unavailable_op  # type: ignore[attr-defined]
    shim.CrissCrossAttention = None  # type: ignore[attr-defined]
    shim.PSAMask = None  # type: ignore[attr-defined]
    shim.point_sample = None  # type: ignore[attr-defined]
    sys.modules["mmcv.ops"] = shim


def allow_official_egohos_checkpoint_types() -> None:
    """Allow the legacy NumPy metadata types used by the official checkpoint."""
    import torch  # noqa: PLC0415

    torch.serialization.add_safe_globals(
        [
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--egohos-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    repo = args.egohos_repo.resolve()
    sys.path.insert(0, str(repo))
    install_mmcv_ops_import_shim()
    from mmseg.apis import inference_segmentor, init_segmentor  # noqa: PLC0415

    allow_official_egohos_checkpoint_types()

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
    model = init_segmentor(str(args.config), str(args.checkpoint), device=args.device)
    labels: list[np.ndarray] = []
    for frame_rgb in frames:
        prediction = inference_segmentor(model, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))[0]
        prediction = np.asarray(prediction, dtype=np.uint8)
        if prediction.shape != (args.height, args.width):
            raise RuntimeError(f"unexpected EgoHOS output shape: {prediction.shape}")
        if prediction.max(initial=0) > 2:
            raise RuntimeError(f"two-hand model returned unexpected class id {int(prediction.max())}")
        labels.append(prediction)
    labels_array = np.stack(labels)
    metadata = {
        "backend": "EgoHOS-twohands-ccda",
        "input_video": str(args.input_video.resolve()),
        "fps": fps,
        "original_hw": list(original_hw),
        "crop_box_tl_hw": list(crop_box),
        "mask_hw": [args.height, args.width],
        "classes": {"0": "background", "1": "left_hand", "2": "right_hand"},
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    save_npz_atomic(
        args.output,
        labels=labels_array,
        hand_masks=(labels_array > 0),
        frame_indices=frame_indices,
        metadata_json=metadata_array(metadata),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "labels_shape": list(labels_array.shape),
                "hand_pixel_fraction": float(np.mean(labels_array > 0)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

