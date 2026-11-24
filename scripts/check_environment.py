#!/usr/bin/env python3
"""Check WING imports, Full-K4 execution, and optional video/CUDA support."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_core() -> None:
    import torch

    from wing.models.wan22.spectral_plan import (
        CurrentSpectralPlanPredictor,
        PlanConditionAdapter,
        PlanTokenProjection,
        SpectralPlanNormalizer,
        orthonormal_dct_matrix,
        spectral_plan_alignment_loss,
    )

    torch.set_num_threads(1)
    torch.manual_seed(42)
    dct = orthonormal_dct_matrix(8)
    if not torch.allclose(dct @ dct.T, torch.eye(8), atol=1e-6):
        raise RuntimeError("Temporal DCT basis is not orthonormal.")
    normalizer = SpectralPlanNormalizer(horizon=8, latent_dim=32, low_frequencies=4)
    normalizer.set_statistics(torch.zeros(4, 32), torch.ones(4, 32))
    predictor = CurrentSpectralPlanPredictor(
        visual_dim=12, text_dim=16, plan_dim=24, latent_dim=32,
        num_queries=4, num_heads=4, depth=1,
    )
    projection = PlanTokenProjection(32, 24)
    adapter = PlanConditionAdapter(24, 4, init_gate=0.05)
    prediction = predictor(
        current_visual_tokens=torch.randn(2, 5, 12),
        instruction_tokens=torch.randn(2, 3, 16),
        instruction_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    target = normalizer(torch.randn(2, 8, 32))
    output = adapter(torch.randn(2, 7, 24), projection(prediction))
    loss = output.square().mean() + spectral_plan_alignment_loss(prediction, target)
    loss.backward()
    if prediction.shape != (2, 4, 32) or not torch.isfinite(loss):
        raise RuntimeError("Full-K4 prediction shape or loss is invalid.")
    for gradient in (predictor.out.weight.grad, adapter.gate.grad):
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("Full-K4 backward pass produced an invalid gradient.")


def check_video() -> None:
    import av
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg executable not found; install FFmpeg and its shared libraries.")
    result = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True, timeout=10,
    )
    print(result.stdout.splitlines()[0])
    with tempfile.TemporaryDirectory(prefix="wing_video_") as directory:
        video = Path(directory) / "smoke.mp4"
        with av.open(str(video), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=2)
            stream.width = 32
            stream.height = 32
            stream.pix_fmt = "yuv420p"
            for value in (0, 127):
                frame = av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24",
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        decoder = VideoDecoder(str(video), device="cpu")
        frames = decoder.get_frames_at(indices=[0, 1]).data
        if tuple(frames.shape) != (2, 3, 32, 32):
            raise RuntimeError(f"Unexpected decoded frame shape: {tuple(frames.shape)}.")
        # Release FFmpeg's file handle before removing the temporary directory.
        del frames, decoder


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the driver and container GPU access.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support the BF16 recipe.")
    matrix = torch.ones(4, 4, device="cuda", dtype=torch.bfloat16)
    result = matrix @ matrix
    torch.cuda.synchronize()
    if not bool((result == 4).all()):
        raise RuntimeError("CUDA BF16 matrix multiplication returned an unexpected result.")
    print(f"CUDA runtime: {torch.version.cuda}; visible devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {properties.name}; VRAM: {properties.total_memory / 2**30:.1f} GiB")


def check_training() -> None:
    importlib.import_module("deepspeed")
    for executable in ("nvcc", "c++"):
        if shutil.which(executable) is None:
            print(f"NOTE: {executable} not found; CUDA extension compilation is not verified.")
    print("DeepSpeed import only; optimizer extension compilation and distributed execution are not tested.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true", help="Check PyAV encoding and TorchCodec CPU decoding.")
    parser.add_argument("--training", action="store_true", help="Also import DeepSpeed and inspect build-tool availability.")
    parser.add_argument("--require-cuda", action="store_true", help="Require CUDA and exercise BF16 matrix multiplication.")
    args = parser.parse_args()
    print(f"Platform: {platform.system()} {platform.machine()}; Python: {platform.python_version()}")
    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10 or newer is required.")
        return 1

    packages = [
        ("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
        ("accelerate", "accelerate"), ("transformers", "transformers"),
        ("hydra-core", "hydra"), ("omegaconf", "omegaconf"),
    ]
    if args.video:
        packages += [("av", "av"), ("torchcodec", "torchcodec")]
    if args.training:
        packages += [("deepspeed", None)]
    failures = []

    def run(label, callback):
        try:
            callback()
            print(f"PASS: {label}", flush=True)
        except Exception as exc:
            failures.append(label)
            print(f"FAIL: {label}: {exc}", flush=True)

    def package(distribution, module):
        version = importlib.metadata.version(distribution)
        if module is not None:
            importlib.import_module(module)
        print(f"{distribution}: {version}")

    for distribution, module in packages:
        label = distribution if module is not None else f"{distribution} package metadata"
        run(label, lambda d=distribution, m=module: package(d, m))
    run("Full-K4 CPU forward/backward", check_core)
    if args.video:
        run("two-frame CPU video round trip", check_video)
    if args.training:
        run("DeepSpeed import", check_training)
    if args.require_cuda:
        run("CUDA BF16 execution", check_cuda)
    if failures:
        print(f"Environment check failed ({len(failures)} checks).")
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
