# Installation

The policy environment targets **Linux x86_64, Python 3.11, PyTorch 2.7.1,
torchvision 0.22.1, and CUDA 12.8 wheels**. Python 3.10 or newer is declared by
the package; release checks were run with Python 3.11.13. See
[verification scope and resources](environment.md).

Run commands from the repository root. Python dependency pins live in
`pyproject.toml`; `environment.yml` supplies Python and FFmpeg without duplicating
the Python package list. Simulation and hardware runtimes are installed separately.

## Conda setup

```bash
conda env create -f environment.yml
conda activate wing
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train,data,test]'
python -m pip check
```

The Conda file selects FFmpeg 4.4, matching the major/minor version used for the
release checks. This is an environment recipe, not a lockfile for every
transitive dependency. The wheel pairing follows the
[official PyTorch 2.7.1 instructions](https://pytorch.org/get-started/previous-versions/#v271).

## venv alternative

Install Python 3.11 and its `venv` module first. On Ubuntu 22.04, install the
system video libraries and build tools with:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential ninja-build
```

Then create the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train,data,test]'
python -m pip check
```

For inference only, `python -m pip install -e .` installs the core package
without training, data-preparation, and test extras.

## Verify the environment

```bash
python scripts/check_environment.py --video
python scripts/check_environment.py --training --video --require-cuda
```

The first command checks imports and the Full-K4 forward/backward path on CPU.
With `--video`, it also writes and decodes a temporary two-frame clip. The second
additionally imports DeepSpeed and requires a usable CUDA device with BF16
support. Requested checks return a nonzero exit code on failure. Neither
command downloads weights or starts training.

### CUDA and training extensions

The host NVIDIA driver must support the CUDA runtime in the PyTorch wheels.
Verify that the environment or container can access the GPU using the second
check above. GPU availability and extension execution were not verified in the
CPU release-review environment.

The PyTorch wheels supply runtime libraries. Building DeepSpeed CUDA extensions
additionally requires a compatible CUDA toolkit (`nvcc`) and a C++ compiler.
Match the toolkit to PyTorch's CUDA build and set `CUDA_HOME` for your
installation. If an optimizer extension fails to compile, inspect the build
log. `optimizer_type=auto` may fall back to PyTorch AdamW; select the optimizer
explicitly when comparing training runs.

Attention uses PyTorch scaled-dot-product attention; a separate FlashAttention
installation is not required.

### Video decoding

Keep `torchcodec==0.5` together with the repository's PyTorch pin. Install FFmpeg
shared libraries as well as the command-line executable. The versioned
[TorchCodec guide](https://github.com/pytorch/torchcodec/blob/v0.5.0/README.md#installing-torchcodec)
describes FFmpeg 4–7 support. The default PyPI TorchCodec wheel decodes on CPU;
CUDA video decoding needs the separate upstream NVDEC setup. Policy CUDA
inference does not require CUDA video decoding.

If TorchCodec cannot load its libraries, check the torch/torchcodec versions,
active environment, and FFmpeg libraries found by the dynamic linker. Run the
`--video` check after fixing the installation to verify actual decoding.

## Model assets

The policy backbone is
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), with a Wan VAE
and UMT5 text encoder/tokenizer.
The configurations describe the public model IDs and expected component files.
The default model cache is `checkpoints/`. Configure downloads or an existing
cache with:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
```

Leave `WAN_MODEL_PATH` unset to use the public model ID. For an existing local
download, point to the model-ID directory:

```bash
export WAN_MODEL_PATH="$PWD/checkpoints/Wan-AI/Wan2.2-TI2V-5B"
export WAN_TOKENIZER_PATH="$WAN_MODEL_PATH"
```

To use previously downloaded components without network downloads, set
`DIFFSYNTH_SKIP_DOWNLOAD=true`; all required files must already exist under the
corresponding model-ID subdirectories. A policy checkpoint alone does not contain
the VAE/text assets needed by every inference configuration. Supply action-DiT
initialization and WING-LAM/plan normalization assets where the training recipe
requires them. See [checkpoints and download placeholders](checkpoints.md).
Neither pretrained weights nor dataset assets are bundled.

## Separate environments

LIBERO, RoboTwin, RoboCasa, and robot SDKs have their own simulator/hardware
dependencies. Follow the matching [LIBERO](libero.md), [RoboTwin](robotwin.md),
[RoboCasa](robocasa.md), or [real-robot](real_robot.md) guide. RoboTwin and
RoboCasa support separate simulator/client and GPU-server environments. The
LIBERO evaluator loads the policy in its simulator process, so that environment
must satisfy both dependency sets. Specific upstream simulator revisions were
not certified by the release checks; record them for each reported evaluation.

Do not expose an unauthenticated policy server to the public internet.

Logging services are optional and disabled in the supplied training defaults.
Paths in the examples are placeholders for files you prepare locally; do not
upload datasets, credentials, hardware serial numbers, or run directories with
the source release.
