# Environment and resource planning

## Verified software and scope

The repository checks were run on Linux x86_64 with the following software.
This records the review environment, not a claim about the hardware used for
the paper's experiments.

| Component | Observed version |
| --- | --- |
| Python | 3.11.13 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| TorchCodec | 0.5 |
| PyAV | 16.0.1 |
| FFmpeg | 4.4.2 |
| NumPy | 1.26.4 |
| Accelerate | 1.12.0 |
| DeepSpeed package | 0.18.5 |
| Transformers | 4.49.0 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |

Core CPU spectral tests, statistics regression checks, configuration composition,
CLI import/help checks, and a PyAV-to-TorchCodec video round trip passed during
the release review. This was an existing shared environment; a fresh Conda
solve/install has not yet been validated. Its system-wide `pip check` also
reported an unsupported-platform error for the unrelated Rtree package, which
is not a WING dependency.

DeepSpeed package metadata was available, but importing it failed in the
CPU-only review container with a process-lookup error. Training environment
checks therefore did not pass. Package presence does not establish that
DeepSpeed CUDA extensions compile or execute. Re-run the training check in the
clean target GPU environment.

CUDA devices were unavailable to the review process; full GPU training,
policy inference, simulator rollouts, and real-robot motion were not validated
in that environment. GPU model, driver, minimum VRAM, peak RAM, and throughput
measurements remain **TBD**.

Run these checks on the target machine after following [installation](installation.md):

```bash
python -m pip check
python scripts/check_environment.py --video
python scripts/check_environment.py --training --video --require-cuda
```

The environment checker reports local package versions and exercises the
requested components. It is a setup smoke check, not an end-to-end benchmark.

## Training settings in the supplied recipes

The benchmark guides show eight-GPU launches. `scripts/train.sh` itself defaults
to one process unless `NUM_GPUS` is set; it does not automatically select eight
GPUs. The default distributed backend is DeepSpeed ZeRO-1 with no CPU offload.

| Task | GPUs in documented launch | Batch per GPU | Accumulation | Effective batch at 8 GPUs | Workers per process |
| --- | --- | --- | --- | --- | --- |
| LIBERO | 8 | 16 | 1 | 128 | 8 |
| RoboTwin | 8 | 16 | 1 | 128 | 4 |
| RoboCasa | 8 | 4 | 2 | 64 | 4 |

All three use BF16, 32 action steps, nine sampled video frames, and Full-K4
guidance. Action dimensions are 7, 14, and 44 respectively. Multiple cameras and
their composed image resolution affect memory usage; see the data configs.
Changing GPU count, batch size, or accumulation changes the training setup and
must be recorded for reproduction.

## Hardware planning

| Workload | Plan for | Measurement status |
| --- | --- | --- |
| Environment and core checks | CPU host; no model checkpoints needed | CPU checks verified |
| Target, text, and VAE preprocessing | CUDA GPU, video decode workers, fast local storage | Peak memory and throughput TBD |
| Policy training | Multi-GPU CUDA server; the documented recipes use 8 GPUs | GPU model and minimum VRAM TBD |
| Policy inference | A CUDA GPU with BF16 support; benchmark VRAM on the selected camera layout | Minimum VRAM and latency TBD |
| Simulation | Upstream simulator dependencies and rendering support, plus policy inference resources | Simulator revision and rollout validation TBD |
| Real robots | GPU policy server and a separately configured ROS/Piper host | Hardware-specific validation required |

For provisioning, start from a server with **80 GB-class GPUs, at least 128 GB
host RAM, and local NVMe storage** for full policy training. These are
conservative planning recommendations, not measured minimum requirements or a
guarantee that every configuration fits. With eight processes the supplied
loaders can create 32–64 worker processes, so budget CPU cores, RAM, and shared
memory accordingly.

Precompute VAE/text caches before long runs and profile a short training run
with the actual data. On smaller hardware, reduce per-GPU batch size and adjust
accumulation deliberately; confirm memory use and update timing before starting
the full schedule. The checked-in DeepSpeed config does not enable CPU offload.

## Storage

Provision space separately for source datasets, public model components,
WING-LAM targets, text/VAE caches, and training checkpoints. A single total disk
requirement is not specified because dataset size and cache representation vary.
VAE caches can exceed the compressed source videos. Use a small representative
subset to estimate cache size before processing the full dataset.

Full resume states include optimizer state and use more storage than policy
weight exports. Use `checkpoint_keep_last` and `checkpoint_state_keep_last` to
control retention, and retain the normalization/config assets required for each
published policy.
