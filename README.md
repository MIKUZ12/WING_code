# WING: World Action Learning via INteraction-Centric Spectral Latent Guidance

[Project page](https://mikuz12.github.io/wing/) · [Paper](https://arxiv.org/abs/2610.03607) · [Checkpoints](#checkpoints) · [Installation](docs/installation.md) · [Citation](#citation)

WING transfers interaction knowledge from human videos to robot policies.
WING-LAM separates camera and interaction motion, then distills interaction
latents into an RGB-only student. The policy predicts low-frequency temporal
DCT coefficients from current observations, language, and robot state, and uses
them to guide action generation.

The released policy uses **Full-K4 spectral guidance**: an eight-step,
32-dimensional latent trajectory, with all four coefficients K0–K3 retained.

## Release status

This repository includes WING-LAM training and data preparation, robot-policy
training and evaluation for **LIBERO, RoboTwin 2.0, and RoboCasa–GR1**, and
real-robot deployment. Simulator packages, datasets, and pretrained weights are
external inputs.

**Checkpoint download entries below are placeholders.** Stage-II policy training
requires a Stage-I planner checkpoint. The Stage-I human-video training recipe
and dataset are not included. End-to-end reproduction depends on obtaining the
listed assets; installing the code alone does not provide them.

## Results reported in the paper

| Benchmark | Setting | Average success |
| --- | --- | --- |
| LIBERO | Four-suite average | 99.20% |
| RoboTwin 2.0 | Clean and randomized average | 93.80% |
| RoboCasa–GR1 | 24-task tabletop average | 57.7% |

These are paper results, not measurements from the environment checks in this
repository. See the [project page](https://mikuz12.github.io/wing/) for comparisons
and videos. Released checkpoint-to-result mappings are pending.

## Quick start

Run commands from the repository root on Linux. The following creates the policy
environment; simulator and robot environments are configured separately.

```bash
conda env create -f environment.yml
conda activate wing
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train,data,test]'
python -m pip check
python scripts/check_environment.py --video
```

The last command checks imports, the Full-K4 forward/backward path, and CPU video
decoding without downloading weights. A successful check ends with
`Environment check passed.` Check the training/GPU environment separately:

```bash
python scripts/check_environment.py --training --video --require-cuda
```

Follow [installation and troubleshooting](docs/installation.md) for a `venv`
alternative, CUDA/compiler requirements, and model-cache setup. See
[tested environment and resource planning](docs/environment.md) for the scope of
verification and the distributed training settings.

Once checkpoints and data are available:

1. Prepare the assets in the [checkpoint guide](docs/checkpoints.md).
2. Build dataset statistics, text/VAE caches, and WING-LAM targets using the
   matching benchmark guide below.
3. Train or evaluate with the matching task, camera layout, action dimension,
   and normalization statistics.

For example, after completing the [LIBERO preparation](docs/libero.md):

```bash
export PHASE1_SPECTRAL_CHECKPOINT="$PWD/checkpoints/wing_stage1.pt"
export ACTION_DIT_CHECKPOINT="$PWD/artifacts/checkpoints/action_dit_libero.pt"
NUM_GPUS=8 bash scripts/train.sh task=libero \
  data.train.base_dataset.vae_latent_cache_required=true \
  output_dir=artifacts/runs/libero_full_k4
```

## Checkpoints

The download fields are explicit publication placeholders, not working URLs.
Replace each `TODO` with its public download link when the asset is released.

| Asset | Download placeholder | Used for |
| --- | --- | --- |
| WING Stage-I planner | `TODO: WING_STAGE1_URL` | Initializing Stage-II policy training |
| WING-LAM teacher | `TODO: WING_LAM_TEACHER_URL` | Student distillation and motion decoding |
| WING-LAM RGB student | `TODO: WING_LAM_STUDENT_URL` | Extracting spectral training targets |
| WING LIBERO policy bundle | `TODO: WING_LIBERO_URL` | LIBERO evaluation |
| WING RoboTwin policy bundle | `TODO: WING_ROBOTWIN_URL` | RoboTwin evaluation |
| WING RoboCasa policy bundle | `TODO: WING_ROBOCASA_URL` | RoboCasa evaluation |
| WING real-robot policy bundle | `TODO: WING_REAL_ROBOT_URL` | Matching hardware deployment |

Each policy bundle must include its resolved training config, dataset
normalization JSON, spectral normalization statistics, and camera/action
specification. The Wan backbone and source LAM checkpoint are separate assets.
See [asset layout and loading requirements](docs/checkpoints.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Conda/venv setup, CUDA, video decoding, troubleshooting |
| [Environment and resources](docs/environment.md) | Verified software, hardware planning, test limits |
| [Checkpoints](docs/checkpoints.md) | Download placeholders, asset layout, metadata requirements |
| [Configuration](docs/configuration.md) | Model, data, and task configuration |
| [Policy training](docs/training.md) | WING-LAM assets, Stage-I initialization, resume |
| [LIBERO](docs/libero.md) | Data preparation, training, evaluation |
| [RoboTwin 2.0](docs/robotwin.md) | Data conversion, caches, training, evaluation |
| [RoboCasa–GR1](docs/robocasa.md) | 24-task preparation, training, remote evaluation |
| [WING-LAM](docs/lam.md) | Tracks/masks, teacher training, RGB distillation |
| [Real robots](docs/real_robot.md) | Policy server, ROS/Piper integration, motion checks |

## Feedback and contributions

Use the repository's Issues tab for bugs and reproduction questions. Include
the command, task/config overrides, package versions, and relevant traceback.
Remove credentials, private paths, and data before sharing logs or configs.
For a fix, submit a pull request describing the behavior change and validation.

## Citation

```bibtex
@misc{liu2026worldactionlearninginteractioncentric,
  title={World Action Learning via Interaction-Centric Spectral Latent Guidance},
  author={Zhiming Liu and Yikun Miao and Ying Chen and Hongrui Yin and Fangqi Zhu and Xiaoyi Pang and Quanxin Shou and Zhengyang Yan and Haodong Wang and Song Guo},
  year={2026},
  eprint={2610.03607},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.03607}
}
```

## License

WING source is released under the [MIT License](LICENSE), with the third-party
exceptions stated there. Embedded source retains its notices; see
[third-party code and dependencies](docs/third_party.md). External datasets,
weights, simulators, and hardware SDKs follow their own terms.
