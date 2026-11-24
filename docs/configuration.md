# Configuration layout

The released policy training flow is the robot-policy stage. Select the
benchmark with `task=libero`, `task=robotwin`, or `task=robocasa`; no stage or
grounding suffix is needed in task names. The internal training mode is `policy`.

## Training and evaluation entry points

| Benchmark | Training task | Evaluation config |
| --- | --- | --- |
| LIBERO | `configs/task/libero.yaml` | `configs/sim_libero.yaml` |
| RoboTwin | `configs/task/robotwin.yaml` | `configs/sim_robotwin.yaml` |
| RoboCasa | `configs/task/robocasa.yaml` | `configs/sim_robocasa.yaml` |

Use `bash scripts/train.sh task=robotwin` after preparing the assets described
in the benchmark guide. VAE preprocessing has separate `<benchmark>_vae_preencode`
tasks. Evaluation filenames follow one convention, but their contents match the
respective runners: LIBERO/RoboTwin use Hydra runners, while RoboCasa configures
the policy server used by `scripts/robocasa/run_full_eval.sh`.

## Model configs

- `configs/model/wing.yaml` is the complete released WING model. It inherits
  `wing_joint.yaml` and enables fixed K0-K3 spectral-plan conditioning with an
  eight-step, 32-dimensional latent trajectory, plan loss, and action adapters.
  All three policy-training tasks select this config.
- `configs/model/wing_joint.yaml` contains the shared video/action architecture,
  schedulers, and initialization settings. It is a base for WING and the VAE
  preprocessing tasks, not a second complete policy recipe.

The unused action-only model config has been removed. `wing.yaml` now refers
to the complete released model rather than that former base variant.

## Data configs

`configs/data/robotwin.yaml` reads three cameras and 14-dimensional actions and
state. A sample covers 33 source frames; video frames are sampled every four
frames, yielding nine video frames and 32 action steps. It is used for dataset
statistics and VAE preprocessing and does not need LAM targets.

`configs/data/robotwin_lam.yaml` wraps the same reader with
`wing.datasets.spectral_targets.SpectralTargetDataset`. It adds cached
eight-step, 32-dimensional WING-LAM targets and their validity masks without
changing the camera, action, or state representation. The target extractor uses
a four-frame pair gap. This is the policy-training dataset, not another camera
resolution or robot embodiment.

RoboCasa similarly separates `robocasa.yaml` and `robocasa_lam.yaml`, sharing
settings through `robocasa_common.yaml`. LIBERO uses `libero_lam.yaml`; its
preprocessing scripts unwrap the base reader without loading target caches.

## WING-LAM and existing artifacts

The teacher and dataset modules are `wing.latent_action.lam` and
`wing.latent_action.lam_dataset`. Public names are `WINGLAM`,
`WINGLAMLossWeights`, `WINGLAMPairsDataset`, and `lam_loss`.

Renaming does not change learned tensor keys. Teacher-target metadata uses
`wing_lam_checkpoint`; RGB-student metadata uses `init_wing_lam_checkpoint`.
Re-export assets with earlier teacher metadata field names before using them
with this release; the readers do not translate those field names. Cache readers and
the statistics fitter use the target field names declared in `manifest.json`,
so existing completed caches with explicit target keys can be reused. New
caches and model batches use `latent_*` fields. Use new directories when
regenerating caches instead of mixing differently named shard formats.

Update saved config imports, task names, and environment variables to the new
examples before reuse. RoboCasa's upstream GR1 robot and environment identifiers
are retained because they identify the simulator embodiment, not this method.
