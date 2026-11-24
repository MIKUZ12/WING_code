# Checkpoints and model assets

## Download status

The entries below are publication placeholders. No WING download URL is claimed
to be available in this snapshot. Replace the `TODO` values with public links
when publishing; until then, supply equivalent assets locally.

| Asset | Download placeholder | Suggested local destination |
| --- | --- | --- |
| Stage-I spectral planner | `TODO: WING_STAGE1_URL` | `checkpoints/wing_stage1.pt` |
| WING-LAM teacher | `TODO: WING_LAM_TEACHER_URL` | `checkpoints/wing_lam.pt` |
| WING-LAM RGB student | `TODO: WING_LAM_STUDENT_URL` | `checkpoints/wing_lam_rgb_student.pt` |
| LIBERO policy bundle | `TODO: WING_LIBERO_URL` | `checkpoints/libero/` |
| RoboTwin policy bundle | `TODO: WING_ROBOTWIN_URL` | `checkpoints/robotwin/` |
| RoboCasa policy bundle | `TODO: WING_ROBOCASA_URL` | `checkpoints/robocasa/` |
| Real-robot policy bundle | `TODO: WING_REAL_ROBOT_URL` | `checkpoints/real_robot/` |
| Source LAM checkpoint | `TODO: SOURCE_LAM_URL` | `checkpoints/LAM_400k.ckpt` |

The external video backbone is
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
Follow [installation](installation.md) to configure its download cache or a local
model directory. AllTracker and EgoHOS assets are needed for the teacher
data-preparation workflow described in [WING-LAM](lam.md).

## Policy bundle contents

Publish these files together for each benchmark/checkpoint:

```text
checkpoints/<benchmark>/
  policy.pt
  config.yaml
  dataset_stats.json
  spectral_stats.pt
```

`config.yaml` must record the trained architecture, camera order/resolution,
action/state dimensions, sampling stride, and spectral settings. Provide the
exact evaluation command, dataset/task order, checkpoint step, and associated
paper result in the asset description. These result mappings remain pending
until the checkpoint bundles are published.

Training and inference load tensor state dictionaries. Module renaming does not
rename learned tensor keys. Saved Hydra targets must reference the installed
`wing` modules. Teacher-target metadata uses `wing_lam_checkpoint`, and student
initialization metadata uses `init_wing_lam_checkpoint`; re-export assets that
use earlier field names before distributing them. Whole-object Python pickle
checkpoints and optimizer-state resumes across different environments are not a
supported migration format.

## Training assets

Stage-II training requires:

1. Stage-I planner weights and a dimension-matched ActionDiT initialization.
2. Dataset normalization JSON, text embeddings, and the selected VAE cache.
3. WING-LAM target caches and their matching Full-K4 normalization statistics.

The [training guide](training.md) shows how to create ActionDiT initialization
from the Stage-I checkpoint. Action dimensions are 7 for LIBERO, 14 for
RoboTwin, and 44 for the supplied RoboCasa recipe. Stage-I human-video training
data and recipes are not bundled, so a Stage-I checkpoint is required to use
the documented Stage-II flow.

## Inference assets

A trained policy still needs its matching normalization files and Wan VAE.
Text conditioning requires either the configured text encoder/tokenizer or
cached embeddings for the exact formatted instruction. Stage-I initialization
weights and offline training target caches are not required by the deployment
client when a complete final policy checkpoint is loaded. See
[artifact relocation](real_robot.md#relocate-inference-artifacts) before moving
a bundle to another machine.

Keep public configs, cache manifests, indices, and checkpoint metadata free of
credentials and private machine paths. Dataset indices can include resolved
video paths; URL encoding a path does not anonymize it. Publish portable paths
or regenerate indices for the distributed data layout.
