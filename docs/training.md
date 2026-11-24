# Training WING

WING uses Wan 2.2 TI2V-5B for the video backbone, an ActionDiT initialized from
that backbone, and fixed full-K4 spectral-plan conditioning. The released flow
uses an externally supplied Stage-I planner checkpoint and one Stage-II task
per simulator. Download entries and required files are listed in the
[checkpoint guide](checkpoints.md).

## Common setup

Run commands from the repository root. Set either local model directories or
leave the public model identifier as the default.

```bash
export WAN_MODEL_PATH="$PWD/checkpoints/Wan-AI/Wan2.2-TI2V-5B"
export WAN_TOKENIZER_PATH="$WAN_MODEL_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

`scripts/train.sh` uses Accelerate with ZeRO-1 by default. Override
`ACCELERATE_CONFIG` to use another retained config.

## WING-LAM preparation

Follow `docs/lam.md` for WING-LAM training and RGB-student export. Simulator
cache generation below consumes the resulting RGB-student checkpoint.

Set the source LAM and trained student paths before following a simulator guide:

```bash
export LAM_CHECKPOINT="$PWD/checkpoints/LAM_400k.ckpt"
export WING_LAM_STUDENT="$PWD/outputs/wing_lam_rgb_student.pt"
```

## Stage-I checkpoint

Stage-II training requires an existing WING planner checkpoint. Stage-I human
video training data and recipes are not distributed in this minimal release.
The planner download is currently a publication placeholder. Once weights are
available, set `PHASE1_SPECTRAL_CHECKPOINT` to the local planner file.

```bash
export PHASE1_SPECTRAL_CHECKPOINT="$PWD/checkpoints/wing_stage1.pt"
```

Create one ActionDiT initialization per action dimension from that checkpoint.
LIBERO uses 7, RoboTwin uses 14, and RoboCasa uses 44:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/wing.yaml \
  --dit-checkpoint "$PHASE1_SPECTRAL_CHECKPOINT" \
  --output artifacts/checkpoints/action_dit_libero.pt --action-dim 7
```

Set `PHASE1_SPECTRAL_CHECKPOINT` to the Stage-I weights file and
`ACTION_DIT_CHECKPOINT` to the dimension-matched converted file before
Stage-II training. Repeat the command with `--action-dim 14` or `44` and a
distinct output filename for the other simulators.

Resume a complete run state with `resume=artifacts/runs/<run>/checkpoints/state/step_NNNNNN`
and set `phase_init=null`. Resume keeps optimizer and scheduler state; use
`phase_init.path=...` only for a fresh Stage-II run.
