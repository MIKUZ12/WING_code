# RoboCasa-GR1

Install WING editable in both policy and simulator environments
(`python -m pip install -e .`) after installing the upstream RoboCasa-GR1 stack.

## Data and caches

The retained config uses the official 24-task GR1 dataset order. Set the root
containing those 24 LeRobot directories and the external simulator checkout:

```bash
export ROBOCASA_DATA_ROOT="$PWD/datasets/robocasa"
export ROBOCASA_SOURCE="$PWD/external/robocasa-gr1-tabletop-tasks"
export ROBOCASA_DATASET_STATS="$PWD/artifacts/robocasa/dataset_stats.json"
export ROBOCASA_TEXT_CACHE="$PWD/artifacts/robocasa/text"
export ROBOCASA_VAE_CACHE="$PWD/artifacts/robocasa/vae"
export ROBOCASA_LAM_CACHE="$PWD/artifacts/robocasa/latent_gap4_h8"
export ROBOCASA_SPECTRAL_STATS="$PWD/artifacts/robocasa/latent_k4_stats.pt"
```

```bash
python scripts/data_process/precompute_dataset_stats.py \
  --data robocasa --out "$ROBOCASA_DATASET_STATS"
python scripts/precompute_text_embeds.py task=robocasa_vae_preencode
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/robocasa/precompute_vae.sh
ROBOCASA_LAM_STUDENT="$WING_LAM_STUDENT" \
ROBOCASA_LAM_CHECKPOINT="$LAM_CHECKPOINT" \
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/robocasa/extract_lam.sh
```

## Train

```bash
export PHASE1_CKPT="$PWD/checkpoints/wing_stage1.pt"
export ACTION_DIT_CKPT="$PWD/artifacts/checkpoints/action_dit_robocasa.pt"
export ROBOCASA_RUN_DIR="$PWD/artifacts/robocasa/runs/full_k4"
NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/robocasa/train_full_k4.sh
```

## Evaluate

The evaluation launcher starts the WING policy server supplied by the
deployment release and distributes the 24 tasks over local simulator workers.

```bash
export ROBOCASA_EVAL_CKPT="$ROBOCASA_RUN_DIR/checkpoints/weights/step_100000.pt"
export ROBOCASA_TRAIN_CONFIG="$ROBOCASA_RUN_DIR/config.yaml"
export WING_PYTHON=python
export ROBOCASA_PYTHON=python
GPU_IDS=0,1,2,3,4,5,6,7 SERVICES_PER_GPU=1 \
  bash scripts/robocasa/run_full_eval.sh
```

RoboCasa evaluation additionally requires `configs/sim_robocasa.yaml`,
`scripts/deploy_real_bot/serve_wing_policy.py`, and the released
`wing.deployment.robocasa` module.
