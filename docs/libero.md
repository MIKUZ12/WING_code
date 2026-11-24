# LIBERO

Install WING editable in the LIBERO environment (`python -m pip install -e .`)
and install the upstream LIBERO simulator before running evaluation.

## Data and caches

Provide four LeRobot v2.1 roots in the order spatial, object, goal, and
LIBERO-10. The config derives them from `LIBERO_V21_ROOT`.

```bash
export LIBERO_V21_ROOT="$PWD/datasets/libero_v2.1"
export LIBERO_DATASET_STATS="$LIBERO_V21_ROOT/dataset_stats.json"
export LIBERO_TEXT_CACHE="$LIBERO_V21_ROOT/text_embeds"
export LIBERO_VAE_CACHE="$LIBERO_V21_ROOT/vae_full_k4"
export LAM_CACHE_ROOT="$LIBERO_V21_ROOT/latent_full_k4"
export SPECTRAL_STATS_PATH="$LIBERO_V21_ROOT/latent_full_k4_stats.pt"
export WING_LAM_STUDENT="$PWD/outputs/wing_lam_rgb_student.pt"
```

Build text, VAE, WING-LAM target, and normalization caches:

```bash
python scripts/data_process/precompute_dataset_stats.py \
  --data libero_lam --out "$LIBERO_DATASET_STATS"
python scripts/precompute_text_embeds.py task=libero_vae_preencode
torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  task=libero_vae_preencode +preencode_cache_dir="$LIBERO_VAE_CACHE"
python scripts/data_process/extract_lam_pairs.py \
  --dataset "$LIBERO_V21_ROOT/libero_spatial_no_noops_lerobot" \
  --dataset "$LIBERO_V21_ROOT/libero_object_no_noops_lerobot" \
  --dataset "$LIBERO_V21_ROOT/libero_goal_no_noops_lerobot" \
  --dataset "$LIBERO_V21_ROOT/libero_10_no_noops_lerobot" \
  --output "$LAM_CACHE_ROOT" --student "$WING_LAM_STUDENT" \
  --lam "$LAM_CHECKPOINT" --camera-key observation.images.image \
  --pair-gap 4 --horizon 8 --device cuda:0
python scripts/data_process/finalize_lam_cache.py \
  --cache-root "$LAM_CACHE_ROOT" --expected-workers 1
python scripts/data_process/fit_spectral_stats.py \
  --cache-root "$LAM_CACHE_ROOT" --horizon 8 --target-dim 32 \
  --low-frequencies 4 --episode-split all --output "$SPECTRAL_STATS_PATH"
```

## Train

```bash
export PHASE1_SPECTRAL_CHECKPOINT="$PWD/checkpoints/wing_stage1.pt"
export ACTION_DIT_CHECKPOINT="$PWD/artifacts/checkpoints/action_dit_libero.pt"
NUM_GPUS=8 bash scripts/train.sh \
  task=libero \
  data.train.base_dataset.vae_latent_cache_required=true \
  output_dir=artifacts/runs/libero_full_k4
```

## Evaluate

Install LIBERO separately and point its runtime config at the checkout:

```bash
export LIBERO_ROOT="$PWD/external/LIBERO"
python experiments/libero/prepare_runtime_config.py \
  --libero-root "$LIBERO_ROOT" --output-dir outputs/libero_runtime
export LIBERO_CONFIG_PATH="$PWD/outputs/libero_runtime"
python experiments/libero/eval_libero_single.py \
  task=libero \
  ckpt="$LIBERO_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$LIBERO_DATASET_STATS" \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0

python experiments/libero/run_libero_manager.py \
  task=libero \
  ckpt="$LIBERO_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$LIBERO_DATASET_STATS" \
  MULTIRUN.num_gpus=8
```
