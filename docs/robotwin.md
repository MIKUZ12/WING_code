# RoboTwin 2.0

Install WING editable in the RoboTwin environment (`python -m pip install -e .`)
and complete the upstream RoboTwin simulator installation first.

## Convert and cache data

Install RoboTwin separately, then convert each unpacked task directory to
LeRobot format:

```bash
export ROBOTWIN_ROOT="$PWD/external/RoboTwin"
export ROBOTWIN_DATA_ROOT="$PWD/datasets/robotwin"
python scripts/data_process/convert_robotwin_hdf5_to_lerobot.py \
  --src "$ROBOTWIN_ROOT/dataset/adjust_bottle/demo_randomized/data" \
  --dst "$ROBOTWIN_DATA_ROOT/adjust_bottle_demo_randomized" \
  --task-name adjust_bottle --fps 30
```

Build one JSON list in lexicographic task-directory order. The same list is
used by the data config and target extraction, so dataset indices remain equal:

```bash
export ROBOTWIN_DATASET_DIRS="$(python -c 'import json,os,pathlib; root=pathlib.Path(os.environ["ROBOTWIN_DATA_ROOT"]); print(json.dumps([str(p.resolve()) for p in sorted(root.iterdir()) if (p/"meta/info.json").is_file()]))')"
export ROBOTWIN_DATASET_STATS="$PWD/artifacts/robotwin/dataset_stats.json"
export ROBOTWIN_TEXT_CACHE="$PWD/artifacts/robotwin/text"
export ROBOTWIN_VAE_CACHE="$PWD/artifacts/robotwin/vae"
export ROBOTWIN_LAM_CACHE="$PWD/artifacts/robotwin/latent_gap4_h8"
export ROBOTWIN_SPECTRAL_STATS="$PWD/artifacts/robotwin/latent_k4_stats.pt"

python scripts/data_process/precompute_dataset_stats.py \
  --data robotwin --out "$ROBOTWIN_DATASET_STATS"
python scripts/precompute_text_embeds.py task=robotwin_vae_preencode
torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  task=robotwin_vae_preencode +preencode_cache_dir="$ROBOTWIN_VAE_CACHE"
mapfile -t ROBOTWIN_DATASETS < <(python -c 'import json,os; print(*json.loads(os.environ["ROBOTWIN_DATASET_DIRS"]), sep="\n")')
DATASET_ARGS=()
for dataset in "${ROBOTWIN_DATASETS[@]}"; do DATASET_ARGS+=(--dataset "$dataset"); done
python scripts/data_process/extract_lam_pairs.py "${DATASET_ARGS[@]}" \
  --output "$ROBOTWIN_LAM_CACHE" \
  --student "$WING_LAM_STUDENT" --lam "$LAM_CHECKPOINT" \
  --camera-key observation.images.cam_high --pair-gap 4 --horizon 8 \
  --device cuda:0
python scripts/data_process/finalize_lam_cache.py \
  --cache-root "$ROBOTWIN_LAM_CACHE" --expected-workers 1
python scripts/data_process/fit_spectral_stats.py \
  --cache-root "$ROBOTWIN_LAM_CACHE" --horizon 8 --target-dim 32 \
  --low-frequencies 4 --episode-split all --output "$ROBOTWIN_SPECTRAL_STATS"
```

## Train and evaluate

```bash
export PHASE1_SPECTRAL_CHECKPOINT="$PWD/checkpoints/wing_stage1.pt"
export ACTION_DIT_CHECKPOINT="$PWD/artifacts/checkpoints/action_dit_robotwin.pt"
NUM_GPUS=8 bash scripts/train.sh \
  task=robotwin \
  data.train.base_dataset.vae_latent_cache_required=true \
  output_dir=artifacts/runs/robotwin_full_k4

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin ckpt="$ROBOTWIN_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$ROBOTWIN_DATASET_STATS" \
  EVALUATION.robotwin_root="$ROBOTWIN_ROOT"
```

The manager exposes `experiments/robotwin/wing_policy` to RoboTwin through
`PYTHONPATH`; no files need to be copied into the external simulator checkout.
