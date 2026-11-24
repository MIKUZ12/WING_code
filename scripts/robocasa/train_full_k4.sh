#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
(( ${#GPUS[@]} == NUM_GPUS )) || {
  printf 'GPU_IDS contains %d devices but NUM_GPUS=%d.\n' "${#GPUS[@]}" "${NUM_GPUS}" >&2
  exit 2
}

: "${PHASE1_CKPT:?Set PHASE1_CKPT}"
: "${ACTION_DIT_CKPT:?Set ACTION_DIT_CKPT}"
export PHASE1_SPECTRAL_CHECKPOINT="${PHASE1_CKPT}"
export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CKPT}"
export ROBOCASA_VAE_CACHE="${ROBOCASA_VAE_CACHE:-${ROOT}/artifacts/robocasa/vae_cache_24task_9f_256}"
export ROBOCASA_LAM_CACHE="${ROBOCASA_LAM_CACHE:-${ROOT}/artifacts/robocasa/latent_gap4_h8}"
export ROBOCASA_SPECTRAL_STATS="${ROBOCASA_SPECTRAL_STATS:-${ROOT}/artifacts/robocasa/latent_gap4_h8_dct_k4_stats.pt}"
export ROBOCASA_TEXT_CACHE="${ROBOCASA_TEXT_CACHE:-${ROOT}/artifacts/robocasa/text_embeds_cache}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AUTO_RESUME="${AUTO_RESUME:-0}"
TRAIN_TASK="${ROBOCASA_TRAIN_TASK:-robocasa}"
OUTPUT_DIR="${ROBOCASA_RUN_DIR:-${ROOT}/artifacts/robocasa/runs/full_k4}"

for required in "${PHASE1_CKPT}" "${ACTION_DIT_CKPT}" \
  "${ROBOCASA_VAE_CACHE}/manifest.json" "${ROBOCASA_LAM_CACHE}/manifest.json" \
  "${ROBOCASA_SPECTRAL_STATS}" "${ROBOCASA_TEXT_CACHE}"; do
  [[ -e "${required}" ]] || { printf 'Missing required path: %s\n' "${required}" >&2; exit 2; }
done

cd "${ROOT}"
exec bash scripts/train.sh \
  task="${TRAIN_TASK}" \
  output_dir="${OUTPUT_DIR}" \
  phase_init.path="${PHASE1_CKPT}" \
  model.action_dit_pretrained_path="${ACTION_DIT_CKPT}" \
  ++data.train.base_dataset.vae_latent_cache_required=true \
  batch_size="${PER_GPU_BATCH:-4}" \
  gradient_accumulation_steps="${GRAD_ACCUM:-2}" \
  num_workers="${NUM_WORKERS:-4}" \
  max_steps="${MAX_STEPS:-100000}" \
  save_every="${SAVE_EVERY:-10000}" \
  checkpoint_keep_last="${CHECKPOINT_KEEP_LAST:-3}" \
  eval_every=0 "$@"
