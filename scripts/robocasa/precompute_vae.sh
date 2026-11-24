#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-${#GPUS[@]}}"
OUTPUT="${ROBOCASA_VAE_CACHE:-${ROOT}/artifacts/robocasa/vae_cache_24task_9f_256}"
TASK_CONFIG="${ROBOCASA_VAE_TASK:-robocasa_vae_preencode}"
LOCAL_TMP="${ROBOCASA_LOCAL_TMP_ROOT:-/tmp/wing_robocasa_${UID}}"
(( ${#GPUS[@]} == NUM_GPUS )) || {
  printf 'GPU_IDS contains %d devices but NUM_GPUS=%d.\n' "${#GPUS[@]}" "${NUM_GPUS}" >&2
  exit 2
}

mkdir -p "${LOCAL_TMP}" "${OUTPUT}"
chmod 700 "${LOCAL_TMP}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${LOCAL_TMP}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WING_VIDEO_RETURN_UINT8="${WING_VIDEO_RETURN_UINT8:-1}"

cd "${ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/precompute_vae_latents.py \
  task="${TASK_CONFIG}" \
  output_dir="${ROBOCASA_VAE_RUN_DIR:-${ROOT}/artifacts/robocasa/vae_precompute_run}" \
  +preencode_cache_dir="${OUTPUT}" \
  +preencode_batch_size="${VAE_BATCH_SIZE:-8}" \
  +preencode_shard_size="${VAE_SHARD_SIZE:-256}" \
  +preencode_num_workers="${VAE_NUM_WORKERS:-8}" \
  +preencode_prefetch_factor="${VAE_PREFETCH_FACTOR:-1}" \
  +preencode_seed="${VAE_SEED:-20260822}" \
  +preencode_log_every_batches="${VAE_LOG_EVERY_BATCHES:-20}" \
  +preencode_overwrite=false "$@"
