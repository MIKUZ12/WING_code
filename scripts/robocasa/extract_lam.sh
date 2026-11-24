#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${ROBOCASA_DATA_ROOT:-datasets/robocasa-starvla}"
CACHE_ROOT="${ROBOCASA_LAM_CACHE:-${ROOT}/artifacts/robocasa/latent_gap4_h8}"
STATS_PATH="${ROBOCASA_SPECTRAL_STATS:-${ROOT}/artifacts/robocasa/latent_gap4_h8_dct_k4_stats.pt}"
STUDENT="${ROBOCASA_LAM_STUDENT:?Set ROBOCASA_LAM_STUDENT to the latent action RGB-student checkpoint}"
LAM="${ROBOCASA_LAM_CHECKPOINT:?Set ROBOCASA_LAM_CHECKPOINT to the LAM checkpoint}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
LOG_ROOT="${ROBOCASA_LAM_LOG_ROOT:-${ROOT}/artifacts/robocasa/latent_extract_logs}"

NAMES=(
  gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000
  gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000
)

COMMON=()
for name in "${NAMES[@]}"; do
  dataset="${DATA_ROOT}/${name}"
  [[ -d "${dataset}" ]] || { printf 'Missing dataset: %s\n' "${dataset}" >&2; exit 2; }
  COMMON+=(--dataset "${dataset}")
done
COMMON+=(
  --output "${CACHE_ROOT}" --student "${STUDENT}" --lam "${LAM}"
  --camera-key observation.images.ego_view --pair-gap 4 --horizon 8
  --expected-fps 20 --pair-batch-size "${PAIR_BATCH_SIZE:-64}"
)

mkdir -p "${CACHE_ROOT}" "${LOG_ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "${ROOT}"
"${PYTHON_BIN}" scripts/data_process/extract_lam_pairs.py "${COMMON[@]}" --initialize-only

pids=()
for rank in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "${PYTHON_BIN}" \
    scripts/data_process/extract_lam_pairs.py "${COMMON[@]}" \
    --device cuda:0 --num-shards "${#GPUS[@]}" --shard-index "${rank}" \
    --require-existing-manifest >"${LOG_ROOT}/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for rank in "${!pids[@]}"; do
  wait "${pids[$rank]}" || { printf 'latent action worker %d failed.\n' "${rank}" >&2; failed=1; }
done
(( failed == 0 )) || exit 1

"${PYTHON_BIN}" scripts/data_process/finalize_lam_cache.py \
  --cache-root "${CACHE_ROOT}" --expected-workers "${#GPUS[@]}"
"${PYTHON_BIN}" scripts/data_process/fit_spectral_stats.py \
  --cache-root "${CACHE_ROOT}" --horizon 8 --target-dim 32 --basis dct \
  --low-frequencies 4 --episode-split all --output "${STATS_PATH}"
printf 'latent action cache: %s\nK4 stats: %s\n' "${CACHE_ROOT}" "${STATS_PATH}"
