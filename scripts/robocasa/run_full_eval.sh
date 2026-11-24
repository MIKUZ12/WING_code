#!/usr/bin/env bash
# Official protocol: 24 tasks x 50 episodes, one simulator env per worker.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WING_PYTHON="${WING_PYTHON:-python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-python}"
CONFIG="${CONFIG:-${ROOT}/configs/sim_robocasa.yaml}"
CKPT="${ROBOCASA_EVAL_CKPT:?Set ROBOCASA_EVAL_CKPT}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/evaluate_results/robocasa_full_k4}"
EPISODES="${EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SERVICES_PER_GPU="${SERVICES_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-18880}"
ROBOCASA_SOURCE="${ROBOCASA_SOURCE:-${ROOT}/third_party/robocasa-gr1-tabletop-tasks}"
DRY_RUN="${DRY_RUN:-0}"

TASKS=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ -f "${CONFIG}" ]] || { printf 'Missing config: %s\n' "${CONFIG}" >&2; exit 2; }
[[ -s "${CKPT}" || "${DRY_RUN}" == 1 ]] || { printf 'Missing checkpoint: %s\n' "${CKPT}" >&2; exit 2; }
[[ "${SERVICES_PER_GPU}" =~ ^[1-3]$ ]] || { printf 'SERVICES_PER_GPU must be 1..3.\n' >&2; exit 2; }
TOTAL_WORKERS=$(( ${#GPUS[@]} * SERVICES_PER_GPU ))
(( TOTAL_WORKERS > 24 )) && TOTAL_WORKERS=24

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'gpus=%s workers=%d tasks=24 episodes=%s max_steps=%s replan=%s ckpt=%s\n' \
    "${GPU_IDS}" "${TOTAL_WORKERS}" "${EPISODES}" "${MAX_EPISODE_STEPS}" \
    "${N_ACTION_STEPS}" "${CKPT}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
printf 'task_index\tgpu\tenv_id\tepisodes\tsuccesses\texit_code\tlog\n' >"${OUTPUT_DIR}/summary.tsv"
printf 'episodes_per_task=%s\nmax_episode_steps=%s\nn_action_steps=%s\ngpu_ids=%s\nservices_per_gpu=%s\ncheckpoint=%s\n' \
  "${EPISODES}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" "${GPU_IDS}" \
  "${SERVICES_PER_GPU}" "${CKPT}" >"${OUTPUT_DIR}/settings.txt"

worker() (
  set -u
  worker_id="$1"; gpu="$2"; port=$((BASE_PORT + worker_id))
  worker_dir="${OUTPUT_DIR}/worker_${worker_id}_gpu_${gpu}"
  server_log="${worker_dir}/policy_server.log"
  worker_summary="${worker_dir}/summary.tsv"
  server_pid=''
  mkdir -p "${worker_dir}/numba_cache" "${worker_dir}/matplotlib"
  printf 'task_index\tgpu\tenv_id\tepisodes\tsuccesses\texit_code\tlog\n' >"${worker_summary}"
  trap '[[ -n "${server_pid}" ]] && kill "${server_pid}" 2>/dev/null || true' EXIT INT TERM

  env CUDA_VISIBLE_DEVICES="${gpu}" DIFFSYNTH_SKIP_DOWNLOAD=true \
    PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/src:${ROOT}" \
    "${WING_PYTHON}" scripts/deploy_real_bot/serve_wing_policy.py \
    --config "${CONFIG}" --host 127.0.0.1 --port "${port}" --authkey wing \
    >"${server_log}" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 360); do
    grep -q 'Listening on' "${server_log}" && { ready=1; break; }
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 5
  done
  (( ready == 1 )) || { printf 'Policy server failed: %s\n' "${server_log}" >&2; exit 1; }

  failures=0
  for ((task_index=worker_id; task_index<24; task_index+=TOTAL_WORKERS)); do
    env_id="${TASKS[task_index]}"
    task_log="${worker_dir}/task_$(printf '%02d' "${task_index}").log"
    set +e
    /usr/bin/timeout --signal=INT --kill-after=120s 172800s env \
      CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      MUJOCO_EGL_DEVICE_ID=0 NUMBA_CACHE_DIR="${worker_dir}/numba_cache" \
      MPLCONFIGDIR="${worker_dir}/matplotlib" \
      PYTHONPATH="${ROOT}/src:${ROOT}:${ROBOCASA_SOURCE}" \
      "${ROBOCASA_PYTHON}" scripts/robocasa/eval_remote.py \
      --env-id "${env_id}" --host 127.0.0.1 --port "${port}" \
      --episodes "${EPISODES}" --max-steps "${MAX_EPISODE_STEPS}" \
      --action-horizon "${N_ACTION_STEPS}" --seed 0 >"${task_log}" 2>&1
    rc=$?
    set -e
    summary=$(grep -E '^\{"episodes":' "${task_log}" | tail -n 1 || true)
    successes=$(printf '%s\n' "${summary}" | sed -n 's/.*"successes": \([0-9][0-9]*\).*/\1/p')
    [[ -n "${successes}" ]] || successes=NA
    printf '%d\t%s\t%s\t%s\t%s\t%d\t%s\n' "${task_index}" "${gpu}" \
      "${env_id}" "${EPISODES}" "${successes}" "${rc}" "${task_log}" >>"${worker_summary}"
    (( rc == 0 )) || failures=$((failures + 1))
  done
  exit "${failures}"
)

cd "${ROOT}"
pids=()
for ((worker_id=0; worker_id<TOTAL_WORKERS; worker_id++)); do
  gpu="${GPUS[worker_id % ${#GPUS[@]}]}"
  worker "${worker_id}" "${gpu}" >"${OUTPUT_DIR}/worker_${worker_id}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
overall=0
for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
for f in "${OUTPUT_DIR}"/worker_*/summary.tsv; do tail -n +2 "${f}"; done | \
  sort -t $'\t' -k1,1n >>"${OUTPUT_DIR}/summary.tsv"
exit "${overall}"
