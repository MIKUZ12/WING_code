#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${NUM_GPUS:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'NUM_GPUS must be a positive integer.\n' >&2
  exit 2
}
[[ -f "${ACCELERATE_CONFIG}" ]] || {
  printf 'Missing Accelerate config: %s\n' "${ACCELERATE_CONFIG}" >&2
  exit 2
}

export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "${PYTHON_BIN:-python}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  scripts/train.py "$@"
