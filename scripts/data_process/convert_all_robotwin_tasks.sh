#!/bin/bash
# Batch-convert all downloaded RoboTwin 2.0 aloha-agilex_randomized_500 tasks -> LeRobot.
# Resumable: skips tasks already converted. Each task -> <OUT>/<task>/part0-4 (5 shards),
# matching the single-task adjust_bottle layout. Deletes the unzip scratch after each task
# to save space (keeps the .zip).
#
# Run after the batch download finishes:
#   bash scripts/data_process/convert_all_robotwin_tasks.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

RAW=data/robotwin2.0/_raw_all/dataset
OUT=data/robotwin2.0/all_tasks_lerobot
UNZ_ROOT=data/robotwin2.0/_raw_all/_unz
mkdir -p "$OUT" "$UNZ_ROOT"

tasks=$(ls "$RAW" 2>/dev/null)
[ -z "$tasks" ] && { echo "no downloaded tasks under $RAW"; exit 1; }

for task in $tasks; do
  ZIP="$RAW/$task/aloha-agilex_randomized_500.zip"
  [ -f "$ZIP" ] || { echo "[skip-nozip] $task"; continue; }
  DST="$OUT/$task"
  if [ -f "$DST/part4/meta/info.json" ]; then echo "[skip-done] $task"; continue; fi

  echo "=== [$task] unzip + convert $(date) ==="
  UNZ="$UNZ_ROOT/$task"
  rm -rf "$UNZ"; mkdir -p "$UNZ"
  unzip -o -q "$ZIP" -d "$UNZ" || { echo "[unzip-fail] $task"; continue; }
  # locate the data/ dir holding episode*.hdf5
  H5=$(find "$UNZ" -name "episode0.hdf5" 2>/dev/null | head -1)
  [ -z "$H5" ] && { echo "[no-hdf5] $task"; rm -rf "$UNZ"; continue; }
  SRC=$(dirname "$H5")
  PROMPT="Complete the manipulation task: ${task//_/ }."

  # 5-shard parallel conversion
  pids=()
  for i in 0 1 2 3 4; do
    S=$((i*100)); E=$((S+100)); [ "$i" -eq 4 ] && E=500
    python scripts/data_process/convert_robotwin_hdf5_to_lerobot.py \
      --src "$SRC" --dst "$DST/part$i" --task-name "$task" \
      --instruction "$PROMPT" --fps 30 --overwrite \
      --episode-start "$S" --episode-stop "$E" \
      > "logs/convert_${task}_part$i.log" 2>&1 &
    pids+=($!)
  done
  ok=1
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  rm -rf "$UNZ"
  if [ "$ok" -eq 1 ]; then echo "[done] $task"; else echo "[CONVERT-FAIL] $task (see logs/convert_${task}_part*.log)"; fi
done
echo "=== ALL DONE $(date) ==="
echo "converted tasks: $(ls "$OUT" 2>/dev/null | wc -l)"

