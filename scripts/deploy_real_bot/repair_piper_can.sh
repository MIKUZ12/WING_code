#!/usr/bin/env bash
set -euo pipefail

: "${WING_PY:?Set WING_PY to the Python executable that provides pyAgxArm.}"
: "${ROS_SETUP:?Set ROS_SETUP to the ROS environment setup.bash path.}"
: "${PIPER_WS_SETUP:?Set PIPER_WS_SETUP to the Piper ROS workspace setup.bash path.}"
: "${LEFT_CAN_INTERFACE:?Set LEFT_CAN_INTERFACE to the left-arm CAN interface.}"
: "${RIGHT_CAN_INTERFACE:?Set RIGHT_CAN_INTERFACE to the right-arm CAN interface.}"

BITRATE="${BITRATE:-1000000}"
PYAGX_LD_PRELOAD="${PYAGX_LD_PRELOAD:-}"
devices=("$LEFT_CAN_INTERFACE" "$RIGHT_CAN_INTERFACE")

if [[ -n "${EXTRA_CAN_INTERFACES:-}" ]]; then
  read -r -a extra_devices <<<"$EXTRA_CAN_INTERFACES"
  devices+=("${extra_devices[@]}")
fi

echo "[setup] checking configured Piper CAN interfaces"
for device in "${devices[@]}"; do
  if ! ip link show "$device" >/dev/null 2>&1; then
    echo "[setup] missing interface: $device" >&2
    exit 1
  fi
  ethtool -i "$device" 2>/dev/null | awk -v device="$device" \
    '/bus-info/ {print "  " device " bus-info=" $2}'
done

echo "[setup] resetting CAN interfaces to ${BITRATE} bit/s"
for device in "${devices[@]}"; do
  sudo ip link set "$device" down || true
  sudo ip link set "$device" type can bitrate "$BITRATE" restart-ms 100
  sudo ip link set "$device" txqueuelen 100
  sudo ip link set "$device" up
done

echo "[setup] CAN status"
for device in "${devices[@]}"; do
  ip -details -statistics link show "$device"
done

echo "[setup] passive three-second CAN receive check"
timeout 3 candump -L "${devices[@]}" || true

python_command=("$WING_PY")
if [[ -n "$PYAGX_LD_PRELOAD" ]]; then
  python_command=(env "LD_PRELOAD=$PYAGX_LD_PRELOAD" "$WING_PY")
fi

echo "[setup] querying follower arms without enabling motion"
set +e
LEFT_CAN_INTERFACE="$LEFT_CAN_INTERFACE" \
RIGHT_CAN_INTERFACE="$RIGHT_CAN_INTERFACE" \
BITRATE="$BITRATE" \
"${python_command[@]}" - <<'PY'
import os
import time

from pyAgxArm import AgxArmFactory, create_agx_arm_config


ok = True
for channel in (os.environ["LEFT_CAN_INTERFACE"], os.environ["RIGHT_CAN_INTERFACE"]):
    config = create_agx_arm_config(
        robot="piper",
        comm="can",
        channel=channel,
        bitrate=int(os.environ["BITRATE"]),
    )
    arm = AgxArmFactory.create_arm(config)
    print(f"[pyAgxArm] connecting {channel}")
    arm.connect()
    time.sleep(0.8)
    status = bool(arm.is_ok())
    print(f"[pyAgxArm] {channel} is_ok={status}")
    ok = ok and status
raise SystemExit(0 if ok else 2)
PY
pyagx_status=$?
set -e

# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$PIPER_WS_SETUP"
timeout 8 roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false || true

if [[ "$pyagx_status" -ne 0 ]]; then
  echo "[setup] pyAgxArm could not communicate with both follower arms." >&2
  echo "[setup] Check power, emergency stops, CAN wiring, termination, and interface mapping." >&2
  exit "$pyagx_status"
fi

echo "[setup] pyAgxArm can communicate with both follower arms"
