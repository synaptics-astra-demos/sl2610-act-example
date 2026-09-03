#!/usr/bin/env bash
# deploy_to_board.sh — push the 2-cam QVGA ACT deployment (bf16 vmfb + norm params +
# board code + your follower calibration) to the board, over adb (USB) or ssh.
#
# Edit ./board_config.sh first.
#
# Usage: bash deploy_to_board.sh act_full.vmfb norm_params.npz
set -euo pipefail
usage="usage: bash deploy_to_board.sh ACT_VMFB NORM_PARAMS_NPZ"
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }   # portable realpath
VMFB="$(abspath "${1:?$usage}")"
NPZ="$(abspath "${2:?$usage}")"
cd "$(dirname "$0")"
source ./board_config.sh

for f in "$VMFB" "$NPZ" "$HOST_CALIB"; do
  [[ -f "$f" ]] || { echo "deploy: '$f' not found (check the arguments and HOST_CALIB in board_config.sh)" >&2; exit 1; }
done

# adb (USB) vs ssh transport, chosen by $BOARD.
if [[ "$BOARD" == "adb" ]]; then
  RUN() { adb shell "$@"; }
  PUSH() { adb push "$1" "$2"; }
else
  RUN() { ssh "$BOARD" "$@"; }
  PUSH() { scp "$1" "$BOARD:$2"; }
fi

echo "[deploy] mkdir $BOARD_DEST on board ($BOARD)"
RUN "mkdir -p $BOARD_DEST"

echo "[deploy] pushing model + params + code + board_config..."
PUSH "$VMFB" "$BOARD_DEST/model_full.vmfb"
PUSH "$NPZ"  "$BOARD_DEST/norm_params.npz"
PUSH act_pipeline_2cam.py   "$BOARD_DEST/"
PUSH board_act_loop_2cam.py "$BOARD_DEST/"
PUSH board_probe_2cam.py    "$BOARD_DEST/"
PUSH mjpeg_stream.py        "$BOARD_DEST/"
PUSH ch343_pty_bridge.py    "$BOARD_DEST/"
PUSH board_config.sh        "$BOARD_DEST/"

echo "[deploy] pushing follower calibration -> $BOARD_CALIB"
PUSH "$HOST_CALIB" "$BOARD_CALIB"

echo "[deploy] done. Contents:"
RUN "ls -l $BOARD_DEST $BOARD_CALIB"

cat <<MSG

Next, on the board:
  cd $BOARD_DEST
  source ./board_config.sh                     # exports cameras / motor / calib
  PY="\$BOARD_PY"

  # motor serial bridge -> \$MOTOR_PORT (once per boot; the board has no USB-serial driver):
  \$PY ch343_pty_bridge.py --link "\$MOTOR_PORT" > /tmp/ch343_bridge.log 2>&1 &
  sleep 2; cat /tmp/ch343_bridge.log               # expect "BRIDGE UP ... /dev/ttyFOLLOWER"

  # 0) seam check (cameras -> NPU, no motors, no motion):
  \$PY board_probe_2cam.py

  # 1) DRY-RUN (motors read, NO motion):
  \$PY board_act_loop_2cam.py --duration 15 --fps 30

  # 2) LIVE (arm moves) — clear the workspace. Default jump cap is 5 deg/step; add --vmax 2.5 for a gentler first run:
  \$PY board_act_loop_2cam.py --live --duration 60 --fps 30
MSG
