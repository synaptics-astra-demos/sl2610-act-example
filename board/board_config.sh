#!/usr/bin/env bash
# ============================================================================
# board_config.sh — EDIT to match YOUR board + robot rig.
#
# Used by:
#   - deploy_to_board.sh (host side): how to reach the board + where to put files
#   - board_probe_2cam.py / board_act_loop_2cam.py (on the board): which cameras,
#     which motor port, which calibration file. Those scripts read these as
#     environment variables, so on the board run `source ./board_config.sh` first
#     (or pass the matching --cam-wrist/--cam-top/--motor-port/--calib flags).
#
# You can override any value from the shell without editing this file.
# ============================================================================

# --- How the host reaches the board ---
# "adb"  = USB (a single adb device, no address needed)
# or an ssh target like "root@192.168.1.50"
: "${BOARD:=adb}"
# Where the deployment lives on the board.
: "${BOARD_DEST:=/home/root/act_run_2cam}"

# --- Follower-arm calibration (the SAME file created when you first ran lerobot-teleoperate in Step 1) ---
# Host copy that gets pushed to the board (lerobot stores it under robots/so_follower/<robot.id>.json):
: "${HOST_CALIB:=$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json}"
# Where it lands on the board (board_act_loop_2cam.py loads it from here):
: "${BOARD_CALIB:=/home/root/follower_host.json}"

# --- Cameras (by-id paths ON THE BOARD; `ls /dev/v4l/by-id/` to find yours) ---
: "${CAM_WRIST:=/dev/v4l/by-id/usb-YOUR_WRIST_CAMERA-video-index0}"
: "${CAM_TOP:=/dev/v4l/by-id/usb-YOUR_TOP_CAMERA-video-index0}"
# Rotation applied to BOTH camera frames before inference (0, 90, 180 or 270).
# Must match what you used when recording the dataset (lerobot `rotation:`); 0 if none.
: "${CAM_ROTATION:=0}"

# --- Motor serial port on the board ---
# Created by ch343_pty_bridge.py (the board has no USB-serial kernel driver, so there
# is no /dev/ttyACM* and lerobot-find-port does not apply on the board).
: "${MOTOR_PORT:=/dev/ttyFOLLOWER}"

# --- On-board Python (see the blog's "Prepare the board" note) ---
# The board venv with lerobot[feetech] + the torq-runtime wheel (numpy + cv2 come with lerobot):
: "${BOARD_PY:=/home/root/act-venv/bin/python}"

export BOARD BOARD_DEST HOST_CALIB BOARD_CALIB CAM_WRIST CAM_TOP CAM_ROTATION MOTOR_PORT BOARD_PY
