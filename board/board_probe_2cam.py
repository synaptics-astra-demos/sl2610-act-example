#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""One-shot seam check on the board — cameras -> NPU vmfb -> action, NO motors, NO motion.

Opens the two UVC cameras directly with cv2 (BGR->RGB, CAM_ROTATION to match training), runs the
single-vmfb ACT pipeline once, and prints action-chunk stats in physical degrees. Validates the
camera + normalization + vmfb seam independently of the motor bus.

Run (on the board), after `source ./board_config.sh` so CAM_WRIST/CAM_TOP are set:
  "$BOARD_PY" board_probe_2cam.py
"""
import os, numpy as np, cv2, time
from act_pipeline_2cam import ACTPipeline2Cam, JOINT_ORDER

# Camera device paths come from board_config.sh (env). Placeholders if unset.
CAM_WRIST = os.environ.get("CAM_WRIST", "/dev/v4l/by-id/usb-YOUR_WRIST_CAMERA-video-index0")
CAM_TOP = os.environ.get("CAM_TOP", "/dev/v4l/by-id/usb-YOUR_TOP_CAMERA-video-index0")
CAM_ROTATION = int(os.environ.get("CAM_ROTATION", "0"))   # 0/90/180/270, same as when recording
_CV2_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def grab(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    for _ in range(5):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"camera read failed: {path}")
    if CAM_ROTATION:
        frame = cv2.rotate(frame, _CV2_ROT[CAM_ROTATION])
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # HxWx3 RGB uint8


def main():
    print("[probe] opening cameras...")
    w = grab(CAM_WRIST)
    t = grab(CAM_TOP)
    print(f"[probe] wrist {w.shape} top {t.shape}")
    state = np.zeros(6, np.float32)   # neutral state (motors not required for a seam check)
    print("[probe] loading vmfb + running inference...")
    pipe = ACTPipeline2Cam()
    t0 = time.perf_counter()
    chunk = pipe.infer_raw(w, t, state)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"[probe] inference {ms:.0f} ms (NPU {pipe.last_infer_ms:.0f} ms)  chunk {chunk.shape}")
    print(f"[probe] action[0]  (deg): {np.round(chunk[0], 2)}")
    print(f"[probe] action[50] (deg): {np.round(chunk[50], 2)}")
    print(f"[probe] per-joint range over chunk (deg):")
    for j, name in enumerate(JOINT_ORDER):
        print(f"    {name:14s} [{chunk[:,j].min():7.2f}, {chunk[:,j].max():7.2f}]")


if __name__ == "__main__":
    main()
