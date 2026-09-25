#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""ACT policy NPU inference — 2-camera QVGA, single vmfb (load-once / infer-many).

The trained 2-camera QVGA ACT policy as ONE Torq vmfb. The exported graph wraps
`policy.model` directly, so normalization is NOT in the graph — this module reproduces
lerobot's MEAN_STD normalization in numpy (constants baked into norm_params.npz), runs the
single vmfb on the NPU, and unnormalizes the action chunk to physical joint degrees.

    image_wrist[1,3,240,320] f32  ┐
    image_top  [1,3,240,320] f32  ├─ model_full.vmfb (NPU) ─→ action[1,100,6] f32 (normalized)
    state      [1,6]         f32  ┘
    host: unnormalize → action*a_std + a_mean → joint targets (deg)

Inputs are pre-normalized on host: img = (rgb/255 - img_mean)/img_std (ImageNet, per channel),
NCHW; state = (state - st_mean)/st_std. Camera frames arrive already rotated (CAM_ROTATION, as
recorded) and RGB from the lerobot OpenCV camera.

Run inside the board venv (lerobot + torq-runtime; BOARD_PY in board_config.sh).
"""
import os
import numpy as np
from torq.runtime import VMFBInferenceRunner

HERE = os.path.dirname(os.path.abspath(__file__))
H = 100  # ACT action-chunk horizon
IN_H, IN_W = 240, 320
# SO-101 motor order; matches observation.state / action order in the trained policy
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class ACTPipeline2Cam:
    """Single-vmfb, 2-camera ACT inference. Construct once, call infer_raw() per tick."""

    def __init__(self, vmfb=None, params=None):
        vmfb = vmfb or os.path.join(HERE, "model_full.vmfb")
        params = params or os.path.join(HERE, "norm_params.npz")
        p = np.load(params)
        self.img_mean = p["img_mean"].astype(np.float32).reshape(3, 1, 1)
        self.img_std = p["img_std"].astype(np.float32).reshape(3, 1, 1)
        self.st_mean = p["st_mean"].astype(np.float32).ravel()
        self.st_std = p["st_std"].astype(np.float32).ravel()
        self.a_mean = p["a_mean"].astype(np.float32).ravel()
        self.a_std = p["a_std"].astype(np.float32).ravel()
        self.D = int(p["D"])
        # With the runner's defaults (torq-runtime 2.2.0), this model fails with "failed to
        # writeXram()" (xram_write: accessing memory out of bounds). The cpu allocator avoids it at
        # the same speed; torq-run-module is unaffected.
        self.runner = VMFBInferenceRunner(vmfb, function="main_graph",
                                          runtime_flags=["--torq_device_allocator=cpu"])
        self.last_infer_ms = 0.0

    # ---- preprocessing --------------------------------------------------------------------
    def preprocess_image(self, rgb):
        """rgb: HxWx3 (or [1,H,W,3]) uint8/float 0..255, already RGB + rotated -> f32 NCHW [1,3,240,320]."""
        x = np.asarray(rgb, np.float32)
        if x.ndim == 4:
            x = x[0]
        if x.shape[:2] != (IN_H, IN_W):
            import cv2
            x = cv2.resize(x, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = np.transpose(x, (2, 0, 1)) / 255.0            # CHW, 0..1
        x = (x - self.img_mean) / self.img_std             # ImageNet per-channel
        return np.ascontiguousarray(x[None].astype(np.float32))

    def preprocess_state(self, state):
        st = np.asarray(state, np.float32).reshape(1, self.D)
        return np.ascontiguousarray(((st - self.st_mean) / self.st_std).astype(np.float32))

    # ---- inference ------------------------------------------------------------------------
    def infer(self, img_wrist_nchw, img_top_nchw, state_norm):
        out = self.runner.infer([np.ascontiguousarray(img_wrist_nchw, np.float32),
                                 np.ascontiguousarray(img_top_nchw, np.float32),
                                 np.ascontiguousarray(state_norm, np.float32)])[0]
        self.last_infer_ms = getattr(self.runner, "infer_time_ms", 0.0)
        act = np.asarray(out, np.float32).reshape(H, self.D)
        return (act * self.a_std + self.a_mean).astype(np.float32)   # -> physical deg

    def infer_raw(self, rgb_wrist, rgb_top, state_raw):
        """Live obs (two raw camera frames + raw robot state) -> physical action chunk [100,D] (deg)."""
        return self.infer(self.preprocess_image(rgb_wrist),
                          self.preprocess_image(rgb_top),
                          self.preprocess_state(state_raw))
