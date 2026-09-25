#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""Extract the ACT policy's normalization constants into a portable npz for the board.

The exported ONNX/vmfb wraps `policy.model` directly, so it contains no input normalization or
output unnormalization; those live in the lerobot pre/post-processor safetensors. The board runs
a numpy-only pipeline (no torch), so the constants are baked into `norm_params.npz`:

    img_mean[3,1,1], img_std[3,1,1]   image MEAN_STD (applied to rgb/255, per channel; ImageNet)
    st_mean[D], st_std[D]             observation.state MEAN_STD (motor order)
    a_mean[D], a_std[D]               action MEAN_STD (unnormalize model output)
    D                                 action/state dim

    normalize:    x_img = (rgb/255 - img_mean)/img_std ;  x_st = (st - st_mean)/st_std
    unnormalize:  a_phys = a_norm * a_std + a_mean

Usage: python export_norm_params.py MODEL_DIR OUT.npz
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMG_KEYS = ("observation.images.wrist", "observation.images.top")


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    # locate the normalizer step in the preprocessor pipeline by name (index varies per lerobot version)
    steps = json.load(open(os.path.join(model_dir, "policy_preprocessor.json")))["steps"]
    idx = next(i for i, s in enumerate(steps) if s["registry_name"] == "normalizer_processor")
    pre = load_file(os.path.join(model_dir, f"policy_preprocessor_step_{idx}_normalizer_processor.safetensors"))

    img_mean = np.asarray(pre[f"{IMG_KEYS[0]}.mean"], np.float32).reshape(3, 1, 1)
    img_std = np.asarray(pre[f"{IMG_KEYS[0]}.std"], np.float32).reshape(3, 1, 1)
    for k in IMG_KEYS[1:]:  # both cameras share one set of image stats (ImageNet)
        assert np.allclose(img_mean, np.asarray(pre[f"{k}.mean"], np.float32).reshape(3, 1, 1)), \
            f"{k} image stats differ from {IMG_KEYS[0]}; handle per-camera stats"
    st_mean = np.asarray(pre[f"{STATE_KEY}.mean"], np.float32).ravel()
    st_std = np.asarray(pre[f"{STATE_KEY}.std"], np.float32).ravel()
    a_mean = np.asarray(pre[f"{ACTION_KEY}.mean"], np.float32).ravel()
    a_std = np.asarray(pre[f"{ACTION_KEY}.std"], np.float32).ravel()

    np.savez(out, img_mean=img_mean, img_std=img_std, st_mean=st_mean, st_std=st_std,
             a_mean=a_mean, a_std=a_std, D=np.int64(len(a_mean)))
    print(f"wrote {out}")
    print("  img_mean", img_mean.ravel(), "img_std", img_std.ravel())
    print("  st_mean ", st_mean, "\n  st_std  ", st_std)
    print("  a_mean  ", a_mean, "\n  a_std   ", a_std)


if __name__ == "__main__":
    main()
