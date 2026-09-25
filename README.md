<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated. -->

# Scripts for the ACT-on-Astra guide

These are the scripts referenced by the blog: "Autonomous Robotic Arm Operation with Synaptics Astra™ SL2610".
The board address, camera, motor port, and calibration file are fill-in-the-blank in `board/board_config.sh`.

```
scripts/
├── build/
│   ├── onnx_01_export.py       # Step 5: trained lerobot ACT checkpoint -> self-contained fp32 ONNX
│   └── export_norm_params.py   # Step 5: normalization constants (mean/std) -> norm_params.npz
└── board/
    ├── board_config.sh         # EDIT: board address, cameras, rotation, motor port, calibration
    ├── deploy_to_board.sh      # Step 6: push vmfb + params + code + calibration (adb or ssh)
    ├── board_probe_2cam.py     # Step 6: seam check, cameras -> NPU -> action chunk, no motors
    ├── board_act_loop_2cam.py  # Step 6: the control loop (dry-run by default, --live to move)
    ├── act_pipeline_2cam.py    # numpy normalization + single-vmfb inference, used by the two above
    └── mjpeg_stream.py         # optional --stream-port viewer of what the model sees
```

The model-agnostic middle of the pipeline (cleanup, bf16 conversion, ONNX import, `torq-compile`)
is plain `torq-tools` / `torq-compile` commands; the blog lists them inline in Step 5.

## Prerequisites

All three environments use Python 3.12 and uv. Verified with uv 0.12.13, Torq v2.2.0,
`lerobot==0.6.1` and `torq-tools @ d605a4d`. Install the pinned uv on the host and on the board:
```bash
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
source $HOME/.local/bin/env   # put uv on PATH in this shell
```

- **Host, LeRobot env:** runs the two `build/` scripts. The ONNX export also needs `onnx` and
  `onnxscript`, which lerobot does not install:
  ```bash
  uv venv -p 3.12 venv_lerobot && source venv_lerobot/bin/activate
  uv pip install 'lerobot[feetech,core_scripts,training]==0.6.1' 'onnx==1.23.0' 'onnxscript==0.7.2'
  ```
- **Host, Torq env (x86-64 Linux):** the compiler wheel with its `onnx` extra, plus `torq-tools`
  for `torq-cleanup-model` (skipping the cleanup still compiles, but runs ~11% slower on the NPU):
  ```bash
  uv venv -p 3.12 venv_torq && source venv_torq/bin/activate
  uv pip install "torq-compiler[onnx] @ https://github.com/synaptics-torq/torq-compiler/releases/download/v2.2.0/torq_compiler-2.2.0-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "torq-tools @ git+https://github.com/synaptics-torq/torq-tools@d605a4d" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  ```
- **Board:** one venv (`BOARD_PY` in `board_config.sh`) with lerobot and the aarch64 runtime wheel.
  The CPU-only torch index matters: from PyPI, aarch64 torch also pulls in ~15 CUDA packages the
  board can't use. `--index-strategy unsafe-best-match` lets uv take other packages (e.g. `requests`)
  from PyPI even though the torch index also carries old copies of them:
  ```bash
  # --managed-python: the image's own Python lacks stdlib modules we need
  # (fcntl for the motor serial port, statistics for the control loop)
  uv venv -p 3.12 --managed-python /home/root/act-venv
  source /home/root/act-venv/bin/activate
  uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    'lerobot[feetech]==0.6.1' 'pyusb==1.3.1' \
    'https://github.com/synaptics-torq/torq-compiler/releases/download/v2.2.0/torq_runtime-2.2.0-cp312-cp312-manylinux_2_28_aarch64.whl'
  ```
  `pyusb` is for `board/ch343_pty_bridge.py`: the board image has no USB-serial kernel driver, so
  the bridge drives the motor adapter from userspace and exposes it as `/dev/ttyFOLLOWER`.

## Run order

1. **Step 5, build** (commands in the blog): export, cleanup, bf16, import, `torq-compile`
   -> `act_full.vmfb`; `export_norm_params.py` -> `norm_params.npz`.
2. **Step 6, deploy:** edit `board/board_config.sh`, then
   ```bash
   bash board/deploy_to_board.sh act_full.vmfb norm_params.npz
   # then follow the on-board commands it prints (seam check -> dry-run -> --live)
   ```

## Notes

- The motor adapter must be a **CH343** (USB id `1a86:55d3`, a CDC-ACM device), which is what the
  bridge speaks. Check with `cat /sys/bus/usb/devices/*/idProduct` on the board. A CH340 (`7523`)
  uses a vendor protocol the bridge does not implement.
- Start the bridge once per boot, before the probe/loop (the deploy script prints the command).
  Only one process can claim the adapter; to restart it, `kill` the old bridge first.
- On the board, source the config as `source ./board_config.sh` (with the `./`): the default login
  shell is `sh`, which does not look in the current directory for a bare `source board_config.sh`.
- `CAM_ROTATION` in `board_config.sh` must match the `rotation:` you recorded the dataset with
  (0 if none); the board scripts apply it to both cameras before inference.
- `export_norm_params.py` reads the lerobot normalizer step directly and assumes the two cameras are
  named `wrist` and `top` (as in the blog's `lerobot-record` command). Adjust the key names at the top
  of that script if yours differ.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright © 2026 Synaptics Incorporated.

These scripts are original, self-contained implementations; they depend on (but do not copy or
redistribute) [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0), the
[Torq toolchain](https://github.com/synaptics-torq) (Apache-2.0), and other permissively licensed
packages listed in the prerequisites above. The action-chunk blending constant follows LeRobot's
async-inference default (`0.3 * old + 0.7 * new`). The CH343 bridge implements the public
USB CDC-ACM specification.
