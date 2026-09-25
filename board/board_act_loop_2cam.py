# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""Board deployment of the 2-camera QVGA ACT policy: CPU owns motors+cameras, Torq NPU owns inference.

  * TWO cameras: wrist + top, 320x240 MJPG, RGB, rotated per CAM_ROTATION (from board_config.sh).
  * ONE vmfb (model_full.vmfb) via ACTPipeline2Cam (numpy MEAN_STD normalization baked in).
  * The follower calibration file you made on the host, loaded through lerobot's normal path.

Faithful ACT execution (equivalent to `lerobot-rollout --strategy.type=base`): a MOTOR thread
owns all robot IO at the control rate and replays the active [100,6] chunk step-by-step; an
INFERENCE thread continuously re-plans on the NPU from the latest observation and swaps in the
newer chunk (receding horizon). Overlapping chunks are blended (LeRobot's async-chunking scheme).

Safety (real arm under model control):
  * DRY-RUN by default: targets computed/logged, NO send_action. Pass --live to move.
  * range clamp to the training envelope (a_mean +/- ENV_SIGMA*a_std).
  * initial ramp + per-step jump cap (pathology guard).
  * Ctrl-C / exit -> disconnect -> torque released.

Run (on the board), after `source ./board_config.sh` so cameras/motor/calib are set:
  "$BOARD_PY" board_act_loop_2cam.py [--live] [--fps 30] [--duration 60] [--vmax 5]
  # cameras / rotation / motor port / calibration default to the board_config.sh env values;
  # override per-run with --cam-wrist / --cam-top / --cam-rotation / --motor-port / --calib.
"""
import argparse, threading, time, statistics, os
from pathlib import Path
import numpy as np
import cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Rotation
from act_pipeline_2cam import ACTPipeline2Cam, JOINT_ORDER

# Rig-specific values default to board_config.sh (env); override with CLI flags below.
CAM_WRIST = os.environ.get("CAM_WRIST", "/dev/v4l/by-id/usb-YOUR_WRIST_CAMERA-video-index0")
CAM_TOP = os.environ.get("CAM_TOP", "/dev/v4l/by-id/usb-YOUR_TOP_CAMERA-video-index0")
CAM_ROTATION = int(os.environ.get("CAM_ROTATION", "0"))   # 0/90/180/270, same as when recording
MOTOR_PORT = os.environ.get("MOTOR_PORT", "/dev/ttyFOLLOWER")
BOARD_CALIB = os.environ.get("BOARD_CALIB", "/home/root/follower_host.json")
H = 100          # chunk horizon
BLEND_NEW = 0.7  # overlap blend weight for the newer chunk (LeRobot async default)


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.obs_wrist = None
        self.obs_top = None
        self.obs_state = None
        self.obs_step = 0
        self.new_chunks = []
        self.any_chunk = False
        self.running = True
        self.infer_ms = []


def inference_thread(pipe, sh):
    while sh.running:
        with sh.lock:
            w = None if sh.obs_wrist is None else sh.obs_wrist.copy()
            t = None if sh.obs_top is None else sh.obs_top.copy()
            state = None if sh.obs_state is None else sh.obs_state.copy()
            t_plan = sh.obs_step
        if w is None or t is None or state is None:
            time.sleep(0.01)
            continue
        t0 = time.perf_counter()
        chunk = pipe.infer_raw(w, t, state)            # [100,6] physical deg (NPU)
        dt = (time.perf_counter() - t0) * 1e3
        with sh.lock:
            sh.new_chunks.append((t_plan, chunk))
            sh.any_chunk = True
            sh.infer_ms.append(dt)


def cam_keys(obs):
    """Map the two camera obs keys to (wrist, top) by name."""
    imgs = [k for k in obs if not k.endswith(".pos")]
    w = next((k for k in imgs if "wrist" in k.lower()), None)
    t = next((k for k in imgs if "top" in k.lower()), None)
    if w is None or t is None:  # fall back to declared order
        w, t = imgs[0], imgs[1]
    return w, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually command the motors (default: dry-run)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--vmax", type=float, default=5.0, help="safety jump cap: max deg/step vs previous command (default 5; try 2.5 for a gentler first run)")
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds to ease from start pose into the first target")
    ap.add_argument("--env-sigma", type=float, default=4.0, help="range clamp = a_mean +/- sigma*a_std")
    ap.add_argument("--calib", default=BOARD_CALIB, help="follower calibration JSON on the board (default from board_config.sh BOARD_CALIB)")
    ap.add_argument("--cam-wrist", default=CAM_WRIST, help="wrist camera device (default from board_config.sh CAM_WRIST)")
    ap.add_argument("--cam-top", default=CAM_TOP, help="top camera device (default from board_config.sh CAM_TOP)")
    ap.add_argument("--cam-rotation", type=int, default=CAM_ROTATION, choices=[0, 90, 180, 270], help="rotate both camera frames, as when recording (default from board_config.sh CAM_ROTATION)")
    ap.add_argument("--motor-port", default=MOTOR_PORT, help="follower motor serial port (default from board_config.sh MOTOR_PORT)")
    ap.add_argument("--stream-port", type=int, default=0, help="if >0, serve a live MJPEG of the model-input camera views on this port")
    args = ap.parse_args()

    streamer = None
    if args.stream_port:
        from mjpeg_stream import Streamer
        streamer = Streamer(args.stream_port).start()
        print(f"[loop] MJPEG stream on board :{args.stream_port}  -> open http://<board-ip>:{args.stream_port}", flush=True)
    mode = "LIVE (motors WILL move)" if args.live else "DRY-RUN (no motion; torque holds position)"
    print(f"[loop] mode: {mode}  fps={args.fps} dur={args.duration}s  blend(new={BLEND_NEW})  "
          f"jump<={args.vmax}deg/step  ramp={args.ramp}s  env-sigma={args.env_sigma}")
    print(f"[loop] calib: {args.calib}")

    rot = {0: Cv2Rotation.NO_ROTATION, 90: Cv2Rotation.ROTATE_90, 180: Cv2Rotation.ROTATE_180, 270: Cv2Rotation.ROTATE_270}[args.cam_rotation]
    cam_w = OpenCVCameraConfig(index_or_path=args.cam_wrist, fps=30, width=320, height=240, fourcc="MJPG", rotation=rot)
    cam_t = OpenCVCameraConfig(index_or_path=args.cam_top, fps=30, width=320, height=240, fourcc="MJPG", rotation=rot)
    # id + calibration_dir make lerobot load <calibration_dir>/<id>.json itself and verify the
    # motors' stored calibration matches it on connect (prompting to rewrite it if not).
    calib = Path(args.calib).resolve()
    cfg = SO101FollowerConfig(port=args.motor_port, cameras={"wrist": cam_w, "top": cam_t},
                              use_degrees=True, disable_torque_on_disconnect=True,
                              id=calib.stem, calibration_dir=calib.parent)
    robot = SO101Follower(cfg)
    if not robot.calibration:
        raise SystemExit(f"[loop] calibration file not found: {args.calib}")
    print("[loop] connecting...", flush=True)
    robot.connect()
    pipe = ACTPipeline2Cam()
    a_mean, a_std = pipe.a_mean, pipe.a_std
    lo, hi = a_mean - args.env_sigma * a_std, a_mean + args.env_sigma * a_std
    print("[loop] connected + NPU pipeline loaded. warming up...", flush=True)
    for _ in range(5):
        robot.get_observation()

    sh = Shared()
    obs = robot.get_observation()
    KW, KT = cam_keys(obs)
    print(f"[loop] camera keys: wrist={KW!r} top={KT!r}", flush=True)
    with sh.lock:
        sh.obs_state = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], np.float32)
        sh.obs_wrist = np.asarray(obs[KW])
        sh.obs_top = np.asarray(obs[KT])

    infer = threading.Thread(target=inference_thread, args=(pipe, sh), daemon=True)
    infer.start()

    print("[loop] waiting for first NPU chunk...", flush=True)
    while not sh.any_chunk and sh.running:
        time.sleep(0.02)

    budget = 1.0 / args.fps
    step_ms, clamp_hits, jump_hits, faithful = [], 0, 0, 0
    n_chunks_used, covered, gaps = 0, 0, 0
    ramp_steps = max(1, int(args.ramp * args.fps))
    cmd_prev = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], np.float32)
    action_buf = {}
    step_no = 0
    t_start = time.perf_counter()
    t_end = t_start + args.duration
    print(f"[loop] running for {args.duration}s ...", flush=True)
    try:
        while time.perf_counter() < t_end and sh.running:
            t0 = time.perf_counter()
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"[loop] motor comms lost ({type(e).__name__}); stopping cleanly (unplugged?)")
                break
            state = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], np.float32)
            with sh.lock:
                sh.obs_state = state
                sh.obs_wrist = np.asarray(obs[KW])
                sh.obs_top = np.asarray(obs[KT])
                sh.obs_step = step_no
                pending = sh.new_chunks
                sh.new_chunks = []
                nss_ms = statistics.mean(sh.infer_ms[-5:]) if sh.infer_ms else 0.0
            for t_plan, chunk in pending:
                n_chunks_used += 1
                for i in range(H):
                    ts = t_plan + i
                    if ts < step_no:
                        continue
                    a = chunk[i].astype(np.float32)
                    if ts not in action_buf:
                        action_buf[ts] = a
                    else:
                        action_buf[ts] = BLEND_NEW * a + (1.0 - BLEND_NEW) * action_buf[ts]
            if step_no in action_buf:
                target = action_buf[step_no]
                covered += 1
            else:
                target = cmd_prev
                gaps += 1
            action_buf.pop(step_no - 1, None)
            clamped = np.clip(target, lo, hi)
            if np.any(clamped != target):
                clamp_hits += 1
            target = clamped
            if step_no < ramp_steps:
                limit = max(0.5, args.vmax / ramp_steps * (step_no + 1))
            else:
                limit = args.vmax
            delta = target - cmd_prev
            capped = np.clip(delta, -limit, limit)
            if np.any(np.abs(delta) > limit + 1e-6):
                jump_hits += 1
            else:
                faithful += 1
            cmd = cmd_prev + capped
            if args.live:
                try:
                    robot.send_action({f"{j}.pos": float(cmd[i]) for i, j in enumerate(JOINT_ORDER)})
                except Exception as e:
                    print(f"[loop] send_action failed ({type(e).__name__}); stopping cleanly (unplugged?)")
                    break
            cmd_prev = cmd
            if streamer is not None:
                streamer.set_status(
                    mode="LIVE" if args.live else "DRY-RUN",
                    fps=(step_no + 1) / max(time.perf_counter() - t_start, 1e-6),
                    nss_ms=nss_ms,
                    state=[float(x) for x in state], target=[float(x) for x in cmd])
                if step_no % 2 == 0:   # ~15 fps stream; keeps motor tick light
                    streamer.update("wrist", cv2.cvtColor(np.ascontiguousarray(obs[KW]), cv2.COLOR_RGB2BGR))
                    streamer.update("top", cv2.cvtColor(np.ascontiguousarray(obs[KT]), cv2.COLOR_RGB2BGR))
            step_no += 1
            dt = time.perf_counter() - t0
            step_ms.append(dt * 1e3)
            sleep = budget - dt
            if sleep > 0:
                time.sleep(sleep)
    finally:
        sh.running = False
        infer.join(timeout=2.5)
        try:
            robot.disconnect()
            print("[loop] disconnected (torque released).")
        except Exception as e:
            print(f"[loop] WARN: clean disconnect failed ({type(e).__name__}: {e}); motors may be energized.")

    with sh.lock:
        ims = list(sh.infer_ms)
    n = len(step_ms)
    print(f"\n[loop] SUMMARY ({'LIVE' if args.live else 'DRY-RUN'})")
    print(f"  control steps: {n}  target {args.fps} fps  chunks executed: {n_chunks_used}")
    if step_ms:
        print(f"  step work: mean {statistics.mean(step_ms):.1f} ms  max {max(step_ms):.1f} ms")
    if ims:
        print(f"  NPU inferences: {len(ims)}  mean {statistics.mean(ims):.0f} ms  min {min(ims):.0f}  max {max(ims):.0f}")
        print(f"  replans/sec: {len(ims)/args.duration:.2f}")
    print(f"  planned-covered steps: {covered}/{n}   hold-gaps: {gaps}/{n}")
    print(f"  range-clamp hits: {clamp_hits}/{n}   safety-jump caps: {jump_hits}/{n}   faithful: {faithful}/{n}")
    print(f"  last commanded target (deg): {np.round(cmd_prev, 2)}")


if __name__ == "__main__":
    main()
