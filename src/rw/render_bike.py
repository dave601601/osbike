"""Headless render of the LQR bike to MP4, with a checkerboard floor so motion
is visible, and an optional forward-drive speed loop.

Render-only: injects a checker texture/material into the XML *string* (the
committed XML on disk is untouched). Reuses controller.py / lqr.py exactly.

Usage:
    python render_bike.py <v_target> <out.mp4> [seconds]
    python render_bike.py 2.0 balance_lqr_drive_20s.mp4 20
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import model as M
import controller as C
import lqr

V_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
OUT      = sys.argv[2] if len(sys.argv) > 2 else str(M.ROOT / "balance_lqr_drive_20s.mp4")
SECONDS  = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
N_STEPS  = int(SECONDS / M.DT)
FRAME_EVERY, FPS, W, H = 8, 30, 720, 480

NOCONTROL = os.environ.get("NOCONTROL", "") != ""        # ctrl=0 (passive)
HALFWIDTH = os.environ.get("TIRE_HALFWIDTH", "")         # 타이어 반폭 override [m]

# --- inject checker floor into the XML string (render model only) ---
with open(M.XML) as f:
    xml = f.read()
asset = f"""  <visual>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.24 0.29" rgb2="0.29 0.34 0.40"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.1"/>
  </asset>
"""
xml = xml.replace("  <worldbody>", asset + "  <worldbody>", 1)
xml = xml.replace('rgba="0.55 0.55 0.55 1"/>', 'material="grid"/>', 1)  # ground geom
if HALFWIDTH:
    xml = xml.replace('fromto="0 -0.008 0  0 0.008 0"',
                      f'fromto="0 -{HALFWIDTH} 0  0 {HALFWIDTH} 0"')
rm = mujoco.MjModel.from_xml_string(xml)   # render model (checker floor)

# --- gains: params/lqr_gains.json 단일 소스 (없으면 python src/rw/lqr.py 로 생성) ---
import json
if os.path.exists(M.PARAMS / "lqr_gains.json"):
    G = C.Gains(**json.load(open(M.PARAMS / "lqr_gains.json")))
else:
    K, _ = lqr.flywheel_balance()
    G = C.Gains(float(K[0]), float(K[1]), float(K[2]), -2.0, 2.0, 0.5, 2.0, 0.5)

d = mujoco.MjData(rm)
a = np.radians(2.0) / 2
d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
mujoco.mj_forward(rm, d)
integ = 0.0

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_BODY, "frame")
cam.distance, cam.azimuth, cam.elevation = 3.2, 120, -12

renderer = mujoco.Renderer(rm, height=H, width=W)
frames, min_up_z, x0 = [], 1.0, float(d.qpos[0])
for step in range(N_STEPS):
    ctrl, integ, st = C.controller(d, G, integ, V_TARGET, M.DT)
    d.ctrl[:] = 0.0 if NOCONTROL else np.asarray(ctrl)
    mujoco.mj_step(rm, d)
    min_up_z = min(min_up_z, float(np.asarray(st.up_z)))
    if step % FRAME_EVERY == 0:
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
renderer.close()

x1, vfwd = float(d.qpos[0]), float(d.qvel[0])
print(f"[result] v_target={V_TARGET}  Δx={x1-x0:+.3f} m  final v_fwd={vfwd:+.3f} m/s  "
      f"min_up_z={min_up_z:.4f}  frames={len(frames)}", file=sys.stderr)
if min_up_z <= 0.7 and not NOCONTROL:
    print("FELL — aborting encode", file=sys.stderr); sys.exit(4)

import imageio
imageio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8)
print(f"OK out={OUT} Δx={x1-x0:+.2f}m v_fwd={vfwd:.2f} min_up_z={min_up_z:.4f}")
