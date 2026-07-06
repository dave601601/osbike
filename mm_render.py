"""moving-mass free-fork 자전거 렌더 (격자 바닥, 추적 카메라).

사용:  python mm_render.py <v_target> <out.mp4> [seconds] [pert_deg]
       python mm_render.py 1.0 mm_ride_1ms.mp4 20 0.5
"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mm_model as M
import mm_controller as C

V_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT      = sys.argv[2] if len(sys.argv) > 2 else "/home/bike/bike/mm_ride_1ms.mp4"
SECONDS  = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
PERT     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
N_STEPS  = int(SECONDS / M.DT)
FRAME_EVERY, FPS, W, H = 8, 30, 720, 480

with open(M.XML) as f:
    xml = f.read()
visual = f"""  <visual>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.24 0.29" rgb2="0.29 0.34 0.40"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.1"/>
  </asset>
"""
xml = xml.replace("  <worldbody>", visual + "  <worldbody>", 1)
xml = xml.replace('rgba="0.55 0.55 0.55 1"/>', 'material="grid"/>', 1)
rm = mujoco.MjModel.from_xml_string(xml)

G = C.Gains(**json.load(open("mm_lqr_gains.json")))
d = mujoco.MjData(rm)
a = np.radians(PERT) / 2
d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
d.qvel[0] = V_TARGET                                # 발사 시작
d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = V_TARGET / M.WHEEL_R
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
    d.ctrl[:] = np.asarray(ctrl)
    mujoco.mj_step(rm, d)
    min_up_z = min(min_up_z, float(np.asarray(st.up_z)))
    if step % FRAME_EVERY == 0:
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
renderer.close()

x1 = float(d.qpos[0])
print(f"[result] v_target={V_TARGET} pert={PERT}°  Δx={x1-x0:+.2f}m  "
      f"v_end={float(d.qvel[0]):+.2f}  min_up_z={min_up_z:.4f}", file=sys.stderr)
if min_up_z <= 0.7:
    print("FELL — aborting encode", file=sys.stderr); sys.exit(4)
import imageio
imageio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8)
print(f"OK out={OUT} Δx={x1-x0:+.2f}m min_up_z={min_up_z:.4f}")
