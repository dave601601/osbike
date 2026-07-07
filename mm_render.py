"""moving-mass free-fork 자전거 렌더 (격자 바닥, 추적 카메라).

사용:  python mm_render.py <v_target> <out.mp4> [seconds] [pert_deg]
       MM_YAW=turn30 python mm_render.py 2.0 mm_turn.mp4 20 0.5
MM_YAW: straight(기본) | turn (+40° 완만 선회) | scurve(참고: 실패 케이스)
무게추 조향은 완만해야 안정 — yaw_ref 슬루 3°/s (엔벨로프상 robust clean 영역).
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
FORCE = os.environ.get("MM_FORCE", "")               # 슬라이더 힘한계 override [N]
if FORCE:
    xml = xml.replace('ctrlrange="-20 20"', f'ctrlrange="-{FORCE} {FORCE}"')
rm = mujoco.MjModel.from_xml_string(xml)

base = json.load(open("mm_lqr_gains.json"))
if FORCE:  # 힘에 맞게 balance LQR 재설계 + 컨트롤러 클립도 그 힘으로
    import mm_lqr
    f = float(FORCE)
    stroke = float(rm.jnt_range[mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")][1])
    K, _, _ = mm_lqr.design(rm, y_max=stroke, F_max=f)
    M.CTRL_HI[M.A_SLIDE], M.CTRL_LO[M.A_SLIDE] = f, -f
    G = C.Gains(float(K[0]), float(K[1]), float(K[2]), float(K[3]),
                base['kp_v'], base['ki_v'], base['k_yaw'], base['kd_yaw'], base['lean_max'])
else:
    G = C.Gains(**base)
LEANMAX = os.environ.get("MM_LEANMAX", "")            # heading 최대 lean 명령 [deg]
if LEANMAX:
    G = G._replace(lean_max=np.radians(float(LEANMAX)))
d = mujoco.MjData(rm)
a = np.radians(PERT) / 2
d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
d.qvel[0] = V_TARGET                                # 발사 시작
d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = V_TARGET / M.WHEEL_R
mujoco.mj_forward(rm, d)
integ = 0.0

# 카메라: MM_CAM=chase(측면, 기본) | high(고각) | top(거의 top-down, 회전 잘 보임)
# azimuth 고정(world) 트래킹이라 선회 시 자전거가 화면 안에서 회전 → 고각일수록 체감↑
CAM = os.environ.get("MM_CAM", "chase")
_dist, _az, _elev = {"chase": (3.2, 120, -12),
                     "high":  (5.5, 90, -55),
                     "top":   (7.0, 90, -80)}.get(CAM, (3.2, 120, -12))
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_BODY, "frame")
cam.distance, cam.azimuth, cam.elevation = _dist, _az, _elev

MODE = os.environ.get("MM_YAW", "straight")
TURN_DEG = float(os.environ.get("MM_TURN_DEG", "40"))
def yaw_target(t):
    if MODE in ("turn", "turn30"):
        return np.radians(TURN_DEG) if t >= 2 else 0.
    if MODE == "scurve":          # 참고: 무게추 조향 authority 초과(전복) 데모
        return np.radians(30.) if 2 <= t < 12 else (np.radians(-30.) if t >= 12 else 0.)
    return 0.
SLEW = np.radians(float(os.environ.get("MM_SLEW", "3.0")))   # yaw_ref 슬루 [°/s]
ALLOW_FALL = os.environ.get("MM_ALLOW_FALL", "") != ""       # 낙하해도 인코딩(한계 데모)

renderer = mujoco.Renderer(rm, height=H, width=W)
frames, min_up_z, x0 = [], 1.0, float(d.qpos[0])
yref = 0.0
for step in range(N_STEPS):
    yref += float(np.clip(yaw_target(step * M.DT) - yref, -SLEW * M.DT, SLEW * M.DT))
    ctrl, integ, st = C.controller(d, G, integ, V_TARGET, M.DT, yaw_ref=yref)
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
if min_up_z <= 0.7 and not ALLOW_FALL:
    print("FELL — aborting encode", file=sys.stderr); sys.exit(4)
import imageio
imageio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8)
print(f"OK out={OUT} Δx={x1-x0:+.2f}m min_up_z={min_up_z:.4f}")
