"""정면(종방향) 오르막 등판 테스트 — 실제 경사 plane + ② baseline (LQR+외곽루프).

물리 예측: 등판 필요 추진력 = M g sinφ (10° → 22.5 N). 스톡 드라이브 ±4 Nm 은
바퀴반경 0.3 m 기준 13.3 N 한계 → 10° 정속 등판 불가 (감속 → v_min 붕괴 → 낙하).
필요 토크 ≈ M g sinφ·R ≈ 6.7 Nm + 구름손실 여유. docs 의 "오르막 ~4° 한계"(스톡)와 정합:
sinφ = 13.3/(13.2·9.81) → 5.9° 이론치 − 손실.

경사는 중력 틸트가 아니라 **회전시킨 plane**(euler 0 -φ 0, +x 가 오르막) — 물리 등가지만
렌더에서 실제 경사로 보임. 자전거는 경사면 위에 피치 맞춰 착지, 초기속도는 경사 방향.

사용:  python mm_climb.py                    # tau × v_target 스윕 (10°)
       python mm_climb.py 8 2.0 out.mp4      # tau, v_target 지정 렌더
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys

import numpy as np
import mujoco

import mm_model as M
import mm_lqr
import mm_controller as C

SLOPE_DEG = 10.0
FORCE = 60.0
HORIZON = 30.0
V_MIN = 0.65


def build(slope_deg=SLOPE_DEG, tau=4.0, force=FORCE):
    xml = open(M.XML).read()
    xml = xml.replace('<geom name="ground" type="plane" size="0 0 0.05" '
                      'rgba="0.55 0.55 0.55 1"/>',
                      f'<geom name="ground" type="plane" size="0 0 0.05" '
                      f'euler="0 {-slope_deg} 0" rgba="0.55 0.55 0.55 1"/>')
    xml = xml.replace('ctrlrange="-20 20"', f'ctrlrange="-{force} {force}"')
    xml = xml.replace('ctrlrange="-4 4"', f'ctrlrange="-{tau} {tau}"')
    return mujoco.MjModel.from_xml_string(xml)


def init_pose(m, d, v0, slope_deg=SLOPE_DEG, pert_deg=0.5):
    """경사면 위 착지: 피치 = 경사각(노즈업), 두 바퀴 중심이 면에서 법선 0.30 m."""
    th = np.radians(slope_deg)
    q_pitch = np.array([np.cos(-th / 2), 0.0, np.sin(-th / 2), 0.0])  # R_y(-φ)
    a = np.radians(pert_deg) / 2
    q_roll = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
    qw = np.zeros(4)
    mujoco.mju_mulQuat(qw, q_pitch, q_roll)
    d.qpos[3:7] = qw
    d.qpos[0:3] = [-M.WHEEL_R * np.sin(th), 0.0, M.WHEEL_R * np.cos(th) + 1e-4]
    d.qvel[0:3] = [v0 * np.cos(th), 0.0, v0 * np.sin(th)]
    d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = v0 / M.WHEEL_R
    mujoco.mj_forward(m, d)


def make_gains(m):
    import json
    K, _, _ = mm_lqr.design(m, y_max=0.15, F_max=FORCE)
    b = json.load(open(M.PARAMS / "mm_lqr_gains.json"))
    return C.Gains(*[float(k) for k in K], b["kp_v"], b["ki_v"], b["k_yaw"],
                   b["kd_yaw"], b["lean_max"], b["ki_yaw"], b["k_lat"],
                   b["kd_lat"], b["yaw_corr_max"], b["lat_slew"])


def climb(tau, v_target, slope_deg=SLOPE_DEG, sec=HORIZON, frames_out=None,
          frame_every=8, cam=None, renderer=None):
    m = build(slope_deg, tau)
    M.CTRL_HI[M.A_SLIDE], M.CTRL_LO[M.A_SLIDE] = FORCE, -FORCE
    M.CTRL_HI[M.A_REAR], M.CTRL_LO[M.A_REAR] = tau, -tau
    G = make_gains(m)
    d = mujoco.MjData(m)
    init_pose(m, d, v_target, slope_deg)
    cs, px = C.CtrlState(), 0.0
    cdt = 5 * M.DT
    th = np.radians(slope_deg)
    v_end, fell_t = v_target, None
    for step in range(int(sec / M.DT)):
        if step % 5 == 0:
            ctrl, cs, st = C.controller(d, G, cs, v_target, cdt, yaw_ref=0.0,
                                        path=(px, 0.0, 0.0))
            px += v_target * cdt
            d.ctrl[:] = np.asarray(ctrl)
            v_end = float(st.v_fwd)
        mujoco.mj_step(m, d)
        if frames_out is not None and step % frame_every == 0:
            renderer.update_scene(d, camera=cam)
            frames_out.append(renderer.render())
        w, x, y, z = d.qpos[3:7]
        # up_z 는 피치 포함 → 경사 피치(cos φ≈0.985) 감안해 롤 기준 판정
        if 1 - 2 * (x * x + y * y) < 0.7 * np.cos(th):
            fell_t = step * M.DT
            break
    dist = float(d.qpos[0]) / np.cos(th)
    ok = fell_t is None and v_end >= max(0.8 * v_target, V_MIN)
    return dict(ok=ok, fell_t=fell_t, v_end=round(v_end, 2),
                dist=round(dist, 1), m=m, d=d)


def sweep():
    print(f"=== {SLOPE_DEG:g}° 정면 등판: tau × v_target (30s, 성공=완주+속도유지) ===")
    print(f"{'tau[Nm]':>8} " + "".join(f"{v:>16}" for v in (1.0, 1.5, 2.0, 3.0)))
    for tau in (4.0, 6.0, 7.0, 8.0, 10.0):
        row = []
        for v in (1.0, 1.5, 2.0, 3.0):
            r = climb(tau, v)
            row.append(f"OK v={r['v_end']} {r['dist']}m" if r["ok"] else
                       (f"x{r['fell_t']:.1f}s {r['dist']}m" if r["fell_t"]
                        else f"슬립 v={r['v_end']}"))
        print(f"{tau:>8.0f} " + "".join(f"{c:>16}" for c in row))


def render(tau, v_target, out, sec=20.0):
    m = build(SLOPE_DEG, tau)
    W, H = 720, 480
    xml = open(M.XML).read()   # 격자 재질 렌더용 재빌드
    xml = xml.replace("  <worldbody>",
                      f'  <visual><global offwidth="{W}" offheight="{H}"/>'
                      f'<headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>\n'
                      '  <asset><texture name="grid" type="2d" builtin="checker" '
                      'rgb1="0.20 0.24 0.29" rgb2="0.29 0.34 0.40" width="512" height="512"/>'
                      '<material name="grid" texture="grid" texrepeat="12 12" '
                      'reflectance="0.2"/></asset>\n  <worldbody>', 1)
    xml = xml.replace('<geom name="ground" type="plane" size="0 0 0.05" '
                      'rgba="0.55 0.55 0.55 1"/>',
                      f'<geom name="ground" type="plane" size="0 0 0.05" '
                      f'euler="0 {-SLOPE_DEG} 0" material="grid"/>')
    xml = xml.replace('ctrlrange="-20 20"', f'ctrlrange="-{FORCE} {FORCE}"')
    xml = xml.replace('ctrlrange="-4 4"', f'ctrlrange="-{tau} {tau}"')
    m = mujoco.MjModel.from_xml_string(xml)
    M.CTRL_HI[M.A_SLIDE], M.CTRL_LO[M.A_SLIDE] = FORCE, -FORCE
    M.CTRL_HI[M.A_REAR], M.CTRL_LO[M.A_REAR] = tau, -tau
    G = make_gains(m)
    d = mujoco.MjData(m)
    init_pose(m, d, v_target)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "frame")
    cam.distance, cam.azimuth, cam.elevation = 3.5, 130, -8
    renderer = mujoco.Renderer(m, height=H, width=W)
    cs, px = C.CtrlState(), 0.0
    cdt = 5 * M.DT
    frames, min_up = [], 1.0
    for step in range(int(sec / M.DT)):
        if step % 5 == 0:
            ctrl, cs, st = C.controller(d, G, cs, v_target, cdt, yaw_ref=0.0,
                                        path=(px, 0.0, 0.0))
            px += v_target * cdt
            d.ctrl[:] = np.asarray(ctrl)
        mujoco.mj_step(m, d)
        if step % 8 == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
        w, x, y, z = d.qpos[3:7]
        min_up = min(min_up, 1 - 2 * (x * x + y * y))
    renderer.close()
    th = np.radians(SLOPE_DEG)
    print(f"[render] dist={d.qpos[0]/np.cos(th):.1f}m v_end={d.qvel[0]/np.cos(th):.2f} "
          f"min_up_z={min_up:.3f} (경사 피치 보정 전)")
    import imageio
    imageio.mimwrite(out, frames, fps=30, codec="libx264", quality=8)
    print(f"OK -> {out}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        render(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
    else:
        sweep()
