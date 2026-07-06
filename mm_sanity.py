"""moving-mass 플랜트 sanity 체크 (회귀용).

검사: ① DOF 인덱스가 mm_model 상수와 일치 ② 의도치 않은 내부 접촉 없음(유령 브레이크
교훈) ③ trail 실측 ④ 무제어 낙하(역진자 확인) ⑤ 주행 시 self-steering 부호.
사용:  python mm_sanity.py
"""
import numpy as np
import mujoco
import mm_model as M

m = M.m
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# ① 인덱스 검증
print("① DOF 인덱스")
jix = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (m.jnt_qposadr[j], m.jnt_dofadr[j])
       for j in range(m.njnt)}
check("nq=11, nv=10, nu=2", (m.nq, m.nv, m.nu) == (11, 10, 2), f"{(m.nq, m.nv, m.nu)}")
check("slide_y qpos/dof", jix["slide_y"] == (M.Q_SLIDE, M.V_SLIDE), f"{jix['slide_y']}")
check("steer qpos/dof",   jix["steer"] == (M.Q_STEER, M.V_STEER), f"{jix['steer']}")
check("rear/front dof",   (jix["rear_spin"][1], jix["front_spin"][1]) == (M.V_REAR, M.V_FRONT))

# ② 접촉: 직립 정지 + 200스텝 후 — ground↔wheel 만 허용
print("② 접촉 (유령 브레이크 검사)")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
def contact_pairs(d):
    return {tuple(sorted((gname(d.contact[i].geom1), gname(d.contact[i].geom2))))
            for i in range(d.ncon)}
ok_pairs = {("front_geom", "ground"), ("ground", "rear_geom")}
p0 = contact_pairs(d)
for _ in range(200):
    d.ctrl[:] = 0; mujoco.mj_step(m, d)
p1 = contact_pairs(d)
bad = (p0 | p1) - ok_pairs
check("내부 접촉 없음", not bad, f"위반: {bad}" if bad else f"{sorted(p0 | p1)}")

# ③ trail
print("③ trail")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "steer")
anc, ax = d.xanchor[j], d.xaxis[j]
x_g = anc[0] - anc[2] * ax[0] / ax[2]
wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "front_wheel")
trail = (x_g - d.xpos[wid][0]) * 1000
check("trail 60~90mm", 60 < trail < 90, f"{trail:.1f}mm")

# 물리 파라미터 출력 (mm_lqr 교차검증용)
Mtot = float(m.body_mass[1:].sum())
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
h = float(sum(m.body_mass[i] * d.xipos[i][2] for i in range(1, m.nbody)) / Mtot)
sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mass_slider")
print(f"   M_tot={Mtot:.2f}kg  h_CoM={h:.3f}m  slider h={d.xipos[sb][2]:.3f}m "
      f"m={m.body_mass[sb]:.1f}kg")

# ④ 무제어 낙하 (2° 섭동)
print("④ 무제어 낙하")
def passive(deg, v0=0.0, T=5000, rec=None):
    d = mujoco.MjData(m)
    a = np.radians(deg) / 2; d.qpos[3:7] = [np.cos(a), np.sin(a), 0, 0]
    d.qvel[0] = v0; d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = v0 / M.WHEEL_R
    mujoco.mj_forward(m, d)
    for k in range(T):
        d.ctrl[:] = 0; mujoco.mj_step(m, d)
        if rec is not None and k < 200:
            q = d.qpos; w, x, y, z = q[3], q[4], q[5], q[6]
            rec.append((-(2 * (y * z - w * x)), d.qpos[M.Q_STEER]))
        if 1 - 2 * (d.qpos[4]**2 + d.qpos[5]**2) < 0.7:
            return k
    return T
t_fall = passive(2.0) * M.DT
check("역진자 (2°→낙하 <3s)", 0.3 < t_fall < 3.0, f"{t_fall:.2f}s")

# ⑤ self-steering: v0=1.5, lean+(오른쪽) → steer−(오른쪽, 기운 쪽) 이어야 함
print("⑤ self-steering (free fork, v0=1.5)")
rec = []
passive(2.0, v0=1.5, rec=rec)
lean150, steer150 = rec[150]
check("steer가 기운 쪽(lean+·steer−)", lean150 * steer150 < 0,
      f"lean={lean150:+.3f} steer={np.degrees(steer150):+.1f}°")

print("\n" + ("SANITY FAIL: " + ", ".join(FAIL) if FAIL else "SANITY ALL OK"))
raise SystemExit(1 if FAIL else 0)
