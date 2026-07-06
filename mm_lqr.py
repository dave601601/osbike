"""moving-mass 균형 LQR — 접촉선 기준 (roll 진자 + 이동질량) 해석 모델 → 이산 리카티.

동역학 (소각, v≈0, free fork 무시 — 라그랑주). 부호 규약: lean θ+ = 오른쪽 기움,
슬라이더 y+ = 프레임 왼쪽 (joint axis 0 1 0). 그래서:
    [I0    −m·hm] [θ̈]   [M g h θ − m g y]      ← y+(왼쪽) 질량은 복원 토크
    [−m·hm   m ] [ÿ]  = [F − m g θ − c ẏ]      ← 기울면 중력이 추를 기운쪽(−y)으로
  I0=접촉선 기준 roll 관성(슬라이더 중앙 포함), hm=슬라이더 높이, c=조인트 감쇠.
  결합항 −m·hm: 추를 왼쪽(+ÿ)으로 가속하면 반작용이 프레임을 오른쪽으로 먼저 밈
  (비최소위상). 주의: 이 세 부호를 +로 쓰면 LQR이 추를 넘어지는 쪽으로 민다
  (실제로 저지른 실수 — 시뮬에서 passive보다 빨리 넘어지는 증상으로 발각).

FD(mjd_transitionFD)를 안 쓰는 이유: 강체 접촉 강성 아티팩트 (lqr.py와 동일 교훈).

정적 한계 (이론): lean_max = asin(m g d_max / (M g h)) — 기본값(2kg, ±0.15m)이면 ≈2.5°.

사용:  python mm_lqr.py     # 설계 + 고유값 + mm_lqr_gains.json 저장
"""
import json
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are, expm
import mm_model as M


def params(model=None):
    """모델에서 (Mtot, h, I0, m_s, hm, c) 추출. I0=접촉선(지면 x축) 기준 roll 관성."""
    m = model or M.m
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    Mtot = 0.0; mz = 0.0; I0 = 0.0
    for i in range(1, m.nbody):
        mi = float(m.body_mass[i])
        if mi == 0.0:
            continue
        xi = d.xipos[i]
        Ri = d.ximat[i].reshape(3, 3)
        Iw = Ri @ np.diag(m.body_inertia[i]) @ Ri.T
        I0 += float(Iw[0, 0]) + mi * (xi[1]**2 + xi[2]**2)
        Mtot += mi; mz += mi * float(xi[2])
    h = mz / Mtot
    sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mass_slider")
    m_s = float(m.body_mass[sb]); hm = float(d.xipos[sb][2])
    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
    c = float(m.dof_damping[m.jnt_dofadr[sj]])
    return Mtot, h, I0, m_s, hm, c


def state_space(model=None):
    """x=[lean, roll_rate, y, y_dot], u=F  →  연속 (A_c, B_c)."""
    Mtot, h, I0, m_s, hm, c = params(model)
    g = 9.81
    Minv = np.linalg.inv(np.array([[I0, -m_s * hm], [-m_s * hm, m_s]]))
    Mgh, mg = Mtot * g * h, m_s * g
    A = np.zeros((4, 4)); B = np.zeros((4, 1))
    A[0, 1] = 1.0; A[2, 3] = 1.0
    for r, i in ((1, 0), (3, 1)):        # r=상태행, i=Minv 행
        A[r, 0] = Minv[i, 0] * Mgh - Minv[i, 1] * mg
        A[r, 2] = -Minv[i, 0] * mg
        A[r, 3] = -Minv[i, 1] * c
        B[r, 0] = Minv[i, 1]
    return A, B, (Mtot, h, I0, m_s, hm, c)


def design(model=None, lean_max_deg=4.0, rrate_max=0.5, y_max=0.15, ydot_max=1.0,
           F_max=20.0):
    """이산 LQR. Bryson Q/R. 반환 K(4,), 닫힌루프 |eig|."""
    A_c, B_c, p = state_space(model)
    n = 4
    Maug = np.zeros((n + 1, n + 1)); Maug[:n, :n] = A_c; Maug[:n, n:] = B_c
    Ed = expm(Maug * M.DT)
    A_d, B_d = Ed[:n, :n], Ed[:n, n:]
    Q = np.diag([1 / np.radians(lean_max_deg)**2, 1 / rrate_max**2,
                 1 / y_max**2, 1 / ydot_max**2])
    R = np.array([[1 / F_max**2]])
    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d).ravel()
    eig = np.linalg.eigvals(A_d - B_d @ K.reshape(1, -1))
    return K, eig, p


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    K, eig, (Mtot, h, I0, m_s, hm, c) = design()
    g = 9.81
    print("=== 파라미터 ===")
    print(f"  M={Mtot:.2f}kg h={h:.3f}m I0={I0:.3f}kg·m²  slider m={m_s}kg hm={hm:.3f}m c={c}")
    print(f"  불안정 시정수 = {1/np.sqrt(Mtot*g*h/I0):.3f}s")
    print(f"  정적 한계각 = asin(m·g·d/(M·g·h)) = "
          f"{np.degrees(np.arcsin(m_s*0.15/(Mtot*h))):.2f}°  (d=0.15m)")
    print("\n=== 4-state LQR ===")
    print(f"  K = {K}   (u = -(K·x) 관례, mm_controller.balance_mass)")
    print(f"  closed-loop |eig| = {np.abs(eig)}  안정: {np.all(np.abs(eig) < 1)}")
    full = dict(k_lean=float(K[0]), k_rrate=float(K[1]), k_y=float(K[2]),
                k_ydot=float(K[3]), kp_v=2.0, ki_v=0.5)
    json.dump(full, open("mm_lqr_gains.json", "w"), indent=2)
    print("-> mm_lqr_gains.json 저장")
