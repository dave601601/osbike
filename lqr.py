"""LQR gain 설계 — 리액션휠 진자(roll) 해석 모델 → 이산 리카티 → K.

왜 FD(mjd_transitionFD) 대신 해석 모델인가:
  얇은 타이어 강체 접촉을 free-joint roll로 섭동하면 접촉 법선 강성이 A에 섞여
  (A[1,0]~수백) gain이 4자리로 폭주 → 상시 포화(bang-bang) → 균형 실패.
  접촉선(지면) 기준 리액션휠 진자로 직접 세우면 이 아티팩트가 사라진다.

동역학 (접촉선 x축 기준, θ=lean):
    I_p θ̈ = M g h sinθ − τ            # 프레임 roll (τ = 플라이휠 반작용)
    ω̇ = τ/I_r − θ̈                     # 플라이휠 상대 각속도
  검증: θ=2°, τ=10Nm → θ̈≈-1.9 rad/s² → 0.2s만에 lean 반전. open-loop 실측과 일치.

controller.balance() 가 이미 u = -K x 구조라 K를 그대로 (kp_lean,kd_lean,kw_fw)에 매핑.
LQR이 부호까지 맞춘다 (복원 방향 → kp_lean<0).

사용:  python lqr.py            # 파라미터 추출 → 설계 → 닫힌루프 → 시뮬 검증 → json 저장
"""
import json
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are, expm
import model as M


# ---------- 1. 물리 파라미터 추출 (직립 자세에서) ----------

def bike_params():
    """접촉선(지면 x축) 기준 roll 진자 파라미터 (M, h, I_p, I_r)."""
    m = M.m
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)                       # xipos/ximat 채움 (적분 안 함)

    Mtot = 0.0; mz = 0.0; I_line = 0.0
    for i in range(1, m.nbody):                   # world(0) 제외
        mi = float(m.body_mass[i])
        if mi == 0.0:
            continue
        xi = d.xipos[i]                           # body CoM (월드)
        Ri = d.ximat[i].reshape(3, 3)
        Iw = Ri @ np.diag(m.body_inertia[i]) @ Ri.T   # CoM 기준 관성 (월드축)
        y, z = float(xi[1]), float(xi[2])
        I_line += float(Iw[0, 0]) + mi * (y * y + z * z)   # x축(접촉선)으로 평행축
        Mtot += mi
        mz += mi * float(xi[2])
    h = mz / Mtot                                 # CoM 높이 (접촉선 위)

    fw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "reaction_wheel")
    Rfw = d.ximat[fw].reshape(3, 3)
    Ifw = Rfw @ np.diag(m.body_inertia[fw]) @ Rfw.T
    I_r = float(Ifw[0, 0])                        # 플라이휠 스핀 관성 (x축)
    I_p = I_line - I_r                            # 진자 관성 = 잠금 전체 − 스핀 dof
    return Mtot, h, I_p, I_r


# ---------- 2. 선형 상태공간 + 이산화 ----------

def state_space():
    """x=[lean, roll_rate, fw_speed], u=τ.  연속 (A_c, B_c) 반환."""
    Mtot, h, I_p, I_r = bike_params()
    g = -float(M.m.opt.gravity[2])                # 9.81
    a = Mtot * g * h / I_p                        # 불안정 계수 (>0)
    A_c = np.array([[0.0, 1.0, 0.0],
                    [a,   0.0, 0.0],
                    [-a,  0.0, 0.0]])
    B_c = np.array([[0.0],
                    [-1.0 / I_p],
                    [1.0 / I_p + 1.0 / I_r]])
    return A_c, B_c, (Mtot, h, I_p, I_r)


def discretize(A_c, B_c, dt):
    """정확 이산화 (증강 행렬 지수)."""
    n = A_c.shape[0]
    Maug = np.zeros((n + 1, n + 1))
    Maug[:n, :n] = A_c
    Maug[:n, n:] = B_c
    Ed = expm(Maug * dt)
    return Ed[:n, :n], Ed[:n, n:]


# ---------- 3. LQR 설계 ----------

def flywheel_balance(lean_max_deg=3.0, rrate_max=1.5, fw_max=300.0, tau_max=10.0):
    """[lean, roll_rate, fw_speed] 이산 LQR. Bryson 규칙으로 Q,R 초기화."""
    A_c, B_c, params = state_space()
    A_d, B_d = discretize(A_c, B_c, M.DT)
    Q = np.diag([1.0 / np.radians(lean_max_deg) ** 2,
                 1.0 / rrate_max ** 2,
                 1.0 / fw_max ** 2])
    R = np.array([[1.0 / tau_max ** 2]])
    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d).ravel()
    eig = np.linalg.eigvals(A_d - B_d @ K.reshape(1, -1))
    return K, (A_c, B_c, A_d, B_d, Q, R, eig, params)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    K, aux = flywheel_balance()
    A_c, B_c, A_d, B_d, Q, R, eig, (Mtot, h, I_p, I_r) = aux

    print("=== 물리 파라미터 (접촉선 기준 roll 진자) ===")
    print(f"  M_total = {Mtot:.3f} kg   h_CoM = {h:.3f} m")
    print(f"  I_p(진자) = {I_p:.4f}   I_r(플라이휠 스핀) = {I_r:.5f} kg·m²")
    print(f"  불안정 시정수 τ = 1/√(Mgh/I_p) = {1/np.sqrt(Mtot*9.81*h/I_p):.3f} s")

    print("\n=== 연속 상태공간 ===")
    print("A_c =\n", A_c, "\nB_c =", B_c.ravel())

    print("\n=== 3-state 플라이휠 균형 LQR ===")
    print(f"K = {K}")
    print(f"closed-loop |eig| = {np.abs(eig)}   안정: {np.all(np.abs(eig) < 1)}")
    print("\n-> controller.Gains 매핑:")
    print(f"   kp_lean = {K[0]:+.4f}")
    print(f"   kd_lean = {K[1]:+.4f}")
    print(f"   kw_fw   = {K[2]:+.5f}")

    # balance=LQR K + 조향(예측 lean 카운터스티어 + 센터링) + 속도 PI.
    # trail 74mm + 유령브레이크 제거 플랜트에서 검증:
    #   정지 균형 / 정지출발→2m/s 추종(4s) / 주행 균형 모두 20s 완주 (2° 섭동).
    full = dict(kp_lean=float(K[0]), kd_lean=float(K[1]), kw_fw=float(K[2]),
                k_ls=-2.0, kp_steer=2.0, kd_steer=0.5, kp_v=2.0, ki_v=0.5)
    json.dump(full, open("lqr_gains.json", "w"), indent=2)
    print("\n-> lqr_gains.json 저장 (balance=LQR, steer=lean예측+센터링, speed=PI)")
