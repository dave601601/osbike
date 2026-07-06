"""캐스케이드 제어 — 루프별 순수함수. balance / steer / speed 를 따로 켜고 끌 수 있게 분리.

모든 함수는 순수 JAX: 상태를 인자로 받고 값을 반환. 내부 상태(속도 적분)는 carry로 넘김.
CPU mujoco(MjData)로도 그대로 호출 가능 (viz.py / render_bike.py).

조향 법칙 이력: v1은 yaw-hold(heading 추종)였으나 trail(caster) 플랜트에서 wheel-flop과
상충해 폐기. 현재는 예측 lean 카운터스티어 + 포크 센터링 — 유령브레이크 제거 플랜트에서
정지 균형 / 정지출발→2m/s 추종 / 주행 균형 모두 20s 검증 (docs/progress/plant.md).
"""
import jax.numpy as jnp
from typing import NamedTuple
import model as M

LEAD = 0.25   # lean 예측 시간 [s] ≈ 불안정 시정수 1/√(Mgh/I_p) (lqr.py 출력과 일치)


class Gains(NamedTuple):
    # balance (flywheel) — lqr.flywheel_balance()의 K를 그대로 (부호 포함)
    kp_lean: float; kd_lean: float; kw_fw: float
    # steer: 예측 lean 커플링 + 포크 센터링/감쇠
    k_ls: float;    kp_steer: float; kd_steer: float
    # speed (rear drive) PI
    kp_v: float;    ki_v: float


class St(NamedTuple):
    """dx에서 뽑은 해석 가능한 신호들."""
    lean: float        # 옆으로 기울기 ≈ sin(roll). +면 오른쪽
    up_z: float        # cos(기울기). <0.7 이면 넘어짐 판정
    roll_rate: float   # gyro x
    yaw: float         # heading (월드 x축 기준) — 현재 제어엔 미사용, 기록용
    yaw_rate: float    # gyro z
    y_lat: float       # lateral 위치 (직진 기준선 = x축) — 기록용
    v_fwd: float       # 전진 속도
    fw_speed: float    # 플라이휠 각속도 (포화 감시)
    steer: float       # 조향각
    steer_rate: float  # 조향 각속도


def _quat_to_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_state(dx) -> St:
    """MJX가 velocity 센서를 안 채우므로 qpos/qvel에서 직접 계산."""
    q, v = dx.qpos, dx.qvel
    w, x, y, z = q[3], q[4], q[5], q[6]
    up_y = 2 * (y * z - w * x)              # 프레임 z축의 월드 y성분 = -sin(roll)
    up_z = 1 - 2 * (x * x + y * y)          # = cos(기울기) → 넘어짐 판정
    return St(
        lean       = -up_y,
        up_z       = up_z,
        roll_rate  = v[3],
        yaw        = _quat_to_yaw(q[M.Q_QUAT]),
        yaw_rate   = v[5],
        y_lat      = q[1],
        v_fwd      = v[0],
        fw_speed   = v[M.V_FLYWHEEL],
        steer      = q[M.Q_STEER],
        steer_rate = v[M.V_STEER],
    )


# ---- 루프 3개 (각각 독립) ----

def balance(st: St, g: Gains):
    """리액션 휠로 직립 유지. u = -(K·x), K는 LQR 해 (kw_fw 항이 플라이휠 포화 방지)."""
    tau = -g.kp_lean * st.lean - g.kd_lean * st.roll_rate - g.kw_fw * st.fw_speed
    return jnp.clip(tau, M.CTRL_LO[M.A_FLYWHEEL], M.CTRL_HI[M.A_FLYWHEEL])


def steer(st: St, g: Gains):
    """예측 lean 쪽 카운터스티어 + 포크 센터링/감쇠.
    센터링이 없으면 포크가 조향 스톱에 안착하는 degenerate 해로 감 (v1에서 확인)."""
    lean_pred = st.lean + LEAD * st.roll_rate
    tau = g.k_ls * lean_pred - g.kp_steer * st.steer - g.kd_steer * st.steer_rate
    return jnp.clip(tau, M.CTRL_LO[M.A_STEER], M.CTRL_HI[M.A_STEER])


def speed(st: St, g: Gains, integ, v_target, dt):
    """뒷바퀴 PI로 v_target 추종. 적분항 integ는 carry로 넘겨받고 갱신해서 반환."""
    err = v_target - st.v_fwd
    integ = integ + err * dt
    tau = g.kp_v * err + g.ki_v * integ
    return jnp.clip(tau, M.CTRL_LO[M.A_REAR], M.CTRL_HI[M.A_REAR]), integ


def controller(dx, g: Gains, integ, v_target, dt):
    """3 루프 합쳐 ctrl 벡터 [flywheel, steer, rear] 생성."""
    st = read_state(dx)
    u_fw = balance(st, g)
    u_st = steer(st, g)
    u_dr, integ = speed(st, g, integ, v_target, dt)
    ctrl = jnp.array([u_fw, u_st, u_dr])
    return ctrl, integ, st
