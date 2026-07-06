"""moving-mass 플랜트 캐스케이드 — balance(무게추 힘) / speed(뒷바퀴 PI). 순수 JAX.

free fork: 조향 입력 없음. steer 는 관측만 (제안서의 인코더 ablation 대응).
balance_mass 는 u = -(K·x) 관례 (K = mm_lqr.design() 결과 그대로).
비최소위상 주의: 무게추를 복원 방향으로 가속하면 반작용이 프레임을 일단 fall 쪽으로
밀었다가 회복 — 결합 4-state 모델(mm_lqr)이 이를 반영하므로 gain 을 손대지 말 것.
"""
import jax.numpy as jnp
from typing import NamedTuple
import mm_model as M


class Gains(NamedTuple):
    # balance (moving mass): x=[lean, roll_rate, y_m, y_m_dot]
    k_lean: float; k_rrate: float; k_y: float; k_ydot: float
    # speed (rear drive) PI
    kp_v: float;   ki_v: float


class St(NamedTuple):
    lean: float        # +면 오른쪽
    up_z: float        # <0.7 넘어짐
    roll_rate: float
    y_m: float         # 무게추 위치 [m]
    ydot: float
    steer: float       # free fork 각 (관측만)
    steer_rate: float
    v_fwd: float
    yaw: float
    y_lat: float


def _quat_to_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_state(dx) -> St:
    q, v = dx.qpos, dx.qvel
    w, x, y, z = q[3], q[4], q[5], q[6]
    return St(
        lean       = -(2 * (y * z - w * x)),
        up_z       = 1 - 2 * (x * x + y * y),
        roll_rate  = v[3],
        y_m        = q[M.Q_SLIDE],
        ydot       = v[M.V_SLIDE],
        steer      = q[M.Q_STEER],
        steer_rate = v[M.V_STEER],
        v_fwd      = v[0],
        yaw        = _quat_to_yaw(q[M.Q_QUAT]),
        y_lat      = q[1],
    )


def balance_mass(st: St, g: Gains):
    """무게추 힘 = -(K·x). lean+(오른쪽) → 무게추를 왼쪽으로 (CoM을 접촉선 왼쪽에)."""
    F = -(g.k_lean * st.lean + g.k_rrate * st.roll_rate
          + g.k_y * st.y_m + g.k_ydot * st.ydot)
    return jnp.clip(F, M.CTRL_LO[M.A_SLIDE], M.CTRL_HI[M.A_SLIDE])


def speed(st: St, g: Gains, integ, v_target, dt):
    err = v_target - st.v_fwd
    integ = integ + err * dt
    tau = g.kp_v * err + g.ki_v * integ
    return jnp.clip(tau, M.CTRL_LO[M.A_REAR], M.CTRL_HI[M.A_REAR]), integ


def controller(dx, g: Gains, integ, v_target, dt):
    st = read_state(dx)
    u_m = balance_mass(st, g)
    u_dr, integ = speed(st, g, integ, v_target, dt)
    return jnp.array([u_m, u_dr]), integ, st
