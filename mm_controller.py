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
    # heading (yaw_ref → lean_ref, 무게추 유도 카운터스티어). 기본 0 = 직진 전용.
    # 경사에서 self-steering이 heading을 내리막으로 끌고 감(2°경사서 yaw -66° 폭주,
    # slip ~1°뿐 = 순수 heading 상실). 처방 = lean_max ≥ 경사각+여유 (경사각만큼의
    # lean은 유효중력에 수직이라 슬라이더 부담 ≈0) + ki_yaw 적분(P-only는 상주오차).
    k_yaw: float = 0.0; kd_yaw: float = 0.0; lean_max: float = 0.035
    ki_yaw: float = 0.0
    # lateral 외곽루프 (crosstrack → yaw_ref 보정). 기본 0 = 미사용.
    # 보정은 slew 제한 필수 — 선회 램프 중 회전하는 기준선에 대한 e/ė가 요동쳐
    # 즉발 보정을 넣으면 낙하 (lean1°+k_lat부터 전 seed 낙하 관측).
    k_lat: float = 0.0; kd_lat: float = 0.0
    yaw_corr_max: float = 0.21; lat_slew: float = 0.026   # ±12°, 1.5°/s
    # 외곽루프 속도 스케줄 frac=clip((v_target−v_lean0)/(v_lean−v_lean0),0,1):
    # lean 캡 = max(frac·lean_max, 0.5°), ki_yaw·k_lat 도 frac 배 → v≤v_lean0 에선
    # 구 P-only(0.5°) 컨트롤러로 정확히 환원. 이유: 저속은 lean당 yaw 권한(g·tanθ/v)이
    # 커져 같은 게인이 고게인화 + marginal 균형이라, 캡만 줄여도(0.5°) 적분/k_lat
    # 동역학 주입만으로 v=0.65 낙하 (isolation 검증). v_target 기준(실측 v_fwd 아님)
    # 이유: 범프/경사로 v가 처질 때 캡이 같이 처지면 드리프트 악순환 (ct -7→-16m 관측).
    v_lean: float = 1.5; v_lean0: float = 0.75


class CtrlState(NamedTuple):
    """controller() 가 스텝마다 들고 다니는 적분/슬루 상태. 초기값 CtrlState() 또는 0.0."""
    v_i: float = 0.0        # 속도 PI 적분
    yaw_i: float = 0.0      # heading 적분 (경사 정상상태 lean bias)
    lat_corr: float = 0.0   # lateral 보정 현재값 (slew 메모리)


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
    yaw_rate: float
    y_lat: float
    x: float           # world 위치/속도 (lateral 외곽루프의 crosstrack 용)
    vx_w: float
    vy_w: float


def _quat_to_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_state(dx) -> St:
    q, v = dx.qpos, dx.qvel
    w, x, y, z = q[3], q[4], q[5], q[6]
    yaw = _quat_to_yaw(q[M.Q_QUAT])
    return St(
        lean       = -(2 * (y * z - w * x)),
        up_z       = 1 - 2 * (x * x + y * y),
        roll_rate  = v[3],
        y_m        = q[M.Q_SLIDE],
        ydot       = v[M.V_SLIDE],
        steer      = q[M.Q_STEER],
        steer_rate = v[M.V_STEER],
        # body-forward 속도 (선회 시 world-x 로 추종하면 속도루프가 헛돎)
        v_fwd      = v[0] * jnp.cos(yaw) + v[1] * jnp.sin(yaw),
        yaw        = yaw,
        yaw_rate   = v[5],
        y_lat      = q[1],
        x          = q[0],
        vx_w       = v[0],
        vy_w       = v[1],
    )


def balance_from(x4, g: Gains, lean_ref=0.0):
    """무게추 힘 = -(K·(x - x_ref)), x4=[lean, roll̇, y_m, ẏ]. 지연보상(Smith)은
    예측 상태를 여기에 넣는다 (mm_delay.py)."""
    F = -(g.k_lean * (x4[0] - lean_ref) + g.k_rrate * x4[1]
          + g.k_y * x4[2] + g.k_ydot * x4[3])
    return jnp.clip(F, M.CTRL_LO[M.A_SLIDE], M.CTRL_HI[M.A_SLIDE])


def balance_mass(st: St, g: Gains, lean_ref=0.0):
    """무게추 힘 = -(K·(x - x_ref)). lean+(오른쪽) → 무게추를 왼쪽으로.
    lean_ref≠0 이면 그 기울기를 유지 → self-steering이 그쪽으로 선회 (무게추 조향)."""
    return balance_from((st.lean, st.roll_rate, st.y_m, st.ydot), g, lean_ref)


def heading(st: St, g: Gains, yaw_ref, yaw_i=0.0, dt=0.0, v_sched=None):
    """yaw 오차 → lean_ref (PI). 좌회전(yaw+)엔 왼쪽 기울기(lean−) → 부호 음수.
    적분은 경사의 정상 외란(지속 lean bias 필요)용 — 포화 중엔 적분 정지(anti-windup).
    lean 캡은 v_sched(=v_target) 스케줄 — Gains.v_lean 주석 참조.
    주의: kd_yaw>0 은 역효과(st.yaw_rate가 lean 중 roll과 섞여 균형 파괴) → 기본 0."""
    err = jnp.arctan2(jnp.sin(yaw_ref - st.yaw), jnp.cos(yaw_ref - st.yaw))
    frac = 1.0 if v_sched is None else sched_frac(g, v_sched)
    raw = -(g.k_yaw * err + frac * g.ki_yaw * yaw_i - g.kd_yaw * st.yaw_rate)
    cap = jnp.maximum(g.lean_max * frac, 0.0087)
    lean_ref = jnp.clip(raw, -cap, cap)
    yaw_i = jnp.where((jnp.abs(raw) < cap) & (frac > 0), yaw_i + err * dt, yaw_i)
    return lean_ref, yaw_i


def sched_frac(g: Gains, v_sched):
    """외곽루프 속도 스케줄 계수 (Gains.v_lean 주석 참조)."""
    return jnp.clip((v_sched - g.v_lean0) / (g.v_lean - g.v_lean0), 0.0, 1.0)


def lateral(st: St, g: Gains, px, py, chi, corr_prev=0.0, dt=0.0, v_sched=None):
    """crosstrack 외곽루프: 기준선(점 (px,py), 방위 chi)에서의 횡이탈 e 를 yaw_ref 보정으로.
    e>0 = 경로 왼쪽 → 오른쪽으로 조향(yaw_ref < chi). 보정은 ±yaw_corr_max 클립 +
    lat_slew 슬루 제한 (즉발 보정은 선회 램프에서 낙하 유발 — Gains 주석) +
    저속 frac 감쇠 (v≤v_lean0 에선 0으로 슬루아웃)."""
    s, c = jnp.sin(chi), jnp.cos(chi)
    e  = -(st.x - px) * s + (st.y_lat - py) * c
    ed = -st.vx_w * s + st.vy_w * c
    frac = 1.0 if v_sched is None else sched_frac(g, v_sched)
    des = jnp.clip(-frac * (g.k_lat * e + g.kd_lat * ed),
                   -g.yaw_corr_max, g.yaw_corr_max)
    corr = corr_prev + jnp.clip(des - corr_prev, -g.lat_slew * dt, g.lat_slew * dt)
    return chi + corr, corr


def speed(st: St, g: Gains, integ, v_target, dt):
    err = v_target - st.v_fwd
    integ = integ + err * dt
    tau = g.kp_v * err + g.ki_v * integ
    return jnp.clip(tau, M.CTRL_LO[M.A_REAR], M.CTRL_HI[M.A_REAR]), integ


def controller(dx, g: Gains, cs, v_target, dt, yaw_ref=0.0, path=None, x_pred4=None):
    """캐스케이드: [lateral →] heading → lean_ref → balance. k_yaw=0 이면 순수 직립.
    path=(px,py,chi) 를 주면 yaw_ref 대신 crosstrack 보정된 방위를 추종 (k_lat>0 필요).
    x_pred4 를 주면 balance 가 예측 상태로 동작 (지연보상 — 외곽루프는 느려서 현재 상태).
    cs = CtrlState (구 코드의 float integ 도 받음 — 속도 적분으로 승격)."""
    if not isinstance(cs, CtrlState):
        cs = CtrlState(v_i=float(cs))
    st = read_state(dx)
    corr = cs.lat_corr
    if path is not None:
        yaw_ref, corr = lateral(st, g, *path, corr_prev=cs.lat_corr, dt=dt,
                                v_sched=v_target)
    lean_ref, yaw_i = heading(st, g, yaw_ref, cs.yaw_i, dt, v_sched=v_target)
    if x_pred4 is None:
        u_m = balance_mass(st, g, lean_ref)
    else:
        u_m = balance_from(x_pred4, g, lean_ref)
    u_dr, v_i = speed(st, g, cs.v_i, v_target, dt)
    return jnp.array([u_m, u_dr]), CtrlState(v_i, yaw_i, corr), st
