"""MJX 배치 RL 환경 — moving-mass 자전거 (mm_envelope 프로토콜의 일반화판).

과제: v_target 추종 + t=2s에 yaw_goal 선회(slew 3°/s) + 기준경로(명령 방위 적분) 유지.
관측(25) = [lean, roll̇, y_m/0.15, ẏ, steer, steeṙ, yaẇ, v_fwd, sin/cos(yaw_err),
            ct/5m, ċt, v_target] + 최근 발행 행동 6개(6×2 — 지연 24ms POMDP 처방)
행동(2)  = [슬라이더 힘/60N, 뒷바퀴 토크/4Nm] (정책 출력 [-1,1])
보상     = 생존 − lean² − 코스(ct) − 헤딩 − 속도 − Δa²(부드러움) − 에너지 − 스트로크접촉
           (생존-only는 "내리막 항복" cheat — docs/progress/mm.md 보강②)

Residual 모드 (DR.res_scale > 0): 최종 명령 = clip(base + res_scale·π, ±1).
base = ② 고전 baseline(LQR 60N + heading PI + k_lat, mm_controller — JAX라 그대로
vmap)이 env 안에서 50Hz로 돎. base의 적분기(CtrlState)·직전 base 명령은 관측에
포함(은닉 상태 제거). base 게인은 명목 모델 고정 — DR 파라미터를 모름(실차와 동일,
robustness 비교의 요점). Δa·에너지 벌점은 최종 명령 기준.

Domain randomization (에피소드 단위, DR() 기본값 = 전부 명목):
  지연 0~24ms(4ms 격자, 서브스텝 정확 적용) · 옆경사(중력 틸트) · μ(바퀴+바닥) ·
  슬라이더/라이더 질량 · 슬라이드 감쇠 · 액추에이터 게인 (모델 leaf 만 per-env vmap,
  brax 식) · 랜덤 푸시(xfrc — MJX가 hfield×cylinder 접촉 미지원이라 범프 프록시.
  최종 채점은 CPU 실지형 mm_envelope.py)

주의: 학습(MJX/GPU) → 평가(CPU) sim2sim 갭 교훈 — 성능 주장은 CPU 하네스로만.
"""
import json
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

XML = "moving_mass_bicycle.xml"
DT = 0.004
CTRL_EVERY = 5                      # 250Hz 물리 / 5 = 50Hz 제어
CTRL_DT = DT * CTRL_EVERY
EP_LEN = 1500                       # 30s
ACT_DIM = 2
ACT_HIST = 6                        # 지연 최대 24ms(6 물리스텝) 커버
CORE_DIM = 13
N_FRAMES = 4                        # 코어 상태 프레임 스태킹 — 미지 지연·DR 파라미터는
                                    # 잠재변수(POMDP)라 단일 프레임으론 식별 불가
                                    # (명령 이력 × 상태 "반응" 이력이 있어야 추론 가능;
                                    #  res_hard2 의 저지연 붕괴 98→12% 가 그 실증)
OBS_DIM = CORE_DIM * N_FRAMES + ACT_HIST * ACT_DIM + 5   # +5 = base 적분기+직전 base 명령
F_MAX, TAU_MAX = 60.0, 4.0
STROKE = 0.15
WHEEL_R = 0.30
TURN_T = 2.0
SLEW = float(jnp.radians(3.0))


class DR(NamedTuple):
    """DR 강도 (기본값 = 명목). jit static 인자."""
    delay_max: int = 0              # 지연 ∈ [delay_min, delay_max] [물리스텝, 0..6]
    delay_min: int = 0
    slope_deg: float = 0.0          # 옆경사 ∈ [slope_lo, slope_deg]
    slope_lo: float = 0.0
    mu_lo: float = 1.4              # μ ∈ [lo, hi] (명목 1.4 = XML 바퀴값)
    mu_hi: float = 1.4
    mass_pct: float = 0.0           # 슬라이더/라이더 질량 ±비율
    damp_hi: float = 1.0            # 슬라이드 감쇠 ×[1/hi, hi]
    gain_pct: float = 0.0           # 액추에이터 게인 ±비율
    push_n: float = 0.0             # 랜덤 푸시 ∈ [push_lo, push_n] [N] (범프 프록시)
    push_lo: float = 0.0
    turn_deg: float = 0.0           # 선회 목표 ±상한 (0 = 직진만)
    v_lo: float = 1.5
    v_hi: float = 1.5
    pert_deg: float = 0.5           # 초기 lean 섭동 ±상한
    res_scale: float = 0.0          # >0: residual 모드 (최종 = clip(base+res·π, ±1))
    res_pen: float = 0.0            # 잔차 크기 벌점 −res_pen·|π|² — scale 1.0 무앵커
                                    # 정책이 과작동 진동 모드(평지 lean 3-5°)에 빠진
                                    # 사고 재발 방지 (잔차 절반 평가로 회복 실증)


class EnvState(NamedTuple):
    dx: mjx.Data
    t: jnp.ndarray
    v_target: jnp.ndarray
    yaw_goal: jnp.ndarray
    yref: jnp.ndarray
    pxy: jnp.ndarray                # (2,) 기준경로 앵커
    n_delay: jnp.ndarray            # 물리스텝 단위 지연
    act_buf: jnp.ndarray            # (ACT_HIST+1, 2) 발행 이력, [0]=최신
    push_n: jnp.ndarray
    cs: jnp.ndarray                 # (3,) base 적분기 [v_i, yaw_i, lat_corr]
    u_base: jnp.ndarray             # (2,) 직전 base 명령 (정규화, obs용)
    frames: jnp.ndarray             # (N_FRAMES, CORE_DIM) 코어 상태 이력, [0]=최신
    burnin: jnp.ndarray             # 초기 위상 랜덤화 세대 (에피소드 통계 제외)
    rng: jnp.ndarray
    ep_ret: jnp.ndarray
    ep_len: jnp.ndarray


_m = mujoco.MjModel.from_xml_path(XML)
_m.actuator_ctrlrange[0] = (-F_MAX, F_MAX)
_m.actuator_ctrlrange[1] = (-TAU_MAX, TAU_MAX)
MX = mjx.put_model(_m)
BID_FRAME = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "frame")
BID_SLIDER = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "mass_slider")
BID_RIDER = int(_m.geom_bodyid[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "rider")])
GID_MU = [mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, g)
          for g in ("ground", "rear_geom", "front_geom")]
DOF_SLIDE = int(_m.jnt_dofadr[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")])
Q_SLIDE, Q_STEER, V_REAR, V_SLIDE, V_STEER, V_FRONT = 8, 9, 6, 7, 8, 9

# residual 모드의 base = ② 고전 baseline. 게인은 명목(60N) 모델로 한 번 설계·고정
# (DR 파라미터를 모르는 배포 컨트롤러 — mm_envelope 의 base 와 동일 구성).
import mm_model as _MM                     # noqa: E402  (mm_controller 클립 한계 소스)
import mm_controller as C                  # noqa: E402
import mm_lqr as _LQR                      # noqa: E402
_MM.CTRL_HI[_MM.A_SLIDE], _MM.CTRL_LO[_MM.A_SLIDE] = F_MAX, -F_MAX
_K4, _, _ = _LQR.design(_m, y_max=STROKE, F_max=F_MAX)
_gj = json.load(open("mm_lqr_gains.json"))
GAINS = C.Gains(float(_K4[0]), float(_K4[1]), float(_K4[2]), float(_K4[3]),
                _gj["kp_v"], _gj["ki_v"], _gj["k_yaw"], _gj["kd_yaw"],
                _gj["lean_max"], _gj["ki_yaw"], _gj["k_lat"], _gj["kd_lat"],
                _gj["yaw_corr_max"], _gj["lat_slew"])

# DR 대상 모델 leaf 만 per-env(축 0), 나머지 공유(None) — 메모리 절약 + 병합 명확화
MODEL_AXES = jax.tree.map(lambda _: None, MX)
MODEL_AXES = MODEL_AXES.replace(
    opt=MODEL_AXES.opt.replace(gravity=0), geom_friction=0,
    body_mass=0, body_inertia=0, dof_damping=0, actuator_gainprm=0)

STATE_AXES = EnvState(dx=0, t=0, v_target=0, yaw_goal=0, yref=0, pxy=0,
                      n_delay=0, act_buf=0, push_n=0, cs=0, u_base=0,
                      frames=0, burnin=0, rng=0, ep_ret=0, ep_len=0)


def _randomize_model(rng, dr: DR):
    """에피소드용 모델 (DR leaf 샘플). vmap(out_axes=MODEL_AXES) 로 배치."""
    k = jax.random.split(rng, 6)
    phi = jax.random.uniform(k[0], (), minval=jnp.radians(dr.slope_lo),
                             maxval=jnp.radians(dr.slope_deg))
    sgn = jnp.sign(jax.random.uniform(k[1], (), minval=-1.0, maxval=1.0) + 1e-9)
    gravity = jnp.array([0.0, -9.81 * jnp.sin(phi) * sgn, -9.81 * jnp.cos(phi)])
    mu = jax.random.uniform(k[2], (), minval=dr.mu_lo, maxval=dr.mu_hi)
    friction = MX.geom_friction
    for gid in GID_MU:
        friction = friction.at[gid, 0].set(mu)
    ms = 1.0 + dr.mass_pct * jax.random.uniform(k[3], (2,), minval=-1.0, maxval=1.0)
    body_mass = MX.body_mass.at[BID_SLIDER].mul(ms[0]).at[BID_RIDER].mul(ms[1])
    body_inertia = MX.body_inertia.at[BID_SLIDER].mul(ms[0]).at[BID_RIDER].mul(ms[1])
    ds = dr.damp_hi ** jax.random.uniform(k[4], (), minval=-1.0, maxval=1.0)
    dof_damping = MX.dof_damping.at[DOF_SLIDE].mul(ds)
    gs = 1.0 + dr.gain_pct * jax.random.uniform(k[5], (2,), minval=-1.0, maxval=1.0)
    gainprm = MX.actuator_gainprm.at[0, 0].mul(gs[0]).at[1, 0].mul(gs[1])
    return MX.replace(opt=MX.opt.replace(gravity=gravity), geom_friction=friction,
                      body_mass=body_mass, body_inertia=body_inertia,
                      dof_damping=dof_damping, actuator_gainprm=gainprm)


def _merge_model(done, new, old):
    """done 인 env 만 새 DR leaf 로 교체 (배치 leaf 한정)."""
    def sel(a, b):
        return jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
    return old.replace(
        opt=old.opt.replace(gravity=sel(new.opt.gravity, old.opt.gravity)),
        geom_friction=sel(new.geom_friction, old.geom_friction),
        body_mass=sel(new.body_mass, old.body_mass),
        body_inertia=sel(new.body_inertia, old.body_inertia),
        dof_damping=sel(new.dof_damping, old.dof_damping),
        actuator_gainprm=sel(new.actuator_gainprm, old.actuator_gainprm))


def _reset_one(rng, dr: DR, stagger=False):
    """stagger=True(최초 리셋 전용): 에피소드 위상 t 를 랜덤화 — 전 env 가 동시에
    timeout 되는 '파도'가 per-iteration 에피소드 통계에 주기적 스파이크(1500/T iter)를
    만들고, 파도 사이엔 성공이 집계에 안 잡히는 선택 편향을 만들기 때문 (실측: it46).
    stagger 세대는 burnin 표시 → 통계 제외 (학습 자체엔 그대로 사용)."""
    k = jax.random.split(rng, 7)
    dx = mjx.make_data(MX)
    pert = jnp.radians(dr.pert_deg) * jax.random.uniform(k[0], (), minval=-1., maxval=1.)
    v0 = jax.random.uniform(k[1], (), minval=dr.v_lo, maxval=dr.v_hi)
    a = pert / 2
    qpos = dx.qpos.at[3].set(jnp.cos(a)).at[4].set(jnp.sin(a))
    qvel = (dx.qvel.at[0].set(v0).at[V_REAR].set(v0 / WHEEL_R)
            .at[V_FRONT].set(v0 / WHEEL_R))
    dx = dx.replace(qpos=qpos, qvel=qvel)
    yaw_goal = jnp.radians(dr.turn_deg) * jax.random.uniform(k[2], (), minval=-1., maxval=1.)
    n_delay = jax.random.randint(k[3], (), dr.delay_min, dr.delay_max + 1)
    push_n = jax.random.uniform(k[4], (), minval=dr.push_lo, maxval=dr.push_n) \
        if dr.push_n > 0 else jnp.array(0.0)
    t0 = jax.random.randint(k[6], (), 0, EP_LEN) if stagger else jnp.array(0)
    st = EnvState(dx=dx, t=t0, v_target=v0, yaw_goal=yaw_goal,
                  yref=jnp.array(0.0), pxy=jnp.zeros(2), n_delay=n_delay,
                  act_buf=jnp.zeros((ACT_HIST + 1, ACT_DIM)), push_n=push_n,
                  cs=jnp.zeros(3), u_base=jnp.zeros(2),
                  frames=jnp.zeros((N_FRAMES, CORE_DIM)),
                  burnin=jnp.array(stagger), rng=k[5],
                  ep_ret=jnp.array(0.0), ep_len=jnp.array(0))
    return st._replace(frames=jnp.tile(_core(st), (N_FRAMES, 1)))


def _core(st: EnvState):
    """코어 상태 13 — 프레임 스태킹 단위."""
    q, v = st.dx.qpos, st.dx.qvel
    w, x, y, z = q[3], q[4], q[5], q[6]
    lean = -(2 * (y * z - w * x))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v_fwd = v[0] * jnp.cos(yaw) + v[1] * jnp.sin(yaw)
    yerr = st.yref - yaw
    s, c = jnp.sin(st.yref), jnp.cos(st.yref)
    ct = -(q[0] - st.pxy[0]) * s + (q[1] - st.pxy[1]) * c
    ctdot = -v[0] * s + v[1] * c
    return jnp.array([lean, v[3], q[Q_SLIDE] / STROKE, v[V_SLIDE], q[Q_STEER],
                      v[V_STEER], v[5], v_fwd, jnp.sin(yerr), jnp.cos(yerr),
                      jnp.clip(ct / 5.0, -2.0, 2.0), jnp.clip(ctdot, -3.0, 3.0),
                      st.v_target])


def _obs(st: EnvState):
    return jnp.concatenate([st.frames.ravel(), st.act_buf[:ACT_HIST].ravel(),
                            st.cs, st.u_base])


def _step_one(st: EnvState, action, mxv, dr: DR):
    """action ∈ [-1,1]^2 발행(residual 모드면 base 와 합성) → 지연 버퍼 →
    5 물리스텝(서브스텝 정확 지연) → r/done."""
    action = jnp.clip(action, -1.0, 1.0)          # 가우시안 샘플이 범위 밖일 수 있음
    rng, k_push = jax.random.split(st.rng)
    tgt = jnp.where(st.t * CTRL_DT >= TURN_T, st.yaw_goal, 0.0)
    yref = st.yref + jnp.clip(tgt - st.yref, -SLEW * CTRL_DT, SLEW * CTRL_DT)
    pxy = st.pxy + st.v_target * CTRL_DT * jnp.array([jnp.cos(yref), jnp.sin(yref)])
    if dr.res_scale > 0:                          # 정적 분기 (dr 은 jit static)
        stc = C.read_state(st.dx)
        yref_c, corr = C.lateral(stc, GAINS, st.pxy[0], st.pxy[1], yref,
                                 corr_prev=st.cs[2], dt=CTRL_DT,
                                 v_sched=st.v_target)
        lean_ref, yaw_i = C.heading(stc, GAINS, yref_c, st.cs[1], CTRL_DT,
                                    v_sched=st.v_target)
        u_m = C.balance_mass(stc, GAINS, lean_ref)
        u_dr, v_i = C.speed(stc, GAINS, st.cs[0], st.v_target, CTRL_DT)
        u_base = jnp.array([u_m / F_MAX, u_dr / TAU_MAX])
        cs = jnp.array([v_i, yaw_i, corr])
        cmd = jnp.clip(u_base + dr.res_scale * action, -1.0, 1.0)
    else:
        u_base, cs, cmd = jnp.zeros(2), st.cs, action
    act_buf = jnp.roll(st.act_buf, 1, axis=0).at[0].set(cmd)   # [j] = j틱 전 발행
    scale = jnp.array([F_MAX, TAU_MAX])
    push = jax.random.uniform(k_push, (2,), minval=-1., maxval=1.) * st.push_n
    xfrc = jnp.zeros_like(st.dx.xfrc_applied)
    xfrc = xfrc.at[BID_FRAME, 1].set(push[0]).at[BID_FRAME, 3].set(push[1] * 0.3)

    def substep(i, d):
        # 물리스텝 i 에 적용되는 명령 = ceil((n_delay - i)/CTRL_EVERY) 틱 전 발행분
        j = (jnp.maximum(st.n_delay - i, 0) + CTRL_EVERY - 1) // CTRL_EVERY
        d = d.replace(ctrl=act_buf[j] * scale)
        return mjx.step(mxv, d)

    dx = jax.lax.fori_loop(0, CTRL_EVERY, substep,
                           st.dx.replace(xfrc_applied=xfrc))
    st2 = st._replace(dx=dx, t=st.t + 1, yref=yref, pxy=pxy, act_buf=act_buf,
                      cs=cs, u_base=u_base, rng=rng)
    st2 = st2._replace(frames=jnp.roll(st.frames, 1, axis=0).at[0].set(_core(st2)))
    q, v = dx.qpos, dx.qvel
    w, x, y, z = q[3], q[4], q[5], q[6]
    up_z = 1 - 2 * (x * x + y * y)
    lean = -(2 * (y * z - w * x))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v_fwd = v[0] * jnp.cos(yaw) + v[1] * jnp.sin(yaw)
    s, c = jnp.sin(yref), jnp.cos(yref)
    ct = -(q[0] - pxy[0]) * s + (q[1] - pxy[1]) * c
    da = cmd - st.act_buf[0]                          # 최종 명령의 직전 대비 변화
    fell = up_z < 0.7
    # lean 벌점은 캡 필수: 캡 없으면 낙하 직전 스텝당 -30까지 커져 리턴이 행동과
    # 무관한 추락 구간에 지배되고(어드밴티지 노이즈화), 생존 보너스를 압도해
    # "빨리 죽는 게 이득"인 자살 균형이 생김 (실제 발생: 19M 스텝 KL~0.001 정체)
    r = (1.0
         - 50.0 * jnp.minimum(lean ** 2, 0.02)
         - 0.5 * jnp.clip(ct / 5.0, -2.0, 2.0) ** 2
         - 0.2 * (1.0 - jnp.cos(yref - yaw))
         - 0.2 * (v_fwd - st.v_target) ** 2
         - 0.1 * jnp.sum(da ** 2)
         - 0.01 * jnp.sum(cmd ** 2)
         - dr.res_pen * jnp.sum(action ** 2)
         - 0.1 * (jnp.abs(q[Q_SLIDE]) > 0.9 * STROKE)
         - 10.0 * fell)
    timeout = st2.t >= EP_LEN
    done = fell | timeout
    st2 = st2._replace(ep_ret=st.ep_ret + r, ep_len=st.ep_len + 1)
    return st2, _obs(st2), r, done, timeout


@partial(jax.jit, static_argnums=(1, 2))
def reset(rng, n, dr: DR):
    k1, k2 = jax.random.split(rng)
    mxv = jax.vmap(lambda k: _randomize_model(k, dr),
                   out_axes=MODEL_AXES)(jax.random.split(k1, n))
    st = jax.vmap(lambda k: _reset_one(k, dr, stagger=True))(jax.random.split(k2, n))
    return st, mxv, jax.vmap(_obs)(st)


@partial(jax.jit, static_argnums=(3,))
def step(st: EnvState, mxv, action, dr: DR):
    """배치 스텝 + done 자동 리셋.
    info: timeout(부트스트랩 구분) · terminal_obs(리셋 전 관측 — timeout 부트스트랩용)
          · fin_ret/fin_len(done 시점 에피소드 통계)."""
    st2, ob_term, r, done, timeout = jax.vmap(
        lambda s, a, m: _step_one(s, a, m, dr),
        in_axes=(STATE_AXES, 0, MODEL_AXES))(st, action, mxv)
    fin_ret, fin_len, fin_burnin = st2.ep_ret, st2.ep_len, st2.burnin
    keys = jax.vmap(lambda k: jax.random.split(k)[1])(st2.rng)
    st_new = jax.vmap(lambda k: _reset_one(k, dr))(keys)
    mxv_new = jax.vmap(lambda k: _randomize_model(k, dr), out_axes=MODEL_AXES)(keys)

    def sel(a, b):
        return jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)

    st3 = jax.tree.map(sel, st_new, st2)
    mxv3 = _merge_model(done, mxv_new, mxv)
    ob = jax.vmap(_obs)(st3)
    return st3, mxv3, ob, r, done, dict(timeout=timeout, terminal_obs=ob_term,
                                        fin_ret=fin_ret, fin_len=fin_len,
                                        fin_burnin=fin_burnin)
