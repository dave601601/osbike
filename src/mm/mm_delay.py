"""지연 보상(delay-aware) — Smith/상태예측기 + steer 포함 6-state 시스템ID (보강③).

목적: "model-based 지연보상은 지형에 깨진다"(RL 명분)를 강한 형태로 검증.
구 실험의 예측기는 self-steering 을 통째로 뺀 4-state 라 반론 여지가 있었다 —
steer 를 포함한 모델로도 깨지는지, 예측오차가 실제로 지형에서 폭발하는지 직접 잰다.

- smith4: `mm_lqr.state_space()` 해석 4-state [lean, roll̇, y_m, ẏ] 롤포워드.
- smith6: [lean, roll̇, y_m, ẏ, steer, steeṙ] 6-state + 입력 [F_slide, τ_rear].
  평지(v=1.5, 60N) 폐루프 롤아웃에 스텝별 여기(excitation)를 더해 DMDc(최소자승)로
  적합 → `mm_sysid_6state.json`. FD 선형화의 접촉강성 오염(교훈#7)을 피하는 경로.
  지형은 두 모델 다 미모델 — 그게 원리적 한계인지가 질문.

예측 구조 (50Hz ZOH + 지연 n스텝, n≤제어주기): 틱에서 새 명령은 t+delay 부터 적용,
그때까지는 직전 명령 u_prev 가 적용 중 → x_pred(t+delay) = 모델(x(t), u_prev, n) 로
롤포워드하고 u = -K·(x_pred - x_ref). delay=0 이면 base 와 동일(무보상 환원).

사용:  python mm_delay.py     # 적합 + 검증 + 예측오차(평지 vs 지형) 리포트
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import numpy as np
import mujoco
from scipy.linalg import expm

import mm_model as M
import mm_lqr
import mm_controller as C

SYSID_JSON = str(M.PARAMS / "mm_sysid_6state.json")


def z6(st):
    """St → 6-state 벡터."""
    return np.array([float(st.lean), float(st.roll_rate), float(st.y_m),
                     float(st.ydot), float(st.steer), float(st.steer_rate)])


class Predictor:
    """이산(물리 dt) 선형모델 롤포워드. n_states=4(해석) 또는 6(sysid)."""

    def __init__(self, A, B, n_states):
        self.A, self.B, self.n = np.asarray(A), np.asarray(B), n_states

    @classmethod
    def smith4(cls, model):
        A_c, B_c, _ = mm_lqr.state_space(model)
        n = 4
        Maug = np.zeros((n + 1, n + 1)); Maug[:n, :n] = A_c; Maug[:n, n:] = B_c
        Ed = expm(Maug * M.DT)
        return cls(Ed[:n, :n], Ed[:n, n:], 4)     # B: F_slide 만

    @classmethod
    def smith6(cls, path=SYSID_JSON):
        d = json.load(open(path))
        return cls(np.array(d["A"]), np.array(d["B"]), 6)

    def predict_seq(self, z, seq):
        """z 를 seq=[(u, n_steps), ...] 명령 타임라인으로 롤포워드 → 앞 4개 반환.
        (지연 > 제어주기면 in-flight 명령이 여럿 → 세그먼트별 적용)"""
        z = np.asarray(z[:self.n], dtype=float).copy()
        for u, n in seq:
            u = np.asarray(u, dtype=float)[:self.B.shape[1]]
            for _ in range(int(n)):
                z = self.A @ z + self.B @ u
        return z[:4]

    def predict(self, z, u_prev, n_steps):
        """단일 세그먼트 편의 래퍼 (지연 ≤ 제어주기)."""
        return self.predict_seq(z, [(u_prev, n_steps)])


def horizon_seq(step, cur, pending, n_delay):
    """예측 horizon [step, step+n_delay) 동안 적용될 명령 타임라인.
    cur = 릴리즈 pop 이후의 현재 적용 명령, pending = [(release_step, u), ...] 오름차순."""
    seq, last, uu = [], step, cur
    for r, u in pending:
        if r >= step + n_delay:
            break
        if r > last:
            seq.append((uu, r - last))
        uu, last = u, r
    seq.append((uu, step + n_delay - last))
    return seq


def _flat_model(force=60.0):
    xml = open(M.XML).read().replace('ctrlrange="-20 20"',
                                     f'ctrlrange="-{force} {force}"')
    return mujoco.MjModel.from_xml_string(xml)


def _gains(m, force=60.0):
    K, _, _ = mm_lqr.design(m, y_max=0.15, F_max=force)
    M.CTRL_HI[M.A_SLIDE], M.CTRL_LO[M.A_SLIDE] = force, -force
    b = json.load(open(M.PARAMS / "mm_lqr_gains.json"))
    return C.Gains(*[float(k) for k in K], b["kp_v"], b["ki_v"], b["k_yaw"],
                   b["kd_yaw"], b["lean_max"], b["ki_yaw"], b["k_lat"],
                   b["kd_lat"], b["yaw_corr_max"], b["lat_slew"])


def fit(v=1.5, force=60.0, sec=240.0, seed=0, out=SYSID_JSON):
    """평지 폐루프 + 스텝별 여기 → z_{k+1} = A z_k + B u_k 최소자승 (DMDc).
    여기는 슬라이더 ±12N·뒷바퀴 ±1Nm 균등랜덤(2스텝 홀드) — 폐루프 상관 편향을
    독립 여기로 깬다. 결정론 sim 이라 잡음항 없음 → lstsq 로 충분."""
    m = _flat_model(force)
    G = _gains(m, force)
    d = mujoco.MjData(m)
    a = np.radians(0.5) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
    d.qvel[0] = v
    d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = v / M.WHEEL_R
    mujoco.mj_forward(m, d)
    rng = np.random.RandomState(seed)
    cs = C.CtrlState()
    n_steps = int(sec / M.DT)
    Z, U = np.zeros((n_steps, 6)), np.zeros((n_steps, 2))
    zoh = np.zeros(2); exc = np.zeros(2)
    k = 0
    for step in range(n_steps):
        if step % 5 == 0:                       # 50Hz 제어 (배포 동일)
            ctrl, cs, st = C.controller(d, G, cs, v, 5 * M.DT, yaw_ref=0.0)
            zoh = np.asarray(ctrl, dtype=float)
        else:
            st = C.read_state(d)
        if step % 2 == 0:                       # 여기: 2스텝 홀드
            exc = np.array([rng.uniform(-12, 12), rng.uniform(-1, 1)])
        u = np.clip(zoh + exc,
                    [M.CTRL_LO[0], M.CTRL_LO[1]], [M.CTRL_HI[0], M.CTRL_HI[1]])
        Z[k], U[k] = z6(st), u
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
        qw, qx, qy, qz = d.qpos[3:7]
        assert 1 - 2 * (qx * qx + qy * qy) > 0.7, f"여기 중 낙하 @{step*M.DT:.1f}s"
        k += 1
    # 회귀 (마지막 20% 는 홀드아웃 검증)
    n_tr = int(k * 0.8)
    X = np.hstack([Z[:k - 1], U[:k - 1]])
    Y = Z[1:k]
    W, *_ = np.linalg.lstsq(X[:n_tr], Y[:n_tr], rcond=None)
    A, B = W[:6].T, W[6:].T
    resid = Y[n_tr:] - X[n_tr:] @ W
    rms = np.sqrt((resid ** 2).mean(axis=0))
    # 5스텝(20ms) 자유 롤포워드 홀드아웃 오차
    P = Predictor(A, B, 6)
    errs = []
    for i in range(n_tr, k - 6, 7):
        zp = P.predict(Z[i], U[i], 5)           # 근사: u 를 5스텝 동결
        errs.append(Z[i + 5][:2] - zp[:2])
    e5 = np.sqrt((np.array(errs) ** 2).mean(axis=0))
    info = dict(A=A.tolist(), B=B.tolist(), dt=M.DT, v=v, force=force, sec=sec,
                seed=seed, holdout_1step_rms=rms.tolist(),
                holdout_5step_rms_lean_rrate=e5.tolist(),
                eig_abs=sorted(np.abs(np.linalg.eigvals(A)).tolist(), reverse=True))
    json.dump(info, open(out, "w"), indent=1)
    return info


def pred_error_report(delay_ms=20, seeds=(0, 1, 2)):
    """평지 vs 지형(s0.5)에서 smith4/smith6 의 n_delay 예측오차 RMS (lean, roll̇).
    base(무보상) 컨트롤러로 주행하며 매 틱 예측만 수행 → 도착 시점 실측과 비교.
    'model-based 가 지형에 깨진다' 의 직접 증거(또는 반증)."""
    import mm_envelope as E
    n_delay = delay_ms // int(M.DT * 1000)
    rows = []
    for terr in ("flat", "s0.5"):
        errs = {"smith4": [], "smith6": []}
        for seed in seeds:
            if terr == "flat":
                m = _flat_model(E.FORCE); zt = 0.0
            else:
                _, slope, mu, bump = E.SEVERITIES[1]
                m, zt = E.build_model(slope, mu, bump, seed)
            G = _gains(m, E.FORCE)
            P4, P6 = Predictor.smith4(m), Predictor.smith6()
            d = mujoco.MjData(m)
            a = np.radians(E.PERT_DEG) / 2
            d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
            d.qpos[2] = 0.30 + zt
            d.qvel[0] = E.V_TARGET
            d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = E.V_TARGET / M.WHEEL_R
            mujoco.mj_forward(m, d)
            cs, yref, px, py = C.CtrlState(), 0.0, 0.0, 0.0
            cdt = E.CTRL_EVERY * M.DT
            slew = np.radians(E.SLEW_DPS) * cdt
            pending, cur = [], np.zeros(2)
            preds = []                          # (도착 step, pred4, pred6)
            for step in range(int(E.HORIZON_S / M.DT)):
                t = step * M.DT
                # 릴리즈 pop 을 틱보다 먼저 — 예측의 u_prev 는 "지금부터 horizon 동안
                # 실제 적용될 명령"이어야 함 (틱 먼저 하면 delay=한 주기일 때 두 틱 전
                # 명령으로 예측하는 off-by-one → 평지 예측오차까지 오염)
                while pending and pending[0][0] <= step:
                    cur = pending.pop(0)[1]
                if step % E.CTRL_EVERY == 0:
                    tgt = np.radians(E.TURN_DEG) if t >= E.TURN_T else 0.0
                    yref += float(np.clip(tgt - yref, -slew, slew))
                    st = C.read_state(d)
                    z = z6(st)
                    preds.append((step + n_delay,
                                  P4.predict(z, cur, n_delay),
                                  P6.predict(z, cur, n_delay)))
                    ctrl, cs, _ = C.controller(d, G, cs, E.V_TARGET, cdt,
                                               yaw_ref=yref, path=(px, py, yref))
                    px += E.V_TARGET * cdt * np.cos(yref)
                    py += E.V_TARGET * cdt * np.sin(yref)
                    pending.append((step + n_delay, np.asarray(ctrl, dtype=float)))
                while preds and preds[0][0] <= step:
                    _, p4, p6 = preds.pop(0)
                    za = z6(C.read_state(d))
                    errs["smith4"].append(za[:2] - p4[:2])
                    errs["smith6"].append(za[:2] - p6[:2])
                d.ctrl[:] = cur
                mujoco.mj_step(m, d)
                qw, qx, qy, qz = d.qpos[3:7]
                if 1 - 2 * (qx * qx + qy * qy) < 0.7:
                    break
        for name in ("smith4", "smith6"):
            e = np.array(errs[name])
            rms = np.sqrt((e ** 2).mean(axis=0))
            rows.append((terr, name, np.degrees(rms[0]), np.degrees(rms[1]), len(e)))
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=== 6-state 시스템ID (평지 v=1.5, 60N) ===")
    info = fit()
    print(f"  1-step holdout RMS: {np.array(info['holdout_1step_rms'])}")
    print(f"  5-step(20ms) holdout RMS [lean, roll̇]: "
          f"{np.array(info['holdout_5step_rms_lean_rrate'])}")
    print(f"  |eig(A)| top3: {info['eig_abs'][:3]}  -> {SYSID_JSON}")
    print(f"\n=== 예측오차 RMS (20ms 예측, [lean°, roll̇°/s], 3 seeds) ===")
    for terr, name, e0, e1, n in pred_error_report():
        print(f"  {terr:<5} {name}: lean {e0:7.3f}°  roll̇ {e1:7.2f}°/s   (n={n})")
