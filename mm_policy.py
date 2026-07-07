"""학습 정책(pickle 체크포인트) → CPU numpy 추론 + CPU 롤아웃 (배포/채점 경로).

성능 주장은 반드시 이 경로(CPU mujoco)로 — 학습은 MJX(GPU)지만 saturation-지배
플랜트는 플랫폼 민감 (docs/progress/mm.md sim2sim 교훈). mm_envelope 의 RL 채점도
여기의 Policy 를 쓴다.

사용:  python mm_policy.py ckpt/flat_cold16k/params_00149.pkl   # flat sim2sim 스모크
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # __main__ 이 아니라 여기서 — mm_env
os.environ.setdefault("OMP_NUM_THREADS", "1")   # import 가 jax 를 끌고 오므로 GPU 선점 방지

import pickle
import sys

import numpy as np
import mujoco

import mm_env as EV          # 상수/GAINS 재사용 (단일 소스) — CPU jax 로 로드됨


RES_SCALE = {"res": 0.3}             # stage → residual 배율 (mm_ppo.STAGES 와 일치)


class Policy:
    """결정론(mean) 추론. obs 정규화 포함. 순수 numpy."""

    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.p, self.nrm = d["params"], d["nrm"]
        self.meta = {k: d.get(k) for k in ("it", "stage")}
        self.in_dim = self.p["pi"][0][0].shape[0]   # 구(25차원) ckpt 하위호환
        self.res_scale = RES_SCALE.get(self.meta.get("stage"), 0.0)

    def __call__(self, obs):
        obs = np.asarray(obs)[: self.in_dim]
        x = np.clip((obs - self.nrm["mean"][: self.in_dim])
                    / np.sqrt(self.nrm["var"][: self.in_dim] + 1e-8), -10.0, 10.0)
        for W, b in self.p["pi"][:-1]:
            x = np.tanh(x @ W + b)
        W, b = self.p["pi"][-1]
        return np.clip(x @ W + b, -1.0, 1.0)


class CpuController:
    """CPU 배포/채점용 통합 컨트롤러: obs 구축 + [residual 이면 base 합성] + 이력 관리.
    mm_envelope 의 RL 채점과 rollout_cpu 가 공유 — obs/합성 정의가 한 곳에만 존재해야
    학습(mm_env)과 어긋나지 않는다. 반환은 물리 단위 [F, tau] (지연 큐는 호출측 소관)."""

    def __init__(self, ckpt_path, res_scale=None):
        import mm_controller as C
        self.C = C
        self.pol = Policy(ckpt_path)
        self.res = self.pol.res_scale if res_scale is None else res_scale
        self.G = EV.GAINS                        # 학습 env 와 동일 base 게인 (단일 소스)
        self.reset()

    def reset(self):
        self.act_hist = np.zeros((EV.ACT_HIST, EV.ACT_DIM))
        self.cs = np.zeros(3)                    # [v_i, yaw_i, lat_corr]
        self.u_base = np.zeros(2)

    def tick(self, d, yref, px, py, v_target):
        q, v = d.qpos, d.qvel
        w, x, y, z = q[3:7]
        lean = -(2 * (y * z - w * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v_fwd = v[0] * np.cos(yaw) + v[1] * np.sin(yaw)
        s, c = np.sin(yref), np.cos(yref)
        ct = -(q[0] - px) * s + (q[1] - py) * c
        ctdot = -v[0] * s + v[1] * c
        core = np.array([lean, v[3], q[EV.Q_SLIDE] / EV.STROKE, v[EV.V_SLIDE],
                         q[EV.Q_STEER], v[EV.V_STEER], v[5], v_fwd,
                         np.sin(yref - yaw), np.cos(yref - yaw),
                         np.clip(ct / 5.0, -2, 2), np.clip(ctdot, -3, 3), v_target])
        obs = np.concatenate([core, self.act_hist.ravel(), self.cs, self.u_base])
        a = self.pol(obs)
        if self.res > 0:
            C, G = self.C, self.G
            stc = C.read_state(d)
            yref_c, corr = C.lateral(stc, G, px, py, yref, corr_prev=self.cs[2],
                                     dt=EV.CTRL_DT, v_sched=v_target)
            lean_ref, yaw_i = C.heading(stc, G, yref_c, self.cs[1], EV.CTRL_DT,
                                        v_sched=v_target)
            u_m = C.balance_mass(stc, G, lean_ref)
            u_dr, v_i = C.speed(stc, G, self.cs[0], v_target, EV.CTRL_DT)
            self.u_base = np.array([float(u_m) / EV.F_MAX, float(u_dr) / EV.TAU_MAX])
            self.cs = np.array([float(v_i), float(yaw_i), float(corr)])
            cmd = np.clip(self.u_base + self.res * a, -1.0, 1.0)
        else:
            cmd = a
        self.act_hist = np.roll(self.act_hist, 1, axis=0)
        self.act_hist[0] = cmd
        return cmd * np.array([EV.F_MAX, EV.TAU_MAX])


def rollout_cpu(ctrl: CpuController, v_target=1.5, turn_deg=0.0, sec=30.0,
                pert_deg=0.5, delay_ms=0, model=None):
    """CPU mujoco 롤아웃 — mm_envelope 와 동일한 지연 큐/판정. ctrl 은 매 호출 reset."""
    m = model
    if m is None:
        xml = open(EV.XML).read().replace('ctrlrange="-20 20"',
                                          f'ctrlrange="-{EV.F_MAX:g} {EV.F_MAX:g}"')
        xml = xml.replace('ctrlrange="-4 4"', f'ctrlrange="-{EV.TAU_MAX:g} {EV.TAU_MAX:g}"')
        m = mujoco.MjModel.from_xml_string(xml)
    ctrl.reset()
    d = mujoco.MjData(m)
    a = np.radians(pert_deg) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
    d.qvel[0] = v_target
    d.qvel[EV.V_REAR] = d.qvel[EV.V_FRONT] = v_target / EV.WHEEL_R
    mujoco.mj_forward(m, d)
    n_delay = delay_ms // int(EV.DT * 1000)
    yref, px, py = 0.0, 0.0, 0.0
    pending, cur = [], np.zeros(2)
    lean2, nst, ct = 0.0, 0, 0.0
    for step in range(int(sec / EV.DT)):
        t = step * EV.DT
        while pending and pending[0][0] <= step:
            cur = pending.pop(0)[1]
        if step % EV.CTRL_EVERY == 0:
            tgt = np.radians(turn_deg) if t >= EV.TURN_T else 0.0
            yref += float(np.clip(tgt - yref, -EV.SLEW * EV.CTRL_DT,
                                  EV.SLEW * EV.CTRL_DT))
            u = ctrl.tick(d, yref, px, py, v_target)
            s, c = np.sin(yref), np.cos(yref)
            ct = float(-(d.qpos[0] - px) * s + (d.qpos[1] - py) * c)
            px += v_target * EV.CTRL_DT * np.cos(yref)
            py += v_target * EV.CTRL_DT * np.sin(yref)
            pending.append((step + n_delay, u))
        while pending and pending[0][0] <= step:
            cur = pending.pop(0)[1]
        d.ctrl[:] = cur
        mujoco.mj_step(m, d)
        w, x, y, z = d.qpos[3:7]
        lean2 += (-(2 * (y * z - w * x))) ** 2
        nst += 1
        if 1 - 2 * (x * x + y * y) < 0.7:
            return dict(survived=False, t_end=round(t, 2),
                        lean_rms=float(np.degrees(np.sqrt(lean2 / nst))),
                        ct_end=round(ct, 2), yaw_end=None)
    w, x, y, z = d.qpos[3:7]
    yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return dict(survived=True, t_end=sec,
                lean_rms=float(np.degrees(np.sqrt(lean2 / nst))),
                ct_end=round(ct, 2), yaw_end=round(yaw, 1))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/flat_cold16k/params_00149.pkl"
    ctrl = CpuController(path)
    print(f"[policy] {path} (stage={ctrl.pol.meta['stage']}, it={ctrl.pol.meta['it']}, "
          f"res={ctrl.res})")
    for v, td in ((1.0, 0.0), (1.5, 0.0), (1.5, 30.0), (2.0, 0.0)):
        r = rollout_cpu(ctrl, v_target=v, turn_deg=td)
        print(f"  CPU flat v={v} turn={td:g}°: {r}")
