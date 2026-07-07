"""학습 정책(pickle 체크포인트) → CPU numpy 추론 + CPU 롤아웃 (배포/채점 경로).

성능 주장은 반드시 이 경로(CPU mujoco)로 — 학습은 MJX(GPU)지만 saturation-지배
플랜트는 플랫폼 민감 (docs/progress/mm.md sim2sim 교훈). mm_envelope 의 RL 채점도
여기의 Policy 를 쓴다.

사용:  python mm_policy.py ckpt/flat_cold16k/params_00149.pkl   # flat sim2sim 스모크
"""
import pickle
import sys

import numpy as np
import mujoco

import mm_env as EV          # 상수(OBS_DIM 등)만 사용 — jax 로드는 감수


class Policy:
    """결정론(mean) 추론. obs 정규화 포함. 순수 numpy."""

    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.p, self.nrm = d["params"], d["nrm"]
        self.meta = {k: d.get(k) for k in ("it", "stage")}

    def __call__(self, obs):
        x = np.clip((obs - self.nrm["mean"]) / np.sqrt(self.nrm["var"] + 1e-8),
                    -10.0, 10.0)
        for W, b in self.p["pi"][:-1]:
            x = np.tanh(x @ W + b)
        W, b = self.p["pi"][-1]
        return np.clip(x @ W + b, -1.0, 1.0)


def rollout_cpu(policy, v_target=1.5, turn_deg=0.0, sec=30.0, pert_deg=0.5,
                delay_ms=0, model=None):
    """CPU mujoco 롤아웃 — mm_env 와 동일한 obs/task 정의 (지연은 발행 버퍼).
    반환: dict(survived, t_end, lean_rms, ct_end, yaw_end)."""
    m = model
    if m is None:
        xml = open(EV.XML).read().replace('ctrlrange="-20 20"',
                                          f'ctrlrange="-{EV.F_MAX:g} {EV.F_MAX:g}"')
        xml = xml.replace('ctrlrange="-4 4"', f'ctrlrange="-{EV.TAU_MAX:g} {EV.TAU_MAX:g}"')
        m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    a = np.radians(pert_deg) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
    d.qvel[0] = v_target
    d.qvel[EV.V_REAR] = d.qvel[EV.V_FRONT] = v_target / EV.WHEEL_R
    mujoco.mj_forward(m, d)
    n_delay = delay_ms // int(EV.DT * 1000)
    yref, px, py = 0.0, 0.0, 0.0
    act_buf = np.zeros((EV.ACT_HIST + 1, EV.ACT_DIM))
    scale = np.array([EV.F_MAX, EV.TAU_MAX])
    lean2, nst, ct = 0.0, 0, 0.0
    n_steps = int(sec / EV.DT)
    for step in range(n_steps):
        t = step * EV.DT
        if step % EV.CTRL_EVERY == 0:
            tgt = np.radians(turn_deg) if t >= EV.TURN_T else 0.0
            yref += float(np.clip(tgt - yref, -EV.SLEW * EV.CTRL_DT,
                                  EV.SLEW * EV.CTRL_DT))
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
                             np.clip(ct / 5.0, -2, 2), np.clip(ctdot, -3, 3),
                             v_target])
            obs = np.concatenate([core, act_buf[:EV.ACT_HIST].ravel()])
            act = policy(obs)
            act_buf = np.roll(act_buf, 1, axis=0)
            act_buf[0] = act
            px += v_target * EV.CTRL_DT * np.cos(yref)
            py += v_target * EV.CTRL_DT * np.sin(yref)
        i = step % EV.CTRL_EVERY
        j = (max(n_delay - i, 0) + EV.CTRL_EVERY - 1) // EV.CTRL_EVERY
        d.ctrl[:] = act_buf[j] * scale
        mujoco.mj_step(m, d)
        w, x, y, z = d.qpos[3:7]
        lean2 += (-(2 * (y * z - w * x))) ** 2
        nst += 1
        if 1 - 2 * (x * x + y * y) < 0.7:
            return dict(survived=False, t_end=round(t, 2),
                        lean_rms=float(np.degrees(np.sqrt(lean2 / nst))),
                        ct_end=round(float(ct), 2), yaw_end=None)
    w, x, y, z = d.qpos[3:7]
    yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return dict(survived=True, t_end=sec,
                lean_rms=float(np.degrees(np.sqrt(lean2 / nst))),
                ct_end=round(float(ct), 2), yaw_end=round(yaw, 1))


if __name__ == "__main__":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/flat_cold16k/params_00149.pkl"
    pol = Policy(path)
    print(f"[policy] {path} (stage={pol.meta['stage']}, it={pol.meta['it']})")
    for v in (1.0, 1.5, 2.0):
        r = rollout_cpu(pol, v_target=v)
        print(f"  CPU flat 직진 v={v}: {r}")
