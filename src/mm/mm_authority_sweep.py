"""Authority 스윕: 무게추 질량×스트로크×힘 → LQR 재설계 → realism 점수(/5).
사람다운(5/5) 균형에 필요한 액추에이터 사양을 찾는다 (하드웨어 사이징).

각 config: mm_model.build → mm_lqr.design(그 force/stroke) → v=2 직진 롤아웃 →
mm_metrics 지표. 힘 클립은 config의 force 로 (기본 컨트롤러 대신 인라인).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import mujoco
import mm_model as M
import mm_lqr
from mm_metrics import _freq, _jerk_rms, HUMAN

V, PERT, T = 2.0, 0.5, 2500          # 10s (정상상태 지표엔 충분)


def run(mass, stroke, force):
    model = M.build(mass, stroke, force)
    K, eig, _ = mm_lqr.design(model, y_max=stroke, F_max=force)
    d = mujoco.MjData(model)
    a = np.radians(PERT) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0, 0]
    d.qvel[0] = V; d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = V / M.WHEEL_R
    mujoco.mj_forward(model, d)
    ys, Fs, leans = [], [], []
    fell = -1
    for k in range(T):
        q = d.qpos; w, x, y, z = q[3], q[4], q[5], q[6]
        lean = -(2 * (y * z - w * x)); rr = d.qvel[3]
        ym = q[M.Q_SLIDE]; yd = d.qvel[M.V_SLIDE]
        F = np.clip(-(K[0] * lean + K[1] * rr + K[2] * ym + K[3] * yd), -force, force)
        d.ctrl[:] = [F, 0.0]; mujoco.mj_step(model, d)
        ys.append(ym); Fs.append(F)
        leans.append(np.degrees(np.arcsin(np.clip(lean, -1, 1))))
        if 1 - 2 * (q[4] ** 2 + q[5] ** 2) < 0.7:
            fell = k; break
    ys = np.array(ys); Fs = np.array(Fs); leans = np.array(leans)
    alive = fell < 0
    i0 = min(500, len(ys))
    yss, Fss = ys[i0:], Fs[i0:]
    m = dict(
        alive=alive, survived_s=(fell if fell >= 0 else T) * M.DT,
        lean_rms_deg=float(np.sqrt(np.mean(leans ** 2))) if len(leans) else 99,
        mass_limit_pct=float(np.mean(np.abs(yss) > 0.95 * stroke)) * 100 if len(yss) else 100,
        mass_freq_hz=_freq(yss, M.DT),
        mass_jerk=_jerk_rms(yss, M.DT),
        force_sat_pct=float(np.mean(np.abs(Fss) > 0.99 * force)) * 100 if len(Fss) else 100,
    )
    score = 0
    if alive:
        for key in ("mass_freq_hz", "mass_limit_pct", "force_sat_pct", "mass_jerk", "lean_rms_deg"):
            score += m[key] <= HUMAN[key]
    m["score"] = score
    return m


if __name__ == "__main__":
    MASS = [2.0, 4.0, 8.0]
    STROKE = [0.15, 0.30]
    FORCE = [20.0, 40.0, 80.0]
    print(f"v={V} 직진, LQR per-config. 사람다움 점수/5 (생존 전제).")
    print(f"{'mass':>5} {'stroke':>6} {'force':>5} | {'생존':>6} {'freq':>5} {'한계%':>5} "
          f"{'포화%':>5} {'jerk':>6} {'leanRMS':>7} | 점수")
    best = []
    for st in STROKE:
        for ms in MASS:
            for fo in FORCE:
                m = run(ms, st, fo)
                mark = " ★" if m["score"] >= 4 else ""
                surv = "∞" if m["alive"] else f"{m['survived_s']:.0f}s"
                print(f"{ms:5.0f} {st:6.2f} {fo:5.0f} | {surv:>6} "
                      f"{m['mass_freq_hz']:5.1f} {m['mass_limit_pct']:5.0f} "
                      f"{m['force_sat_pct']:5.0f} {m['mass_jerk']:6.0f} {m['lean_rms_deg']:7.1f} "
                      f"| {m['score']}/5{mark}")
                if m["score"] >= 4:
                    best.append((ms, st, fo, m["score"]))
    print("\n사람다움 ≥4/5 config:", best if best else "없음 (2kg급으로는 human-like 불가)")
