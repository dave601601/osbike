"""무게추 motion realism + 균형 성능 공통 평가. LQR/MPC/RL 모든 컨트롤러 채점.

'사람다움(human-likeness)' 관점: hands-free 라이더는 무게추(=CoM)를 저주파·저포화로
살살 쓴다. LQR은 약한 액추에이터 탓에 2Hz·풀스트로크·bang-bang으로 혹사 → 이 모듈이
그 차이를 수치로 드러낸다.

지표
  balance:   survived_s, alive, lean_rms/max
  realism:   mass_amp_cm, mass_peak_cm, mass_limit_pct, mass_freq_hz,
             mass_jerk(무차원), force_sat_pct, force_rms
  steering:  yaw_final (yaw_ref_fn 준 경우)

사용:  python mm_metrics.py                 # LQR vs 부드러운 PD 리포트카드
       from mm_metrics import evaluate, report
"""
import numpy as np
import mujoco
import mm_model as M
import mm_controller as C

# 사람 hands-free 대략 기준 (부드러운 CoM 이동). 넘으면 '비현실적'.
HUMAN = dict(mass_freq_hz=1.0, mass_limit_pct=10.0, force_sat_pct=20.0,
             mass_jerk=10.0, lean_rms_deg=5.0)

_SLIDE_J = mujoco.mj_name2id(M.m, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
STROKE = float(M.m.jnt_range[_SLIDE_J][1])          # 0.15
F_LIM = float(M.CTRL_HI[M.A_SLIDE])                 # 20


def _freq(sig, dt):
    """평균교차 기반 지배 주파수 [Hz]."""
    if len(sig) < 8:
        return 0.0
    zc = np.sum(np.diff(np.sign(sig - sig.mean())) != 0)
    return zc / 2.0 / (len(sig) * dt)


def _jerk_rms(y, dt):
    """무게추 jerk RMS [m/s³]. 연속운동 부드러움 지표(작을수록 부드러움).
    사람 CoM(수cm, <1Hz) ≈ O(0.1); LQR bang-bang(±11cm,2Hz) ≈ O(100)."""
    if len(y) < 4:
        return 0.0
    v = np.gradient(y, dt); a = np.gradient(v, dt); j = np.gradient(a, dt)
    return float(np.sqrt(np.mean(j ** 2)))


def evaluate(gains: C.Gains, v_target=2.0, pert_deg=0.5, T=5000,
             yaw_ref_fn=None, slew_dps=3.0, model=None, warmup=500):
    """컨트롤러 하나를 롤아웃하고 지표 dict 반환. 정상상태 통계는 warmup 이후만."""
    m = model or M.m
    d = mujoco.MjData(m)
    a = np.radians(pert_deg) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0, 0]
    d.qvel[0] = v_target
    d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = v_target / M.WHEEL_R
    mujoco.mj_forward(m, d)
    integ = 0.0; yref = 0.0; slew = np.radians(slew_dps)
    ys, Fs, leans, yaws = [], [], [], []
    fell = -1
    for k in range(T):
        if yaw_ref_fn is not None:
            yref += float(np.clip(yaw_ref_fn(k * M.DT) - yref, -slew * M.DT, slew * M.DT))
        ctrl, integ, st = C.controller(d, gains, integ, v_target, M.DT, yaw_ref=yref)
        d.ctrl[:] = np.asarray(ctrl)
        mujoco.mj_step(m, d)
        ys.append(float(d.qpos[M.Q_SLIDE])); Fs.append(float(ctrl[0]))
        leans.append(np.degrees(np.arcsin(np.clip(float(st.lean), -1, 1))))
        yaws.append(np.degrees(float(st.yaw)))
        if 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2) < 0.7:
            fell = k
            break
    ys = np.array(ys); Fs = np.array(Fs); leans = np.array(leans)
    survived_s = (fell if fell >= 0 else T) * M.DT
    i0 = min(warmup, max(len(ys) - 1, 0))
    yss, Fss = ys[i0:], Fs[i0:]
    n = max(len(yss), 1)
    return dict(
        survived_s=survived_s, alive=fell < 0,
        lean_rms_deg=float(np.sqrt(np.mean(leans ** 2))) if len(leans) else 0.0,
        lean_max_deg=float(np.abs(leans).max()) if len(leans) else 0.0,
        mass_amp_cm=float(np.std(yss)) * 100 if len(yss) else 0.0,
        mass_peak_cm=float(np.abs(yss).max()) * 100 if len(yss) else 0.0,
        mass_limit_pct=float(np.mean(np.abs(yss) > 0.95 * STROKE)) * 100,
        mass_freq_hz=_freq(yss, M.DT),
        mass_jerk=_jerk_rms(yss, M.DT),
        force_sat_pct=float(np.mean(np.abs(Fss) > 0.99 * F_LIM)) * 100,
        force_rms=float(np.sqrt(np.mean(Fss ** 2))) if len(Fss) else 0.0,
        yaw_final_deg=float(yaws[-1]) if yaws else 0.0,
    )


def human_score(mtr):
    """사람 기준 통과 개수 / 5 (survived 전제). realism 종합."""
    keys = ["mass_freq_hz", "mass_limit_pct", "force_sat_pct", "mass_jerk", "lean_rms_deg"]
    return sum(mtr[k] <= HUMAN[k] for k in keys), len(keys)


def report(name, mtr):
    """한 컨트롤러 리포트카드 (사람 기준 위반은 ✗ 표시)."""
    def flag(key):
        return "✗" if mtr[key] > HUMAN[key] else "✓"
    passed, total = human_score(mtr)
    print(f"── {name} ──  [사람다움 {passed}/{total}{' + 낙하' if not mtr['alive'] else ''}]")
    print(f"  balance : 생존 {mtr['survived_s']:5.1f}s{'(∞)' if mtr['alive'] else ''}  "
          f"lean RMS {mtr['lean_rms_deg']:.2f}°{flag('lean_rms_deg')} / max {mtr['lean_max_deg']:.1f}°")
    print(f"  realism : 진폭 {mtr['mass_amp_cm']:4.1f}cm  peak {mtr['mass_peak_cm']:4.1f}cm  "
          f"한계접촉 {mtr['mass_limit_pct']:4.0f}%{flag('mass_limit_pct')}  "
          f"주파수 {mtr['mass_freq_hz']:.1f}Hz{flag('mass_freq_hz')}")
    print(f"          : jerk {mtr['mass_jerk']:6.1f} m/s³{flag('mass_jerk')}  "
          f"힘포화 {mtr['force_sat_pct']:3.0f}%{flag('force_sat_pct')}  "
          f"힘RMS {mtr['force_rms']:.1f}N")


if __name__ == "__main__":
    import json
    G_lqr = C.Gains(**json.load(open(M.PARAMS / "mm_lqr_gains.json")))
    # 부드러운 소게인 PD (balance_mass 형식: K=[kp,kd,ky,k_ydot]). 중속에서만 유효.
    G_pd = C.Gains(30.0, 8.0, 5.0, 0.5, 2.0, 0.5)

    print("사람 기준: 주파수<1Hz, 한계접촉<10%, 힘포화<20%, jerk<10m/s³, lean RMS<5°\n")
    report("LQR @ v=2 (현재 gains)", evaluate(G_lqr, v_target=2.0))
    print()
    report("LQR @ v=6", evaluate(G_lqr, v_target=6.0))
    print()
    report("소게인 PD @ v=2 (부드럽게 시도)", evaluate(G_pd, v_target=2.0))
    print()
    report("소게인 PD @ v=6 (self-steering 활용)", evaluate(G_pd, v_target=6.0))
