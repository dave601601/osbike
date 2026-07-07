"""결합(지형×지연) 생존율 엔벨로프 — 재현 가능한 공용 평가 하네스.

docs/progress/mm.md 의 결합×지연 stress-test 를 커밋된 프로토콜로 고정한 것.
LQR(그리고 이후 RL 정책)을 같은 지형 seed·같은 판정으로 채점해 baseline 맵과 겹친다.
결과는 JSON(기본 envelope_lqr.json) → mm_plot_envelope.py 가 그림을 그림.

프로토콜 (컨트롤러 무관 공통):
  - v_target=1.5 m/s, +30° 좌선회(t=2s 시작, yaw_ref 슬루 3°/s), 30s 생존(up_z>0.7)
  - 물리 250Hz / 제어 50Hz ZOH / 액추에이터 지연 = 명령을 n스텝 뒤에 적용.
    지연 격자는 물리스텝(4ms)의 정수배만 사용 — 비정수 지연(5/10/15ms)은 조용히
    양자화되므로 금지 (구 3-seed 맵의 라벨과 다른 이유).
  - 지형 severity = (옆경사°, μ, 범프cm). 옆경사는 중력 y-틸트(내리막 = -y = 오른쪽),
    μ는 바닥+양 바퀴 모두 설정 (MuJoCo 접촉 마찰 = 두 geom 의 element-wise max —
    바닥만 낮추면 바퀴 1.4가 이겨서 무효), 범프는 hfield x∈[-2,78] y∈[±45m]
    (y폭이 좁으면 경사 드리프트로 지형 밖 추락 = 가짜 낙하, docs 교훈 #4).
  - seed → 범프 실현(RandomState(seed), gaussian smooth). 초기섭동 0.5° 고정.

사용:
  python mm_envelope.py                          # LQR(60N), seeds 20 → envelope_lqr.json
  python mm_envelope.py --seeds 5 --workers 4 --out t.json
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # 평가는 배포경로(CPU)로만
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse, json, sys, time
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter

import mm_model as M
import mm_lqr
import mm_controller as C
import mm_delay as D

# --- 프로토콜 상수 (바꾸면 baseline 과 비교 불가 — 바꿀 땐 파일명을 바꿀 것) ---
V_TARGET   = 1.5
TURN_DEG   = 30.0
TURN_T     = 2.0
SLEW_DPS   = 3.0
HORIZON_S  = 30.0
CTRL_EVERY = 5                      # 250Hz 물리 / 5 = 50Hz 제어
FORCE      = 60.0                   # 슬라이더 힘한계 + LQR 재설계 (힘임계 ≥50N)
STROKE     = 0.15
PERT_DEG   = 0.5
SEVERITIES = [                      # (라벨, 옆경사°, μ, 범프cm) — 구 맵과 동일 축
    ("s0.25", 1.0, 1.1, 1.0),
    ("s0.5",  2.0, 0.9, 2.0),
    ("s0.75", 3.0, 0.6, 4.0),
    ("s1.0",  4.0, 0.4, 5.0),
]
DELAYS_MS  = [0, 4, 8, 12, 16, 20]  # 물리스텝(4ms) 정수배만
HF_X, HF_Y = 40.0, 60.0             # hfield 반폭 [m] — y 넓게(경사 드리프트, k_lat=0)
HF_NR, HF_NC, HF_SIGMA = 1800, 60, 1.2   # mm_render 와 동일 해상도/스무딩 (y 15칸/m)


def build_model(slope_deg, mu, bump_cm, seed):
    """지형 variant 모델: hfield 범프 + 중력 틸트(옆경사) + 접촉 μ + 60N 슬라이더."""
    zt = max(bump_cm / 100.0, 1e-3)
    xml = open(M.XML).read()
    xml = xml.replace(
        "  <worldbody>",
        f'  <asset><hfield name="bumps" nrow="{HF_NR}" ncol="{HF_NC}" '
        f'size="{HF_X} {HF_Y} {zt} 0.1"/></asset>\n  <worldbody>', 1)
    xml = xml.replace(
        '<geom name="ground" type="plane" size="0 0 0.05" rgba="0.55 0.55 0.55 1"/>',
        f'<geom name="ground" type="hfield" hfield="bumps" pos="{HF_X - 2} 0 0"/>')
    xml = xml.replace('ctrlrange="-20 20"', f'ctrlrange="-{FORCE} {FORCE}"')
    m = mujoco.MjModel.from_xml_string(xml)
    h = gaussian_filter(np.random.RandomState(seed).rand(HF_NR, HF_NC), sigma=HF_SIGMA)
    m.hfield_data[:] = ((h - h.min()) / (h.max() - h.min())).ravel()
    for gname in ("ground", "rear_geom", "front_geom"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
        m.geom_friction[gid, 0] = mu
    phi = np.radians(slope_deg)
    m.opt.gravity[:] = [0.0, -9.81 * np.sin(phi), -9.81 * np.cos(phi)]
    return m, zt


def run_one(task):
    """한 (severity, delay, seed[, gains 오버라이드]) 롤아웃 → 결과 dict. 결정론(CPU)."""
    sev_i, delay_ms, seed, *rest = task
    over = dict(rest[0]) if rest else {}
    variant = over.pop("ctrl", "base")      # base | smith4 | smith6 | rl
    pol_path = over.pop("policy", "")
    label, slope, mu, bump = SEVERITIES[sev_i]
    m, zt = build_model(slope, mu, bump, seed)
    pred = (D.Predictor.smith4(m) if variant == "smith4"
            else D.Predictor.smith6() if variant == "smith6" else None)
    rlc = None
    if variant == "rl":
        import mm_policy as MP              # 늦은 import (고전 채점 경로 무부담)
        rlc = MP.CpuController(pol_path)
    K, _, _ = mm_lqr.design(m, y_max=STROKE, F_max=FORCE)
    M.CTRL_HI[M.A_SLIDE], M.CTRL_LO[M.A_SLIDE] = FORCE, -FORCE
    base = json.load(open("mm_lqr_gains.json"))
    base.update(over)                       # 어블레이션용 (--k-lat 0 등)
    G = C.Gains(float(K[0]), float(K[1]), float(K[2]), float(K[3]),
                base["kp_v"], base["ki_v"], base["k_yaw"], base["kd_yaw"],
                base["lean_max"], base.get("ki_yaw", 0.0),
                base.get("k_lat", 0.0), base.get("kd_lat", 0.0),
                base.get("yaw_corr_max", 0.21), base.get("lat_slew", 0.026))

    d = mujoco.MjData(m)
    a = np.radians(PERT_DEG) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
    d.qpos[2] = 0.30 + zt
    d.qvel[0] = V_TARGET
    d.qvel[M.V_REAR] = d.qvel[M.V_FRONT] = V_TARGET / M.WHEEL_R
    mujoco.mj_forward(m, d)

    n_steps = int(HORIZON_S / M.DT)
    n_delay = delay_ms // int(M.DT * 1000)
    assert n_delay * int(M.DT * 1000) == delay_ms, "지연은 4ms 정수배만"
    ctrl_dt = CTRL_EVERY * M.DT
    slew = np.radians(SLEW_DPS) * ctrl_dt
    yaw_goal = np.radians(TURN_DEG)
    cs, yref = C.CtrlState(), 0.0
    px, py = 0.0, 0.0                       # 기준경로 앵커 (명령 방위로 v_target 적분)
    pending, cur = [], np.zeros(2)
    min_up_z, status, t_end = 1.0, "ok", HORIZON_S
    ct, ct_max = 0.0, 0.0                   # crosstrack (코스 유지 지표 — 생존만으론
    for step in range(n_steps):             #  "내리막 항복" cheat 을 못 잡음)
        t = step * M.DT
        # 릴리즈 pop 은 틱 앞뒤 두 번: 앞 = 예측용 cur 최신화(안 하면 한 주기 지연에서
        # 두 틱 전 명령으로 예측하는 off-by-one), 뒤 = n_delay=0 즉시적용 유지.
        while pending and pending[0][0] <= step:
            cur = pending.pop(0)[1]
        if step % CTRL_EVERY == 0:
            tgt = yaw_goal if t >= TURN_T else 0.0
            yref += float(np.clip(tgt - yref, -slew, slew))
            if rlc is not None:
                ctrl = np.asarray(rlc.tick(d, yref, px, py, V_TARGET), dtype=float)
            else:
                path = (px, py, yref) if base.get("k_lat", 0.0) > 0 else None
                xp = None
                if pred is not None and n_delay > 0:
                    xp = pred.predict_seq(D.z6(C.read_state(d)),
                                          D.horizon_seq(step, cur, pending, n_delay))
                ctrl, cs, _ = C.controller(d, G, cs, V_TARGET, ctrl_dt,
                                           yaw_ref=yref, path=path, x_pred4=xp)
            px += V_TARGET * ctrl_dt * np.cos(yref)
            py += V_TARGET * ctrl_dt * np.sin(yref)
            ct = float(-(d.qpos[0] - px) * np.sin(yref)
                       + (d.qpos[1] - py) * np.cos(yref))
            ct_max = max(ct_max, abs(ct))
            pending.append((step + n_delay, np.asarray(ctrl, dtype=float)))
        while pending and pending[0][0] <= step:
            cur = pending.pop(0)[1]
        d.ctrl[:] = cur
        mujoco.mj_step(m, d)
        qw, qx, qy, qz = d.qpos[3:7]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        min_up_z = min(min_up_z, up_z)
        if up_z < 0.7:
            status, t_end = "fell", t
            break
        if step % 25 == 0 and not (-1.0 < d.qpos[0] < 2 * HF_X - 3
                                   and abs(d.qpos[1]) < HF_Y - 2.0):
            status, t_end = "edge", t     # 지형 이탈 = 판정 불가 (가짜 낙하 방지)
            break
    qw, qx, qy, qz = d.qpos[3:7]
    yaw_end = float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy),
                                          1 - 2 * (qy * qy + qz * qz))))
    return dict(sev=label, slope_deg=slope, mu=mu, bump_cm=bump,
                delay_ms=delay_ms, seed=seed, survived=(status == "ok"),
                status=status, t_end=round(float(t_end), 3), min_up_z=round(float(min_up_z), 4),
                x_end=round(float(d.qpos[0]), 2), y_end=round(float(d.qpos[1]), 2),
                ct_end=round(ct, 2), ct_max=round(ct_max, 2), yaw_end=round(yaw_end, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="envelope_lqr.json")
    ap.add_argument("--lean-max-deg", type=float, default=None,
                    help="heading lean_max 오버라이드 [deg] (어블레이션)")
    ap.add_argument("--k-lat", type=float, default=None,
                    help="lateral 외곽루프 게인 오버라이드 (0=끔, 어블레이션)")
    ap.add_argument("--ctrl", choices=("base", "smith4", "smith6", "rl"),
                    default="base",
                    help="smith*=지연보상 예측기, rl=학습 정책(--policy 필요)")
    ap.add_argument("--policy", default="", help="rl 채점용 체크포인트 경로")
    args = ap.parse_args()

    over = {}
    if args.lean_max_deg is not None:
        over["lean_max"] = float(np.radians(args.lean_max_deg))
    if args.k_lat is not None:
        over["k_lat"] = args.k_lat
    if args.ctrl != "base":
        over["ctrl"] = args.ctrl
    if args.ctrl == "rl":
        assert args.policy, "--ctrl rl 은 --policy <ckpt.pkl> 필요"
        over["policy"] = args.policy
    tasks = [(i, dms, s, over) for i in range(len(SEVERITIES))
             for dms in DELAYS_MS for s in range(args.seeds)]
    t0 = time.time()
    runs = []
    if args.workers <= 1:
        for k, task in enumerate(tasks):
            runs.append(run_one(task))
            print(f"  {k+1}/{len(tasks)} {runs[-1]}", file=sys.stderr)
    else:
        # spawn 필수: JAX는 멀티스레드라 fork 하면 자식이 잠긴 락을 물려받아 교착
        # (fork로 12워커 전부 jnp 첫 호출에서 멈춘 사고 있었음)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2)):
                runs.append(r)
                if (k + 1) % 20 == 0:
                    print(f"  {k+1}/{len(tasks)}  ({time.time()-t0:.0f}s)",
                          file=sys.stderr, flush=True)

    gj = json.load(open("mm_lqr_gains.json")); gj.update(over)
    cfg = dict(controller=f"LQR {FORCE:.0f}N + heading cascade + speed PI (50Hz ZOH)"
                          + (f" + {args.ctrl} 지연보상" if args.ctrl != "base" else ""),
               delay_comp=args.ctrl, policy=args.policy,
               lean_max_deg=round(float(np.degrees(gj["lean_max"])), 2),
               k_lat=gj.get("k_lat", 0.0), kd_lat=gj.get("kd_lat", 0.0),
               yaw_corr_max_deg=round(float(np.degrees(gj.get("yaw_corr_max", 0.35))), 1),
               v_target=V_TARGET, turn_deg=TURN_DEG, turn_t=TURN_T, slew_dps=SLEW_DPS,
               horizon_s=HORIZON_S, physics_hz=round(1 / M.DT), ctrl_hz=round(1 / (CTRL_EVERY * M.DT)),
               force_n=FORCE, stroke_m=STROKE, pert_deg=PERT_DEG, seeds=args.seeds,
               severities=[dict(zip(("label", "slope_deg", "mu", "bump_cm"), s))
                           for s in SEVERITIES],
               delays_ms=DELAYS_MS,
               hfield=dict(x_half=HF_X, y_half=HF_Y, nrow=HF_NR, ncol=HF_NC, sigma=HF_SIGMA))
    json.dump(dict(config=cfg, runs=runs), open(args.out, "w"), indent=1)

    # 요약표 (생존율 %)
    print(f"\n=== survival % ({args.seeds} seeds, {time.time()-t0:.0f}s) ===")
    print("sev      " + "".join(f"{d:>6}ms" for d in DELAYS_MS))
    for label, *_ in SEVERITIES:
        row = []
        for dms in DELAYS_MS:
            sel = [r for r in runs if r["sev"] == label and r["delay_ms"] == dms]
            edge = sum(r["status"] == "edge" for r in sel)
            pct = 100.0 * sum(r["survived"] for r in sel) / max(len(sel), 1)
            row.append(f"{pct:>7.0f}" + ("!" if edge else " "))
        print(f"{label:<9}" + "".join(row))
    ne = sum(r["status"] == "edge" for r in runs)
    if ne:
        print(f"경고: edge(지형이탈 판정불가) {ne}건 — '!' 표시 셀, HF_Y 늘려 재실행 요망")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
