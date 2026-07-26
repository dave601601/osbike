"""모델오차 축 3-way 그림 — base / smith4 / RL 의 교차를 보여준다.

이 프로젝트의 핵심 주장("RL 의 니치 = 모델오차 하의 강건성")을 한 장으로 요약하는 그림.
RL 은 학습 seed 4개의 평균 ± 범위로 그린다 — seed 분산(1σ~23런)이 하네스 노이즈(~8런)의
3배라, 단일 seed 곡선은 실제보다 정밀해 보인다.

사용:  python src/mm/mm_plot_robust.py [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "results" / "envelopes"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/figures/robustness_3way.png")
LEVELS = [0.0, 0.05, 0.10, 0.20]


def tot(p):
    d = json.load(open(p))
    return sum(bool(r["survived"]) for r in d["runs"])


def series():
    base, s4, rl = [], [], []
    for e in LEVELS:
        if e == 0.0:
            base.append(tot(ENV / "envelope_lqr_klat.json"))
            s4.append(tot(ENV / "envelope_lqr_smith4.json"))
            rl.append([tot(ENV / "rl" / f"envelope_rl_s{s}.json") for s in range(4)
                       if (ENV / "rl" / f"envelope_rl_s{s}.json").exists()])
        else:
            t = f"{e:.2f}"          # 파일명 규칙: param0.05 / param0.10 / param0.20
            base.append(tot(ENV / "robust" / f"envelope_base_param{t}.json"))
            s4.append(tot(ENV / "robust" / f"envelope_smith4_param{t}.json"))
            rl.append([tot(ENV / "robust" / f"envelope_rl_s{s}_param{t}.json")
                       for s in range(4)
                       if (ENV / "robust" / f"envelope_rl_s{s}_param{t}.json").exists()])
    return base, s4, rl


base, s4, rl = series()
x = np.array(LEVELS) * 100
rl_m = np.array([np.mean(v) for v in rl])
rl_lo = np.array([min(v) for v in rl])
rl_hi = np.array([max(v) for v in rl])

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax.plot(x, base, "o-", color="#888888", label="LQR baseline")
ax.plot(x, s4, "s-", color="#c0392b", label="+ Smith predictor (nominal model)")
ax.plot(x, rl_m, "^-", color="#1c6b43", label="RL residual (model-free, 4 seeds)")
ax.fill_between(x, rl_lo, rl_hi, color="#1c6b43", alpha=0.18, linewidth=0)

ax.set_xlabel("Plant parameter error [%]  (mass and actuator gain; controllers keep the nominal model)")
ax.set_ylabel("Survivals out of 960 runs")
ax.set_title("Where model-based control stops paying off", pad=10)
ax.grid(alpha=0.25)
ax.legend(loc="upper right")
ax.set_xlim(-1, 21)
ax.set_ylim(0, 520)

# 이 그림의 요점은 교차 하나다: 명목에선 smith4 가 위, ~5% 부터 RL 이 위.
ax.axvline(5.0, color="#999", lw=1.0, ls=(0, (4, 3)))
ax.annotate("crossover ~5%", xy=(5.0, 470), xytext=(5.8, 470), color="#555")
ax.annotate("model-based advantage gone:\nbase = smith4 (p=0.31 at 10%, 1.00 at 20%)",
            xy=(20, s4[-1]), xytext=(7.0, 90),
            arrowprops=dict(arrowstyle="->", color="#555"), color="#555")

fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print(f"saved {OUT}")
for i, e in enumerate(LEVELS):
    print(f"  err {e:5.0%}: base {base[i]:4}  smith4 {s4[i]:4}  "
          f"RL {rl_m[i]:6.0f} [{rl_lo[i]}-{rl_hi[i]}]  n={len(rl[i])}")
