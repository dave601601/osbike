"""결합(지형×지연) 생존율 맵 렌더 — mm_envelope.py 가 만든 JSON 을 그림.

사용:  python src/mm/mm_plot_envelope.py [results/envelopes/envelope_lqr.json] [out.png]
데이터는 하드코딩하지 않는다 — 맵의 출처는 항상 JSON(프로토콜 포함)이어야 재현/비교 가능.
"""
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]           # 레포 루트 (CWD 무관)
SRC = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/envelopes/envelope_lqr.json")
OUT = (sys.argv[2] if len(sys.argv) > 2 else
       str(ROOT / "results/figures" / (Path(SRC).stem + ".png")))
doc = json.load(open(SRC))
cfg, runs = doc["config"], doc["runs"]
delays = cfg["delays_ms"]
sevs = cfg["severities"]
labels = [f'{s["label"]}\n{s["slope_deg"]:g}deg / mu{s["mu"]:g} / {s["bump_cm"]:g}cm'
          for s in sevs]
data = np.zeros((len(sevs), len(delays)))
for i, s in enumerate(sevs):
    for j, dms in enumerate(delays):
        sel = [r for r in runs if r["sev"] == s["label"] and r["delay_ms"] == dms]
        data[i, j] = 100.0 * sum(r["survived"] for r in sel) / max(len(sel), 1)

green = LinearSegmentedColormap.from_list(
    "s", ["#f2f4f3", "#d7ece0", "#8fd0ab", "#3f9d6b", "#1c6b43"])
fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=150)
fig.subplots_adjust(top=0.86, bottom=0.16)
im = ax.imshow(data, cmap=green, vmin=0, vmax=100, aspect="auto")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                color="#ffffff" if v >= 55 else "#2b2f2c")
ax.set_xticks(range(len(delays))); ax.set_xticklabels(delays)
ax.set_yticks(range(len(sevs))); ax.set_yticklabels(labels)
ax.set_xlabel(f"Actuator delay [ms]  ({cfg['ctrl_hz']} Hz control)")
ax.set_ylabel("Terrain severity  (slope / mu / bump)")
comp = cfg.get("delay_comp", "base")
who = ("RL residual" if comp == "rl" else
       "LQR" if comp == "base" else f"LQR + {comp}")
fig.suptitle(f"{who} combined survival rate:  terrain x delay", y=0.965)
tag = ("" if comp == "base" else
       ", ctrl: LQR base + learned residual" if comp == "rl" else
       f", delay-comp: {comp} (exact model)")
ax.set_title(f"moving-mass, v={cfg['v_target']:g}, {cfg['turn_deg']:g} deg turn, "
             f"{cfg['horizon_s']:g} s hold, {cfg['seeds']} seeds{tag}",
             color="#555", pad=8)
ax.set_xticks(np.arange(-.5, len(delays), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(sevs), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="minor", length=0)
xline = np.searchsorted(delays, 10) - 0.5          # 실차 지연대(>=10ms) 경계
ax.axvline(xline, color="#c0392b", lw=1.6, ls=(0, (4, 2)))
# 캡션도 컨트롤러별로. base 의 "LQR collapses" 문구를 smith4/RL 맵에 그대로 쓰면 거짓말이 됨
left, right = (("LQR robust", "realistic HW delay: LQR collapses = RL / delay-aware target")
               if comp == "base" else
               ("low delay", "realistic HW delay  (compare vs LQR base map)"))
fig.text(0.30, 0.045, left, color="#1c6b43", ha="center")
fig.text(0.66, 0.045, right, color="#c0392b", ha="center")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(f"{cfg['horizon_s']:g} s survival [%]")
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}  (from {SRC}, {cfg['seeds']} seeds)")
