"""모델오차 스윕 결과 집계 — 컨트롤러 × 오차레벨 생존율 + paired 검정.

같은 seed 를 쓰므로 컨트롤러 비교는 반드시 paired(McNemar)로. 40 seeds 에서 2-3pp 차이는
unpaired 로 보면 노이즈에 묻히지만 paired 로는 판정 가능한 경우가 많다.

사용:  python src/mm/mm_robust_table.py [--axis pred|param] [--sev s0.5]
"""
import argparse
import glob
import json
import os
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "results" / "envelopes"


def load(path):
    d = json.load(open(path))
    return d["config"], {(r["sev"], r["delay_ms"], r["seed"]): bool(r["survived"])
                         for r in d["runs"]}


def mcnemar(x, y):
    """양측 exact McNemar. 반환 (x만 생존, y만 생존, p)."""
    k = set(x) & set(y)
    a = sum(1 for c in k if x[c] and not y[c])
    b = sum(1 for c in k if y[c] and not x[c])
    n = a + b
    if n == 0:
        return a, b, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)
    return a, b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("pred", "param"), default="pred")
    ap.add_argument("--sev", default="s0.5")
    args = ap.parse_args()

    nominal = {"base": ENV / "envelope_lqr_klat.json",
               "smith4": ENV / "envelope_lqr_smith4.json"}
    delays = [0, 4, 8, 12, 16, 20]
    rows = {}
    for ctrl, p in nominal.items():
        if p.exists():
            rows[(ctrl, 0.0)] = load(p)[1]
    for p in sorted(glob.glob(str(ENV / "robust" / f"*_{args.axis}*.json"))):
        stem = Path(p).stem.replace("envelope_", "")
        ctrl, lv = stem.split(f"_{args.axis}")
        rows[(ctrl, float(lv.split("_")[0]))] = load(p)[1]

    ctrls = sorted({c for c, _ in rows})
    levels = sorted({l for _, l in rows})
    print(f"=== {args.sev}, survival % by {args.axis}-error level (40 seeds) ===")
    print(f"{'ctrl':8} {'err':>6} | " + " ".join(f"{d:>5}" for d in delays) + " |   all")
    for c in ctrls:
        for l in levels:
            g = rows.get((c, l))
            if g is None:
                continue
            cells = []
            for d in delays:
                s = [v for (sv, dl, _), v in g.items() if sv == args.sev and dl == d]
                cells.append(f"{100*sum(s)/len(s):5.0f}" if s else "    -")
            allv = [v for (sv, _, _), v in g.items() if sv == args.sev]
            print(f"{c:8} {l:6.0%} | " + " ".join(cells) +
                  f" | {100*sum(allv)/len(allv):5.1f}")
        print()

    print("=== paired McNemar: smith4 vs base at each error level (all cells) ===")
    for l in levels:
        a, b = rows.get(("smith4", l)), rows.get(("base", l))
        if a is None or b is None:
            continue
        x, y, p = mcnemar(a, b)
        verdict = ("smith4 better" if x > y else "base better") if p < 0.05 else "n.s."
        print(f"  err {l:5.0%}: smith4-only {x:4}, base-only {y:4}, p={p:9.2e}  {verdict}")


if __name__ == "__main__":
    main()
