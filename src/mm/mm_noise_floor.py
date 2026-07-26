"""하네스 노이즈 바닥 측정 — "컨트롤러를 안 바꾸고 게인만 미세하게 흔들면 맵이 얼마나 움직이나".

동기: s0.5 에서 게인 0.5% 변화가 30s 완주를 3.7s 낙하로 뒤집는 seed 가 다수 관측됐다
(선회 직후 t~3s 집중 낙하). 그렇다면 base/smith4/RL 사이의 총합 차이(11~79런)가
컨트롤러 품질인지 knife-edge 추첨인지 구분해야 한다.

판정: jitter 스프레드 >= 컨트롤러 간 차이  →  그 비교는 현재 seed 수로 판정 불가.

사용:  python src/mm/mm_noise_floor.py
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "results" / "envelopes"


def runs(p):
    d = json.load(open(p))
    return {(r["sev"], r["delay_ms"], r["seed"]): bool(r["survived"]) for r in d["runs"]}


def mcnemar(x, y):
    k = set(x) & set(y)
    a = sum(1 for c in k if x[c] and not y[c])
    b = sum(1 for c in k if y[c] and not x[c])
    n = a + b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
    return a, b, p


def tot(g):
    return sum(g.values())


def main():
    base = runs(ENV / "envelope_lqr_klat.json")
    ref = {"base(committed)": base}
    for n, f in (("smith4", "envelope_lqr_smith4.json"),
                 ("RL v2@0.5", "envelope_rl_v2_half.json")):
        if (ENV / f).exists():
            ref[n] = runs(ENV / f)

    print("=== controller totals (of 960) ===")
    for n, g in ref.items():
        print(f"  {n:16} {tot(g):4}")

    print("\n=== noise floor: identical controller, gains jittered by +-x ===")
    print("  flips = runs whose outcome changed. 총합 차이가 작아도 flip 이 많으면,")
    print("  개별 셀 비교는 추첨이고 총합만 안정적이라는 뜻이다.")
    print(f"  {'jitter':>8} {'total':>6} {'d':>5} {'flips(+/-)':>14} {'sd~sqrt(n)':>11}  paired p")
    sds = []
    for j in ("0.005", "0.01", "0.02", "0.05"):
        f = ENV / "robust" / f"envelope_base_jitter{j}.json"
        if not f.exists():
            continue
        g = runs(f)
        a, b, p = mcnemar(g, base)
        # flip 이 대칭(부호 무작위)이면 총합 차이의 SD = sqrt(불일치 수)
        sd = (a + b) ** 0.5
        sds.append(sd)
        print(f"  {float(j):8.1%} {tot(g):6} {tot(g)-tot(base):+5} "
              f"{a+b:7} ({a}/{b}) {sd:11.1f}  p={p:.2f}")

    if sds:
        # 노이즈 바닥은 **가장 작은** 섭동에서 읽는다. 큰 jitter(5%)는 더 이상 "무시할
        # 만한 교란"이 아니라 실제 성능 저하(총합 -100, flip 28/128 로 편향)라서,
        # 그걸 노이즈라고 부르면 노이즈를 과대평가하게 된다.
        sd = sds[0]
        print(f"\n  -> 총합의 노이즈 1-sigma ~= {sd:.0f} runs (최소 섭동 기준).")
        print(f"     개별 런은 최대 {max(int(s*s) for s in sds)}개가 뒤집힌다.")
        print("     jitter 가 커지면 flip 이 한쪽으로 쏠린다 = 게인이 국소최적이라는 뜻")
        print("     (±5% 게인오차만으로 총합 -100). 그건 노이즈가 아니라 실제 저하.")
        print("\n=== controller differences vs that noise floor ===")
        names = [n for n in ref if n != "base(committed)"]
        pairs = [(n, "base", ref[n], base) for n in names]
        if len(names) == 2:
            pairs.append((names[1], names[0], ref[names[1]], ref[names[0]]))
        for na, nb, ga, gb in pairs:
            d = tot(ga) - tot(gb)
            a, b, p = mcnemar(ga, gb)
            sig = abs(d) / sd if sd else float("inf")
            verdict = ("REAL" if p < 0.01 and sig > 2 else
                       "within noise" if p > 0.05 else "marginal")
            print(f"  {na:12} vs {nb:8}: {d:+5} runs = {sig:4.1f} sigma  "
                  f"paired p={p:8.2e}  -> {verdict}")


if __name__ == "__main__":
    main()
