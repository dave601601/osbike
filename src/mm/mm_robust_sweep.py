"""모델오차 축 스윕 — mm_envelope 를 오차 레벨 × 컨트롤러로 반복 호출.

docs 의 "smith4 는 모델오차 ±5% 에서 무보상보다 나빠진다" 표(10 seeds, s0.5, 12/16ms,
애드혹)를 **커밋된 코드 + 40 seeds + 전 격자**로 재현·확장한다. 그 표가 RL 명분의
핵심 근거인데 재현 경로가 없었음.

두 축은 서로 다른 질문이다 (mm_envelope.run_one 참고):
  --pred-err   컨트롤러가 믿는 모델만 틀림, 플랜트 명목. "예측기 모델오차" (docs 표)
  --param-err  플랜트가 흔들림, 컨트롤러는 명목 모델. "sim-to-real" 프록시.
               RL 학습 DR(mass_pct/gain_pct)이 겨냥한 바로 그 축 = 3-way 비교용.

사용:
  python src/mm/mm_robust_sweep.py --axis pred  --ctrl base smith4
  python src/mm/mm_robust_sweep.py --axis param --ctrl base smith4 --levels 0.05 0.1 0.2
  python src/mm/mm_robust_sweep.py --axis param --ctrl rl --policy ckpt/x/params_00399.pkl
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results" / "envelopes" / "robust"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("pred", "param"), required=True)
    ap.add_argument("--ctrl", nargs="+", default=["base", "smith4"])
    ap.add_argument("--levels", nargs="+", type=float,
                    default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed-from", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--policy", default="")
    ap.add_argument("--res-scale-eval", type=float, default=None)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    flag = "--pred-err" if args.axis == "pred" else "--param-err"
    t0 = time.time()
    jobs = [(c, lv) for c in args.ctrl for lv in args.levels]
    for i, (ctrl, lv) in enumerate(jobs, 1):
        tag = f"{ctrl}_{args.axis}{lv:g}"
        if args.seed_from:
            tag += f"_s{args.seed_from}"
        out = OUTDIR / f"envelope_{tag}.json"
        if out.exists():
            print(f"[{i}/{len(jobs)}] skip {tag} (exists)", flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "src/mm/mm_envelope.py"),
               "--seeds", str(args.seeds), "--seed-from", str(args.seed_from),
               "--workers", str(args.workers), "--ctrl", ctrl,
               flag, str(lv), "--out", str(out)]
        if ctrl == "rl":
            cmd += ["--policy", args.policy]
            if args.res_scale_eval is not None:
                cmd += ["--res-scale-eval", str(args.res_scale_eval)]
        print(f"[{i}/{len(jobs)}] {tag}  ({time.time()-t0:.0f}s elapsed)", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}\n{r.stderr[-1500:]}", flush=True)
            continue
        # mm_envelope 의 요약표를 그대로 흘려보냄 (마지막 8줄)
        print("\n".join(r.stdout.strip().splitlines()[-8:]), flush=True)
    print(f"\nall done in {time.time()-t0:.0f}s -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
