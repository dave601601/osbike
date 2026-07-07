"""자립형 미니 PPO (순수 JAX) — mm_env 학습기.

optax/flax/brax 미설치 + jax 0.10 호환 리스크 회피를 위해 의존성 없이 구현:
MLP(직교 초기화) · 대각 가우시안(전역 log_std, 하한 클램프 = 엔트로피 붕괴 방지) ·
GAE(λ) · 클립 목적함수 · 수제 Adam + 전역 grad clip · 관측 러닝 정규화 ·
timeout/낙하 부트스트랩 구분(terminal_obs 사용).

사용:
  python mm_ppo.py --stage flat --iters 300            # cold start sanity
  python mm_ppo.py --stage task ...                    # 선회+속도 랜덤 (DR 없음)
  python mm_ppo.py --stage dr ...                      # 풀 DR (지연·경사·μ·질량·푸시)
  python mm_ppo.py --sweep                             # num_envs 처리량 스윕
체크포인트/로그: ckpt/<run>/{params_*.pkl, log.jsonl}
"""
import argparse, json, os, pickle, time
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # 8GB GPU 선할당 OOM 회피
import jax
import jax.numpy as jnp
import numpy as np

import mm_env as E

# ---------------- 신경망 (순수 JAX) ----------------

def _orth(rng, shape, scale):
    a = jax.random.normal(rng, shape)
    q, r = jnp.linalg.qr(a if shape[0] >= shape[1] else a.T)
    q = q * jnp.sign(jnp.diag(r))
    return scale * (q if shape[0] >= shape[1] else q.T)[:shape[0], :shape[1]]


def mlp_init(rng, sizes, out_scale):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        rng, k = jax.random.split(rng)
        s = out_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        ps.append((_orth(k, (a, b), s), jnp.zeros(b)))
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


LOG_STD_MIN, LOG_STD_MAX = -2.5, 0.5   # 하한 = 탐색 floor (BC/NLL 붕괴 논의 참조)


def init_params(rng, obs_dim, act_dim, hidden=(256, 256)):
    k1, k2 = jax.random.split(rng)
    return dict(pi=mlp_init(k1, (obs_dim, *hidden, act_dim), 0.01),
                v=mlp_init(k2, (obs_dim, *hidden, 1), 1.0),
                log_std=jnp.full((act_dim,), -1.0))


def dist(params, obs):
    mean = mlp(params["pi"], obs)
    log_std = jnp.clip(params["log_std"], LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def logp_of(mean, log_std, a):
    z = (a - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def entropy_of(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def value_of(params, obs):
    return mlp(params["v"], obs)[..., 0]

# ---------------- Adam + grad clip (수제) ----------------

def adam_init(params):
    z = jax.tree.map(jnp.zeros_like, params)
    return dict(m=z, v=jax.tree.map(jnp.zeros_like, params), t=jnp.array(0))


def adam_step(params, grads, opt, lr, clip=0.5, b1=0.9, b2=0.999, eps=1e-8):
    gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, clip / (gnorm + 1e-9))
    grads = jax.tree.map(lambda g: g * scale, grads)
    t = opt["t"] + 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, opt["v"], grads)
    mh = jax.tree.map(lambda x: x / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda x: x / (1 - b2 ** t), v)
    params = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps),
                          params, mh, vh)
    return params, dict(m=m, v=v, t=t), gnorm

# ---------------- 관측 정규화 ----------------

def norm_init(dim):
    return dict(mean=jnp.zeros(dim), var=jnp.ones(dim), count=jnp.array(1e-4))


def norm_update(st, batch):                      # batch: (B, dim)
    bm, bv, bc = batch.mean(0), batch.var(0), batch.shape[0]
    delta = bm - st["mean"]
    tot = st["count"] + bc
    mean = st["mean"] + delta * bc / tot
    m2 = st["var"] * st["count"] + bv * bc + delta ** 2 * st["count"] * bc / tot
    return dict(mean=mean, var=m2 / tot, count=tot)


def norm_apply(st, obs):
    return jnp.clip((obs - st["mean"]) / jnp.sqrt(st["var"] + 1e-8), -10., 10.)

# ---------------- 롤아웃 + 업데이트 ----------------

@partial(jax.jit, static_argnums=(5, 6))
def rollout(params, nrm, st, mxv, rng, T, dr):
    def one(carry, _):
        st, mxv, rng = carry
        obs = jax.vmap(E._obs)(st)
        nob = norm_apply(nrm, obs)
        rng, k = jax.random.split(rng)
        mean, log_std = dist(params, nob)
        a = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
        lp = logp_of(mean, log_std, a)
        val = value_of(params, nob)
        st, mxv, _, r, done, info = E.step(st, mxv, a, dr)
        out = dict(obs=obs, act=a, logp=lp, val=val, rew=r, done=done,
                   timeout=info["timeout"], tobs=info["terminal_obs"],
                   fin_ret=info["fin_ret"], fin_len=info["fin_len"],
                   fin_burnin=info["fin_burnin"])
        return (st, mxv, rng), out
    (st, mxv, rng), tr = jax.lax.scan(one, (st, mxv, rng), None, length=T)
    last_obs = jax.vmap(E._obs)(st)
    return st, mxv, rng, tr, last_obs


@jax.jit
def gae(params, nrm, tr, last_obs, gamma=0.99, lam=0.95):
    v_last = value_of(params, norm_apply(nrm, last_obs))
    v_term = value_of(params, norm_apply(nrm, tr["tobs"]))     # (T,N)
    vals, rews, done, tout = tr["val"], tr["rew"], tr["done"], tr["timeout"]
    v_next = jnp.concatenate([vals[1:], v_last[None]], 0)
    # done 스텝: 낙하→0, timeout→V(terminal_obs) 로 부트스트랩
    v_next = jnp.where(done, jnp.where(tout, v_term, 0.0), v_next)
    delta = rews + gamma * v_next - vals
    def back(adv, x):
        d, dn = x
        adv = d + gamma * lam * (1 - dn) * adv
        return adv, adv
    _, advs = jax.lax.scan(back, jnp.zeros_like(v_last),
                           (delta, done.astype(jnp.float32)), reverse=True)
    return advs, advs + vals


@partial(jax.jit, static_argnums=(6, 7))
def update(params, opt, nrm, batch, rng, lr, epochs, n_mb,
           clip_eps=0.2, vf_coef=0.5, ent_coef=0.005):
    B = batch["obs"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    batch = {**batch, "adv": adv, "nob": norm_apply(nrm, batch["obs"])}

    def loss_fn(p, mb):
        mean, log_std = dist(p, mb["nob"])
        lp = logp_of(mean, log_std, mb["act"])
        ratio = jnp.exp(lp - mb["logp"])
        l_pi = -jnp.mean(jnp.minimum(
            ratio * mb["adv"],
            jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * mb["adv"]))
        v = value_of(p, mb["nob"])
        l_v = 0.5 * jnp.mean((v - mb["ret"]) ** 2)
        ent = entropy_of(log_std)
        kl = jnp.mean(mb["logp"] - lp)
        frac = jnp.mean((jnp.abs(ratio - 1) > clip_eps).astype(jnp.float32))
        return l_pi + vf_coef * l_v - ent_coef * ent, (l_pi, l_v, ent, kl, frac)

    def epoch(carry, _):
        params, opt, rng = carry
        rng, k = jax.random.split(rng)
        idx = jax.random.permutation(k, B).reshape(n_mb, -1)
        def mb_step(carry, ix):
            params, opt = carry
            mb = jax.tree.map(lambda x: x[ix], batch)
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            params, opt, gn = adam_step(params, g, opt, lr)
            return (params, opt), (l, *aux, gn)
        (params, opt), stats = jax.lax.scan(mb_step, (params, opt), idx)
        return (params, opt, rng), stats
    (params, opt, rng), stats = jax.lax.scan(epoch, (params, opt, rng), None,
                                             length=epochs)
    return params, opt, jax.tree.map(lambda s: s.mean(), stats)

# ---------------- wandb ----------------

WANDB_ENTITY = "bsn012944-kaist-digital-humanities-and-social-sciences-g"
WANDB_PROJECT = "osbike"


def wandb_init(args, dr, run):
    """로그인 안 돼 있으면 offline 폴백 (나중에 `wandb sync wandb/offline-*` 로 업로드)."""
    if args.wandb == "off":
        return None
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    logged_in = bool(os.environ.get("WANDB_API_KEY"))
    try:
        logged_in = logged_in or "api.wandb.ai" in open(
            os.path.expanduser("~/.netrc")).read()
    except OSError:
        pass
    cfg = {k: v for k, v in vars(args).items() if k != "wandb"}
    cfg.update({f"dr_{k}": v for k, v in dr._asdict().items()},
               obs_dim=E.OBS_DIM, act_dim=E.ACT_DIM, ep_len_max=E.EP_LEN,
               ctrl_hz=round(1 / E.CTRL_DT))
    mode = "online" if logged_in else "offline"
    wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=run,
                    config=cfg, mode=mode)
    print(f"[wandb] {mode} ({wb.dir})", flush=True)
    return wb


# ---------------- 학습 루프 ----------------

STAGES = dict(
    flat=E.DR(),
    task=E.DR(turn_deg=45.0, v_lo=1.0, v_hi=2.0, pert_deg=1.0),
    dr=E.DR(delay_max=6, slope_deg=4.0, mu_lo=0.4, mu_hi=1.4, mass_pct=0.2,
            damp_hi=2.0, gain_pct=0.1, push_n=20.0, turn_deg=45.0,
            v_lo=1.0, v_hi=2.0, pert_deg=1.0),
)
STAGES["res"] = STAGES["dr"]._replace(res_scale=0.3)   # residual RL (base+잔차, 풀 DR)
# LQR-취약 도메인 집중 (fine-tune). v1(res_hard)은 모든 축 하한을 동시에 올려
# 평균 추첨이 s0.75×16ms 급(최강 고전도 0% = 물리한계)이 됨 → 성공률 0, 리턴 천장.
# v2: "이길 수 있는 hard" — 지연 하한만 유지(8ms+, base 가 지는 축), 나머지는
# 명목~hard 독립 범위로 결합 질량을 winnable 대역(s0.25-0.6×지연)에 배치.
STAGES["res_hard"] = STAGES["res"]._replace(delay_min=2, slope_lo=2.0,
                                            mu_hi=0.9, push_lo=15.0, push_n=35.0)
STAGES["res_hard2"] = STAGES["res"]._replace(delay_min=2, slope_lo=1.0,
                                             mu_hi=1.2, push_lo=5.0, push_n=30.0)
# residual 천장 어블레이션: res_scale=1.0 이면 clip(base+π)가 전 명령 공간을 커버
# (base 는 prior 로 유지, ±30% 표현력 제약만 제거). ep_len 정체 2회의 원인 판정용.
STAGES["res_full"] = STAGES["res_hard2"]._replace(res_scale=1.0)
# 최종 조합 (교훈 3개 반영): 프레임 스태킹(POMDP 처방) + 지연 0-24ms 전 구간 혼합
# (지연 하한 금지 — res_hard2 저지연 붕괴 교훈) + res_scale 1.0 (천장 제거).
STAGES["res_v2"] = STAGES["res"]._replace(res_scale=1.0)


def train(args):
    dr = STAGES[args.stage]
    run = args.run or f"{args.stage}_n{args.num_envs}"
    outdir = os.path.join("ckpt", run)
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "log.jsonl"), "a")
    rng = jax.random.PRNGKey(args.seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    params = init_params(k1, E.OBS_DIM, E.ACT_DIM)
    if args.init:
        with open(args.init, "rb") as f:
            saved = pickle.load(f)
        params, nrm = saved["params"], saved["nrm"]
        print(f"[init] {args.init}")
    else:
        nrm = norm_init(E.OBS_DIM)
    opt = adam_init(params)
    wb = wandb_init(args, dr, run)
    st, mxv, _ = E.reset(k2, args.num_envs, dr)
    T, N = args.T, args.num_envs
    t0 = time.time()
    ret_ema, len_ema = 0.0, 0.0
    t_prev = time.time()
    for it in range(args.iters):
        rng, kr, ku = jax.random.split(rng, 3)
        st, mxv, kr, tr, last_obs = rollout(params, nrm, st, mxv, kr, T, dr)
        adv, ret = gae(params, nrm, tr, last_obs)
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        batch = dict(obs=flat(tr["obs"]), act=flat(tr["act"]),
                     logp=flat(tr["logp"]), adv=flat(adv), ret=flat(ret))
        nrm = norm_update(nrm, batch["obs"])
        params, opt, stats = update(params, opt, nrm, batch, ku,
                                    args.lr, args.epochs, args.n_mb)
        l, lpi, lv, ent, kl, frac, gn = [float(x) for x in stats]
        steps = (it + 1) * T * N
        t_now = time.time()
        iter_s, t_prev = t_now - t_prev, t_now
        sps = steps / (t_now - t0)
        # 에피소드 통계 (이번 iteration 에서 끝난 에피소드들, burn-in 세대 제외 —
        # 초기 위상 랜덤화로 timeout 파도/선택편향 제거, mm_env._reset_one 주석)
        d = np.asarray(tr["done"]) & ~np.asarray(tr["fin_burnin"])
        metrics = {
            "policy/entropy": ent, "policy/clip_fraction": frac,
            "policy/approx_kl": kl,
            "policy/log_std_mean": float(np.mean(jax.device_get(params["log_std"]))),
            "training/critic_loss": lv,
            "training/critic_rmse": (2 * lv) ** 0.5,
            "training/policy_loss": lpi, "training/total_loss": l,
            "training/grad_norm": gn, "training/num_envs": N,
            "training/batch_size": T * N, "training/lr": args.lr,
            "timing/sps_overall": sps, "timing/sps_iter": T * N / iter_s,
            "timing/iter_seconds": iter_s, "timing/env_steps": steps,
        }
        if d.any():
            fr = np.asarray(tr["fin_ret"])[d]
            fl = np.asarray(tr["fin_len"])[d]
            suc = np.asarray(tr["timeout"])[d]        # 30s 완주 = 성공
            metrics.update({
                "episode/return_mean": float(fr.mean()),
                "episode/return_std": float(fr.std()),
                "episode/return_min": float(fr.min()),
                "episode/return_max": float(fr.max()),
                "episode/length_mean": float(fl.mean()),
                "episode/success_rate": float(suc.mean()),
                "episode/count": int(d.sum()),
            })
            ret_ema = 0.9 * ret_ema + 0.1 * float(fr.mean())
            len_ema = 0.9 * len_ema + 0.1 * float(fl.mean())
        if wb is not None:
            wb.log(metrics, step=steps)
        rec = dict(it=it, steps=steps, ep_ret=round(ret_ema, 1),
                   ep_len=round(len_ema, 1), ent=round(ent, 3), kl=round(kl, 5),
                   clipfrac=round(frac, 3), v_loss=round(lv, 2), sps=int(sps))
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.log_every == 0:
            print(rec, flush=True)
        if it % args.ckpt_every == 0 or it == args.iters - 1:
            with open(os.path.join(outdir, f"params_{it:05d}.pkl"), "wb") as f:
                pickle.dump(dict(params=jax.device_get(params),
                                 nrm=jax.device_get(nrm), it=it,
                                 stage=args.stage, dr=dr._asdict()), f)
    if wb is not None:
        wb.finish()
    print(f"done: {outdir}  ep_ret={ret_ema:.1f} ep_len={len_ema:.1f}")


def sweep(args):
    """num_envs 처리량 스윕 — 사용자 요청: 최대로 띄울 수 있는 지점 탐색."""
    rng = jax.random.PRNGKey(0)
    params = init_params(rng, E.OBS_DIM, E.ACT_DIM)
    nrm = norm_init(E.OBS_DIM)
    dr = STAGES["dr"]                      # DR 켠 상태로 재야 실전 SPS
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        try:
            st, mxv, _ = E.reset(jax.random.PRNGKey(1), n, dr)
            r = jax.random.PRNGKey(2)
            _, _, _, tr, _ = rollout(params, nrm, st, mxv, r, args.T, dr)
            jax.block_until_ready(tr["rew"])           # 컴파일 제외
            t0 = time.time()
            for _ in range(3):
                st, mxv, r, tr, _ = rollout(params, nrm, st, mxv, r, args.T, dr)
            jax.block_until_ready(tr["rew"])
            dt = (time.time() - t0) / 3
            print(f"n={n:>6}: {n*args.T/dt/1e3:8.1f}k steps/s  ({dt:.2f}s/rollout)",
                  flush=True)
        except Exception as e:
            print(f"n={n:>6}: FAIL {type(e).__name__} {str(e)[:120]}", flush=True)
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default="flat")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-mb", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="")
    ap.add_argument("--init", default="", help="체크포인트에서 이어서")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--wandb", choices=("auto", "off"), default="auto",
                    help="auto: 로그인 시 online, 아니면 offline 기록")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sweep(args)
    else:
        train(args)
