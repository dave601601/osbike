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


SPR_PROJ = 128


def init_params(rng, obs_dim, act_dim, hidden=(256, 256), spr_k=0, coma=False):
    # k1/k2 유도는 SPR/COMA 유무와 무관하게 고정 — 키 분할을 바꾸면 보조손실을 꺼도
    # pi/v 초기화가 달라져서 "그것만 켠 대조군"이 성립하지 않는다.
    k1, k2 = jax.random.split(rng)
    p = dict(pi=mlp_init(k1, (obs_dim, *hidden, act_dim), 0.01),
             v=mlp_init(k2, (obs_dim, *hidden, 1), 1.0),
             log_std=jnp.full((act_dim,), -1.0))
    if coma:
        # action-conditioned 크리틱. counterfactual baseline = 잔차 0 (= base 단독).
        p["q"] = mlp_init(jax.random.fold_in(rng, 2),
                          (obs_dim + act_dim, *hidden, 1), 1.0)
    if spr_k:
        k3, k4, k5 = jax.random.split(jax.random.fold_in(rng, 1), 3)
        # SPR (Schwarzer+ 2021) 보조 헤드. 인코더는 pi 트렁크를 그대로 씀 —
        # 표현학습이 정책 표현에 실제로 작용해야 의미가 있고, 별도 인코더를 두면
        # "표현 강화"가 정책과 무관한 곳에서만 일어난다.
        lat = hidden[-1]
        p["spr_tr"] = mlp_init(k3, (lat + act_dim, lat, lat), 1.0)   # 잠재 전이모델
        p["spr_pj"] = mlp_init(k4, (lat, SPR_PROJ), 1.0)             # projection
        p["spr_pd"] = mlp_init(k5, (SPR_PROJ, SPR_PROJ), 1.0)        # prediction
    return p


def encode(ps, x):
    """pi 트렁크(마지막 층 제외) = SPR 인코더."""
    for W, b in ps["pi"][:-1]:
        x = jnp.tanh(x @ W + b)
    return x


def _l2n(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def spr_loss(p, p_t, nob, act, alive):
    """K-step latent 예측. nob (K+1,M,D) / act (K,M,A) / alive (K,M).

    타깃은 EMA 타깃 인코더 + stop-grad (collapse 방지 — 온라인 인코더로 타깃을 만들면
    상수 표현으로 붕괴한다). 손실은 코사인 유사도의 음수.
    """
    z = encode(p, nob[0])
    tot = 0.0
    for k in range(act.shape[0]):
        z = mlp(p["spr_tr"], jnp.concatenate([z, act[k]], -1))
        y = _l2n(mlp(p["spr_pd"], mlp(p["spr_pj"], z)))
        t = _l2n(jax.lax.stop_gradient(mlp(p_t["spr_pj"], encode(p_t, nob[k + 1]))))
        tot = tot - jnp.sum((y * t).sum(-1) * alive[k]) / (alive[k].sum() + 1e-8)
    return tot / act.shape[0]


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


def q_of(params, obs, act):
    return mlp(params["q"], jnp.concatenate([obs, act], -1))[..., 0]


def coma_adv(params, nob, act):
    """COMA 식 counterfactual advantage, residual RL 판.

    A = Q(s, a) - Q(s, 0). 잔차 0 은 곧 base 컨트롤러 단독이므로, 이 차이는
    "이 잔차가 base 대비 실제로 보탠 것"이다. 원 COMA 는 이산 행동공간에서 다른
    에이전트를 고정한 채 정확히 marginalize 하지만, 여기는 에이전트가 하나이고 행동이
    연속이라 base(=0) 라는 자연스러운 기준점 하나로 대체한다.

    주의: Q(s,0) 은 off-distribution 질의다 (정책이 0 근처를 자주 뽑지 않으면 외삽).
    이게 이 방법의 알려진 약점이고, 실패하면 여기가 원인일 가능성이 가장 크다.
    """
    return q_of(params, nob, act) - q_of(params, nob, jnp.zeros_like(act))


def _std_norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

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


@partial(jax.jit, static_argnums=(2, 3))
def build_spr_windows(tr, nrm, K, M, rng):
    """(T,N) 롤아웃에서 K-step 윈도우 M개를 뽑아 (K+1,M,D)/(K,M,A)/(K,M) 로 반환.

    평탄화 배치로는 시퀀스를 못 뽑으므로 별도 풀로 관리. alive 는 done 을 만나면 0 이
    되는 누적 마스크 — 에피소드 경계를 넘는 예측은 학습신호가 아니라 잡음이다.
    """
    T, N = tr["done"].shape
    kt, kn = jax.random.split(rng)
    t0 = jax.random.randint(kt, (M,), 0, T - K)      # t+K 가 범위 안이어야 함
    n0 = jax.random.randint(kn, (M,), 0, N)
    nob = jnp.stack([norm_apply(nrm, tr["obs"][t0 + k, n0]) for k in range(K + 1)])
    act = jnp.stack([tr["act"][t0 + k, n0] for k in range(K)])
    dones = jnp.stack([tr["done"][t0 + k, n0] for k in range(K)]).astype(jnp.float32)
    alive = jnp.cumprod(1.0 - dones, axis=0)
    return dict(nob=nob, act=act, alive=alive)


@partial(jax.jit, static_argnums=(6, 7), static_argnames=("coma",))
def update(params, opt, nrm, batch, rng, lr, epochs, n_mb,
           clip_eps=0.2, vf_coef=0.5, ent_coef=0.005,
           params_t=None, spr=None, spr_coef=0.0, coma=False):
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
        l_spr = (spr_loss(p, params_t, mb["s_nob"], mb["s_act"], mb["s_alive"])
                 if spr is not None else 0.0)
        # Q 는 V 와 같은 타깃(GAE 리턴)으로 회귀. counterfactual 은 이 Q 로만 만든다.
        l_q = 0.5 * jnp.mean((q_of(p, mb["nob"], mb["act"]) - mb["ret"]) ** 2) \
            if coma else 0.0
        tot = (l_pi + vf_coef * l_v - ent_coef * ent
               + spr_coef * l_spr + vf_coef * l_q)
        return tot, (l_pi, l_v, ent, kl, frac, l_spr, l_q)

    def epoch(carry, _):
        params, opt, rng = carry
        rng, k = jax.random.split(rng)
        idx = jax.random.permutation(k, B).reshape(n_mb, -1)
        # SPR 앵커는 별도 풀에서 n_mb 조각으로 나눠 쓴다 (PPO 미니배치와 인덱스 무관 —
        # 시퀀스 윈도우라 평탄화 인덱스로는 못 뽑는다).
        sidx = jnp.arange(spr["nob"].shape[1]).reshape(n_mb, -1) if spr is not None else None

        def mb_step(carry, ixs):
            params, opt = carry
            ix, six = ixs
            mb = jax.tree.map(lambda x: x[ix], batch)
            if spr is not None:
                mb = {**mb, "s_nob": spr["nob"][:, six], "s_act": spr["act"][:, six],
                      "s_alive": spr["alive"][:, six]}
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            params, opt, gn = adam_step(params, g, opt, lr)
            return (params, opt), (l, *aux, gn)
        scan_in = (idx, sidx) if spr is not None else (idx, idx)
        (params, opt), stats = jax.lax.scan(mb_step, (params, opt), scan_in)
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
# v2 사후: 무앵커 scale 1.0 이 과작동 진동 국소최적에 빠짐(평지 lean 3-5°, 쉬운 셀
# 붕괴; 잔차 절반 평가로 즉시 회복 = 방향은 옳고 크기가 과함) → v3 = scale 0.5 +
# 잔차 크기 벌점(res_pen) 앵커.
STAGES["res_v3"] = STAGES["res"]._replace(res_scale=0.5, res_pen=0.05)


def train(args):
    dr = STAGES[args.stage]
    run = args.run or f"{args.stage}_n{args.num_envs}"
    outdir = os.path.join(E.ROOT, "ckpt", run)   # 레포 루트 기준 (CWD 무관)
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "log.jsonl"), "a")
    rng = jax.random.PRNGKey(args.seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    params = init_params(k1, E.OBS_DIM, E.ACT_DIM, spr_k=args.spr, coma=bool(args.coma))
    if args.init:
        with open(args.init, "rb") as f:
            saved = pickle.load(f)
        params, nrm = saved["params"], saved["nrm"]
        if args.reset_log_std:
            # 커리큘럼 함정: 전 스테이지에서 σ가 하한까지 수렴한 채 넘어오면 새 도메인
            # 탐색 불가 (pure_dr: 경사 course-hold 미학습 → heading 항복 실증)
            params = dict(params, log_std=jnp.full_like(params["log_std"], -1.0))
        print(f"[init] {args.init} (log_std reset={args.reset_log_std})")
    else:
        nrm = norm_init(E.OBS_DIM)
    opt = adam_init(params)
    params_t = jax.tree.map(lambda x: x, params) if args.spr else None
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
        # COMA: 롤아웃 시점 파라미터로 counterfactual advantage 를 한 번만 계산해
        # GAE adv 와 같은 위치에 넣는다 (에폭 중 재계산하면 PPO 의 고정-advantage
        # 전제가 깨진다). warmup 동안은 Q 가 난수라 GAE 를 그대로 쓴다.
        if args.coma and it >= args.coma_warmup:
            nob_all = norm_apply(nrm, batch["obs"])
            a_cf = coma_adv(params, nob_all, batch["act"])
            batch["adv"] = ((1 - args.coma_mix) * _std_norm(batch["adv"])
                            + args.coma_mix * _std_norm(a_cf))
        spr = build_spr_windows(tr, nrm, args.spr, args.spr_anchors, ku) if args.spr else None
        params, opt, stats = update(params, opt, nrm, batch, ku,
                                    args.lr, args.epochs, args.n_mb,
                                    params_t=params_t, spr=spr,
                                    spr_coef=args.spr_coef if args.spr else 0.0,
                                    coma=bool(args.coma))
        if args.spr:                       # 타깃 인코더 EMA (collapse 방지)
            params_t = jax.tree.map(lambda t, o: args.spr_tau * t + (1 - args.spr_tau) * o,
                                    params_t, params)
        l, lpi, lv, ent, kl, frac, l_spr, l_q, gn = [float(x) for x in stats]
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
        if args.spr:
            rec["spr"] = round(l_spr, 4)   # -1 에 가까울수록 예측 성공 (코사인)
        if args.coma:
            rec["q_loss"] = round(l_q, 2)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.log_every == 0:
            print(rec, flush=True)
        if it % args.ckpt_every == 0 or it == args.iters - 1:
            with open(os.path.join(outdir, f"params_{it:05d}.pkl"), "wb") as f:
                # n_frames 는 obs 레이아웃을 결정하므로 반드시 저장 — 모듈 상수로
                # 추론하면 학습/채점 프레임 수가 어긋나도 조용히 지나간다.
                pickle.dump(dict(params=jax.device_get(params),
                                 nrm=jax.device_get(nrm), it=it,
                                 stage=args.stage, dr=dr._asdict(),
                                 n_frames=E.N_FRAMES, seed=args.seed,
                                 spr=args.spr, spr_coef=args.spr_coef,
                                 coma=args.coma, coma_mix=args.coma_mix), f)
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
    ap.add_argument("--reset-log-std", action="store_true",
                    help="init 시 log_std 를 -1.0 으로 리셋 (새 도메인 탐색 회복)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--wandb", choices=("auto", "off"), default="auto",
                    help="auto: 로그인 시 online, 아니면 offline 기록")
    ap.add_argument("--spr", type=int, default=0,
                    help="SPR 보조손실: K-step 잠재 예측 (0=끔, 8=다음 8 obs 예측)")
    ap.add_argument("--spr-coef", type=float, default=1.0, help="SPR 손실 가중")
    ap.add_argument("--spr-anchors", type=int, default=8192,
                    help="iteration 당 SPR 윈도우 앵커 수 (전 배치는 메모리 초과)")
    ap.add_argument("--spr-tau", type=float, default=0.99, help="타깃 인코더 EMA")
    ap.add_argument("--coma", action="store_true",
                    help="counterfactual advantage A=Q(s,a)-Q(s,0) (0=base 단독)")
    ap.add_argument("--coma-mix", type=float, default=1.0,
                    help="adv = (1-mix)*GAE + mix*counterfactual")
    ap.add_argument("--coma-warmup", type=int, default=50,
                    help="이 iteration 까지는 GAE 사용 (Q 가 아직 난수)")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sweep(args)
    else:
        train(args)
