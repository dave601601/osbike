"""GPU 병렬 gain 탐색: N개 gain 조합을 vmap으로 한 번에 폐루프 평가 → argmin."""
import jax, jax.numpy as jnp
from mujoco import mjx
import model as M
import controller as C
import rollout as R

V_TARGET = 2.0      # 목표 순항속도 [m/s]
T        = 1500     # rollout 스텝 (×DT=0.004 → 6초)
N        = 4096     # gain 조합 수

# 탐색 범위 [lo, hi] — Gains 필드 순서와 동일.
# 주의: 이 랜덤서치는 LQR(lqr.py)로 대체됨 — 비교 baseline 용도로만 유지.
# balance gain 은 음수가 복원 방향 (v1에서 양수 범위만 탐색해 실패했던 이력).
#         kp_lean kd_lean  kw_fw   k_ls  kp_st  kd_st  kp_v  ki_v
LO = jnp.array([-500., -100., -0.10,  -6.,   0.5,   0.1,  0.5,  0.0])
HI = jnp.array([ -50.,   -5.,  0.00,   0.,   4.0,   1.5,  4.0,  2.0])


def sample_gains(key, n):
    u = jax.random.uniform(key, (n, LO.shape[0]))
    arr = LO + u * (HI - LO)                       # (n, k)
    return jax.vmap(lambda v: C.Gains(*v))(arr)    # Gains(leaves=(n,))


def make_dx0():
    dx = mjx.make_data(M.mx)
    return jax.tree.map(lambda x: x, dx)           # 단일 초기상태 (모든 world 공유)


@jax.jit
def eval_batch(gains, dx0):
    f = lambda g: R.evaluate(g, dx0, V_TARGET, T)
    return jax.vmap(f)(gains)                       # (N,) costs


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    gains = sample_gains(key, N)
    dx0 = make_dx0()

    costs = eval_batch(gains, dx0)                  # 첫 호출 = 컴파일
    costs.block_until_ready()

    best = int(jnp.argmin(costs))
    bg = jax.tree.map(lambda x: float(x[best]), gains)
    print(f"best cost = {float(costs[best]):.3f}  (N={N}, T={T})")
    print("best gains:")
    for k, v in bg._asdict().items():
        print(f"  {k:8s} = {v:.4f}")

    import json
    json.dump(bg._asdict(), open("best_gains.json", "w"), indent=2)
    print("-> best_gains.json 저장")
