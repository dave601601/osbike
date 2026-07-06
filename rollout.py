"""폐루프 rollout (lax.scan) + cost. 가중치 w는 밖에서 주입 → 튜닝 시 재컴파일 최소."""
import jax.numpy as jnp
from jax import lax
from mujoco import mjx
import model as M
import controller as C

# 가중치 초기값 (나중에 조정). fall이 지배적이어야 "안 넘어짐"이 최우선.
W0 = dict(fall=10.0, lat=1.0, yaw=0.5, spd=1.0, ctrl=0.001, fw=1e-4)


def rollout(g: C.Gains, dx0, v_target, T):
    """gain 하나로 T스텝 폐루프. carry=(dx, 속도적분, alive). 스텝별 신호 기록 반환."""
    def step(carry, _):
        dx, integ, alive = carry
        ctrl, integ, st = C.controller(dx, g, integ, v_target, M.DT)
        dx = dx.replace(ctrl=ctrl)
        dx = mjx.step(M.mx, dx)
        alive = alive & (st.up_z > 0.7)                 # 한 번 넘어지면 계속 dead
        rec = dict(y=st.y_lat, yaw=st.yaw, v=st.v_fwd,
                   fw=st.fw_speed, u=ctrl, alive=alive)
        return (dx, integ, alive), rec

    init = (dx0, jnp.float32(0.0), jnp.bool_(True))
    _, traj = lax.scan(step, init, None, length=T)
    return traj


def cost(traj, T, v_target, w=W0):
    """생존 우선 + v_target 추종 + 직진(lateral/heading) 스칼라 비용."""
    alive = traj['alive'].astype(jnp.float32)
    survived = alive.sum()
    denom = alive.sum() + 1e-6
    mmean = lambda x: (x * alive).sum() / denom        # 살아있는 구간 평균

    c  = w['fall'] * (T - survived)                    # ① 넘어짐
    c += w['lat']  * mmean(traj['y'] ** 2)             # ② 직진 (lateral)
    c += w['yaw']  * mmean(traj['yaw'] ** 2)           # ③ heading
    c += w['spd']  * mmean((traj['v'] - v_target) ** 2)  # ④ v_target 추종
    c += w['ctrl'] * mmean((traj['u'] ** 2).sum(-1))   # ⑤ 제어 노력
    c += w['fw']   * mmean(traj['fw'] ** 2)            # ⑥ 플라이휠 포화
    return c


def evaluate(g: C.Gains, dx0, v_target, T, w=W0):
    return cost(rollout(g, dx0, v_target, T), T, v_target, w)
