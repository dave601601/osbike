import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path
import jax, jax.numpy as jnp, mujoco, time
from mujoco import mjx

ROOT = Path(__file__).resolve().parents[2]
m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "reaction_wheel_bicycle.xml"))
mx = mjx.put_model(m)
step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

n = 1024
while True:
    try:
        dx = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(n))
        dx = step(mx, dx); dx.qpos.block_until_ready()      # 컴파일
        t = time.time()
        for _ in range(50): dx = step(mx, dx)
        dx.qpos.block_until_ready()
        print(f"nworld={n:>7}  OK   {n*50/(time.time()-t):>14,.0f} steps/sec")
        n *= 2
    except Exception:
        print(f"nworld={n:>7}  OOM  -> ceiling은 이 바로 아래")
        break
