import jax, jax.numpy as jnp, mujoco, time
from mujoco import mjx

m = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <body><freejoint/><geom size=".15" mass="1" type="sphere"/></body>
</worldbody></mujoco>
""")
mx = mjx.put_model(m)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))   # 모델 공유, 데이터 배치
n = 4096
dx = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(n))

dx = jit_step(mx, dx)            # warm-up (여기서 컴파일)
dx.qpos.block_until_ready()
t = time.time()
for _ in range(100):
    dx = jit_step(mx, dx)
dx.qpos.block_until_ready()
print(f"{n*100/(time.time()-t):,.0f} steps/sec")
