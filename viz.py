"""최적 gain을 일반 mujoco(CPU, 한 개)로 재생. 순수함수 controller를 그대로 재사용.

사용:  python viz.py            # best_gains.json 있으면 로드, 없으면 손 gain
"""
import json, os, time
import numpy as np
import mujoco, mujoco.viewer
import model as M
import controller as C

V_TARGET = 2.0

# 우선순위: lqr_gains.json(LQR 균형+yaw-hold) > best_gains.json(랜덤서치) > 손 gain
if os.path.exists("lqr_gains.json"):
    G = C.Gains(**json.load(open("lqr_gains.json")))
    print("lqr_gains.json 로드 (LQR balance + yaw-hold)")
elif os.path.exists("best_gains.json"):
    G = C.Gains(**json.load(open("best_gains.json")))
    print("best_gains.json 로드")
else:
    G = C.Gains(kp_lean=40., kd_lean=4., kw_fw=0.01,
                k_lat=2., kp_yaw=5., kd_yaw=1.,
                kp_v=2., ki_v=0.5)
    print("손 gain 사용")


def main():
    m = M.m
    d = mujoco.MjData(m)
    # 초기 2도 roll 섭동 (밸런스 동작 확인용)
    a = np.radians(2.0) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0., 0.]
    mujoco.mj_forward(m, d)
    integ = 0.0
    dt = M.DT
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            # controller 는 JAX 함수지만 numpy 값으로도 호출 가능 → CPU에서 그대로 재사용
            ctrl, integ, st = C.controller(d, G, integ, V_TARGET, dt)
            d.ctrl[:] = np.asarray(ctrl)
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0, dt - (time.time() - t0)))   # 실시간 페이싱


if __name__ == "__main__":
    main()
