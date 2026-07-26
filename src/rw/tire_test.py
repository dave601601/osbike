"""무제어(토크=0) passive 안정성 — 타이어 반폭(halfwidth)의 영향.

가설: 얇은 타이어일수록 접지가 점에 가까워 roll 불안정(역진자). 폭이 넓어지면
접지 패치가 넓어져 CoM 투영이 ±w 안에 있는 동안 정적으로 버팀.
정적 안정 한계 lean ≈ atan(w / h_CoM).  (w=반폭, h≈0.52m)

각 (반폭, 초기 roll 섭동)에서 ctrl=[0,0,0]으로 rollout → 넘어질 때까지 시간 측정.
XML은 문자열에서 fromto만 바꿔 로드 (디스크 XML 불변, 질량은 1.2 고정 → 폭만 변수).

사용:  python tire_test.py
"""
import numpy as np
import mujoco
import model as M

T = 5000                                   # 20 s
HALFWIDTHS = [0.008, 0.02, 0.04, 0.08, 0.15]   # 반폭 [m] (현재=0.008=1.6cm 전폭)
PERTS = [0.5, 1.0, 2.0, 5.0, 10.0]         # 초기 roll 섭동 [deg]
H_COM = 0.520                              # lqr.bike_params()에서


def make_model(halfwidth):
    xml = open(M.XML).read()
    xml = xml.replace('fromto="0 -0.008 0  0 0.008 0"',
                      f'fromto="0 -{halfwidth} 0  0 {halfwidth} 0"')
    return mujoco.MjModel.from_xml_string(xml)


def survive_steps(m, pert_deg):
    """ctrl=0 으로 최대 T스텝. up_z<0.7 시점 반환 (안 넘어지면 T)."""
    d = mujoco.MjData(m)
    a = np.radians(pert_deg) / 2.0
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0.0, 0.0]
    mujoco.mj_forward(m, d)
    for k in range(T):
        d.ctrl[:] = 0.0                    # ← 무제어
        mujoco.mj_step(m, d)
        q = d.qpos
        up_z = 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5])
        if up_z < 0.7:
            return k
    return T


if __name__ == "__main__":
    print("무제어(ctrl=0) 낙하 시간 [초].  'STABLE'=20s 유지.  전폭=2×반폭")
    print(f"정적 안정 한계 lean = atan(w/h), h={H_COM}m\n")
    hdr = "반폭(전폭)      " + "".join(f"{p:>7.1f}°" for p in PERTS) + "   | 정적한계"
    print(hdr)
    print("-" * len(hdr))
    for w in HALFWIDTHS:
        m = make_model(w)
        cells = []
        for p in PERTS:
            k = survive_steps(m, p)
            cells.append("STABLE" if k >= T else f"{k * M.DT:.2f}s")
        basin = np.degrees(np.arctan(w / H_COM))
        label = f"{w*100:.1f}cm({w*200:.0f}cm)"
        print(f"{label:14s}" + "".join(f"{c:>8s}" for c in cells) + f"   | {basin:4.1f}°")
