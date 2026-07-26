"""moving-mass 플랜트 로드 + 신호 인덱스. (reaction-wheel 스택의 model.py와 병렬 구조)"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import mujoco
from mujoco import mjx

# --- 레포 루트 기준 경로 (CWD 무관). src/mm/mm_model.py → parents[2] = 레포 루트.
# mm_* 모듈은 전부 이 파일을 import 하므로 여기가 경로 단일 소스.
ROOT    = Path(__file__).resolve().parents[2]
ASSETS  = ROOT / "assets"
PARAMS  = ROOT / "params"
RESULTS = ROOT / "results"

XML = str(ASSETS / "moving_mass_bicycle.xml")

m  = mujoco.MjModel.from_xml_path(XML)
mx = mjx.put_model(m)
DT = float(m.opt.timestep)                # 0.004

# --- qpos (nq=11): freejoint 7 + rear_spin, slide_y, steer, front_spin ---
Q_POS   = slice(0, 3)
Q_QUAT  = slice(3, 7)
Q_SLIDE = 8               # 무게추 lateral 위치 [m]
Q_STEER = 9               # 조향각 (free fork — 관측만)

# --- qvel (nv=10): freejoint 6 + 4 조인트 ---
V_LIN   = slice(0, 3)
V_ANG   = slice(3, 6)     # body frame [roll, pitch, yaw] rate
V_REAR  = 6
V_SLIDE = 7
V_STEER = 8
V_FRONT = 9

# --- actuator (nu=2) ---
A_SLIDE, A_REAR = 0, 1
CTRL_LO = m.actuator_ctrlrange[:, 0].copy()   # [-20, -4]
CTRL_HI = m.actuator_ctrlrange[:, 1].copy()   # [ 20,  4]

WHEEL_R = 0.30


def build(mm_mass=2.0, mm_range=0.15, force=20.0):
    """무게추 질량/이동범위/힘한계를 바꾼 변형 모델 (설계영역 스윕용). 문자열 치환."""
    xml = open(XML).read()
    xml = xml.replace('size="0.05 0.04 0.03" mass="2.0"',
                      f'size="0.05 0.04 0.03" mass="{mm_mass}"')
    xml = xml.replace('range="-0.15 0.15"', f'range="-{mm_range} {mm_range}"')
    xml = xml.replace('ctrlrange="-20 20"', f'ctrlrange="-{force} {force}"')  # slide_force
    return mujoco.MjModel.from_xml_string(xml)
