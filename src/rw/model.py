"""모델 로드 + 신호 인덱스 상수. mx는 모든 world가 공유하는 정적 모델."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # VRAM 통째 선점 방지
from pathlib import Path
import mujoco
from mujoco import mjx

# --- 레포 루트 기준 경로 (CWD 무관). src/rw/model.py → parents[2] = 레포 루트.
ROOT   = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
PARAMS = ROOT / "params"

XML = str(ASSETS / "reaction_wheel_bicycle.xml")

m  = mujoco.MjModel.from_xml_path(XML)   # CPU 모델 (뷰어/초기화용)
mx = mjx.put_model(m)                     # GPU 모델 (배치 시뮬 공유)
DT = float(m.opt.timestep)                # 0.004

# --- sensordata 인덱스 (dim=11) ---
S_FRAME_UP = slice(0, 3)   # 프레임 z축(월드). [0,0,1]=직립
S_GYRO     = slice(3, 6)   # 각속도 (roll,pitch,yaw rate) — body frame
S_ACC      = slice(6, 9)
S_FW_W     = 9             # 플라이휠 각속도
S_REAR_W   = 10            # 뒷바퀴 각속도

# --- qpos 인덱스 (nq=11) ---
Q_POS   = slice(0, 3)     # xyz
Q_QUAT  = slice(3, 7)     # (w,x,y,z)
Q_STEER = 9               # 조향각

# --- qvel(dof) 인덱스 (nv=10). freejoint: [선속도(월드) 3, 각속도(body) 3] ---
V_LIN     = slice(0, 3)   # 선속도 (월드)
V_ANG     = slice(3, 6)   # 각속도 (body frame): [roll, pitch, yaw]_rate
V_FLYWHEEL = 7            # 플라이휠 각속도 (MJX 센서 대신 여기서 읽음)
V_STEER    = 8            # 조향 각속도

# --- actuator 인덱스 (nu=3) / ctrlrange ---
A_FLYWHEEL, A_STEER, A_REAR = 0, 1, 2
CTRL_LO = m.actuator_ctrlrange[:, 0].copy()   # [-10,-3,-4]
CTRL_HI = m.actuator_ctrlrange[:, 1].copy()   # [ 10, 3, 4]
