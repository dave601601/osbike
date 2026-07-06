# LQR (고전 균형 제어)

## 2026-07-06 — LQR 플라이휠 균형 + 조향 yaw-hold 로 20s 균형 성공

### 결론
`lqr.py` 구현 완료. **LQR 플라이휠 균형(roll 3-state) + 조향 yaw-hold(kp_yaw=10, kd_yaw=2)**
조합으로 섭동 **5°까지 정지·주행(2 m/s) 모두 20s 유지** (8°에서 토크 포화로 실패).
검증된 gain은 `lqr_gains.json`, `viz.py`가 우선 로드.

### 진행 경로 (왜 이렇게 됐나)
1. **기존 `best_gains.json`(랜덤서치)은 균형 실패** — sweep 자기 IC에서도 ~1.08s에 넘어짐.
   원인: (a) sweep의 `kp_lean/kd_lean` 탐색범위가 양수 = 복원과 반대 부호(양의 피드백),
   (b) sweep IC(완벽 직립) ≠ viz IC(2° 섭동), (c) 애초에 단순 flywheel-PD로는 안 섬.
2. **플랜트는 제어가능 확인** — open-loop 일정토크 +10 N·m 가 2° lean 을 0.2s만에 반전.
   언더액추에이션 아님 → 컨트롤러 설계 문제.
3. **FD 선형화(`mjd_transitionFD`)는 실패** — 얇은 타이어 강체 접촉을 free-joint roll로
   섭동하면 접촉 법선 강성이 A에 섞임(A[1,0]~194, 물리적으로 말 안 됨) → gain 4자리 폭주
   (kp_lean≈-46700) → 상시 포화(bang-bang) → 여전히 ~1s에 실패.
4. **해석 모델로 전환(정답)** — 접촉선(지면 x축) 기준 리액션휠 진자:
   `I_p θ̈ = M g h sinθ − τ`, `ω̇ = τ/I_r − θ̈`.
   파라미터(자동 추출): M=13.5 kg, h=0.520 m, I_p=4.30, I_r=0.0211 kg·m², 불안정 τ=0.25s.
   `A_c[1,0]=Mgh/I_p=16.0` 로 open-loop 실측과 정합.
   Bryson Q/R → 이산 리카티 → **K=[-295.7, -57.8, -0.0325]**, 닫힌루프 |eig|<1.
5. **해석 LQR도 단독으로는 실패** — 트레이스 결과 **roll↔yaw 스파이럴**로 넘어짐.
   free-castering 앞포크가 lean→steer→yaw→lean 결합을 만들고, 플라이휠이 스핀업하며
   yaw가 단조 증가(−0.75 rad)해서 쓰러짐. 3-state roll 모델이 못 보는 모드.
6. **조향 yaw-hold 추가로 해결** — `heading()` 루프(kp_yaw=10, kd_yaw=2)로 yaw를 잡으니
   결합이 끊겨 20s 안정. 단, 너무 세면(40/5) 재불안정(4.4s) → 스윗스팟 존재.

### 검증 (survived / 5000 steps = 20s)
| 섭동 | 정지 v=0 | 주행 v=2 |
|---|---|---|
| 1–5° | 5000 (OK) | 5000 (OK) |
| 8° | 378 (실패) | 375 (실패) |

8° 실패는 예상된 한계: 10 N·m 로 정적으로 버틸 수 있는 lean ≈ 8.2° (M g h sinθ = τ_max).

### 다음 후보
- **MIMO LQR**: 조향을 상태(steer, steer_rate, yaw_rate)+입력으로 포함해 균형+조향 동시 최적화
  → yaw-hold gain 손튜닝 제거, 결합을 명시적으로 처리. (지금은 balance만 LQR, 조향은 손 PD)
- 큰 섭동(>8°) 회복: 토크 예산↑(플라이휠 관성/토크) 또는 조향 주도 카운터스티어.
- 외부 lateral 위치 루프(k_lat) 복원해 직진선 추종.
- FD 선형화를 쓰려면 접촉 완화(soft solref)나 접촉선 기준 좌표로 재설정 필요.

### 파일
- `lqr.py` — `bike_params()` / `state_space()` / `discretize()` / `flywheel_balance()`
- `lqr_gains.json` — 검증된 full Gains
- `viz.py` — lqr_gains.json 우선 로드
- `balance_lqr_20s.mp4` — 20s 균형 렌더 (서브에이전트, 커밋 제외)
