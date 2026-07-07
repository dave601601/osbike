# moving-mass + free-fork (연구 타겟 플랜트)

## 2026-07-07 (최종 확정) — 무게추 조향: LQR이 bang-bang으로 포화 → 선회는 chaotic

### 근본 원인 (앞선 두 설명 모두 틀렸음, 이게 진짜)
무게추는 authority가 작아(B 작음) 불안정극(τ=0.25s)을 잡으려면 LQR gain이 거대해짐
(K0 ~ 수천~수만). → **힘이 시간의 85~96% ±20N 포화 = 사실상 relay/bang-bang**.
- **직진·소보정(≈5°): robust 안정.** 대칭이라 bang-bang limit-cycle이 안정 —
  초기섭동 0.49~0.51° 전부 50s 생존.
- **진짜 선회(≥10°): chaotic.** 선회가 슬라이더를 한쪽으로 치우쳐 stroke 여유를 없애면
  bang-bang이 발산 임계로 감. **초기섭동 0.5000°→25s, 0.5001°→9s** (0.0001° 차이로
  낙하시각 급변), 0.48→4s/0.49→22s/0.51→7s — 완전 비예측. "40°에 정착하나?"에
  robust한 yes/no 자체가 없음.

### 왜 그동안 flip-flop 했나 (교훈)
chaotic이라 실행마다 결과가 달라짐: 20s 호라이즌→"성공", 40s→"낙하", CPU vs GPU 상이,
inline vs mm_controller 상이, F_max 설계 바꾸면 또 상이. 각 실행을 결정론으로 오해해
매번 다른 메커니즘("오버슛"/"stroke 소진"/"authority")을 갖다 붙였음. **실제론 하나:
weakly-actuated plant → LQR 거대게인 → bang-bang → 선회 영역에서 chaotic.**
(사용자가 "잘하다 다른 이유로 넘어진다" 관찰로 chaotic 발견의 실마리 제공.)

### 무엇이 진짜 참인가 (robust하게)
- ✅ 직진 주행 균형 (v≥0.5), ✅ ~5° heading 미세보정 → **robust**.
- ⚠️ ≥10° 선회 → **chaotic/marginal** (되기도 하고 안 되기도, 예측 불가). robust하게
  "된다"고 주장 불가.
- 40° 영상(mm_turn40_falls.mp4)은 **한 chaotic 실현**(이 IC에서 25s 낙하)일 뿐,
  결정론적 거동 아님.
- kd_yaw는 여전히 역효과(roll 섞임) → 0 유지.

### 함의 (연구 — 오히려 강력)
이건 **LQR이 이 플랜트에서 부적합함을 보이는 결정적 증거**다: 약한 액추에이터 →
게인 포화 → chaos. 제약을 명시 처리하는 **MPC** 나 **RL** 이 정확히 필요해지는 지점.
제안서의 "RL vs 고전 robustness 비교"에서 고전(LQR)의 실패 모드를 정량화한 핵심 데이터.
authority를 키우면(무게추↑/스트로크↑) 포화가 풀려 선회가 robust해지는지는 다음 스윕 대상.

### ⚠️ 수치 marginality (중요한 함정)
이 컨트롤러는 내부루프가 F포화 ~87%인 **거의 bang-bang**이라 수치적으로 예민:
**동일 config가 CPU-JAX vs GPU-JAX에서 다른 궤적**(turn40 slew10: GPU 5° dip / CPU 25° dip;
S커브는 둘 다 전복). 배포 경로=CPU(numpy 추론)이므로 **CPU 기준으로만 성능 주장**할 것.
엔벨로프 표도 전부 CPU 측정. 교훈: 포화 지배 마진 컨트롤러는 플랫폼 민감 → 학습(MJX=GPU)
정책을 실기(CPU)로 옮길 때 sim2sim 갭 주의.

### CPU 단일 선회 엔벨로프 (최대 lean°, 생존 시, v=2)
| 목표°\slew | 3°/s | 5°/s | 10°/s | 20°/s |
|---|---|---|---|---|
| 10° | 3.4 | 전복 | 3.4 | 3.4 |
| 20° | 4.0 | 전복 | 19.9 | 25.5 |
| 30~60° | 3.8 | 3.6 | 22.9 | 전복 |

slew 5°/s의 비단조성(10·20°는 전복, 30°+는 생존)이 marginal/chaotic의 증거.
방향반전 ±target(@2s,10s, slew10): ±10 전복 / ±15 생존 / ±20·±30 전복 — 역시 비단조.

### 구조 (`mm_controller.py`)
```
yaw_ref → heading(yaw오차→lean_ref, ±lean_max) → balance_mass(lean을 lean_ref로) → 힘
```
- `balance_mass(st,g,lean_ref)`: `-(K·(x−x_ref))`, x_ref=[lean_ref,0,0,0].
- `heading`: `lean_ref=-(k_yaw·yaw_err−kd_yaw·yaw_rate)`, `clip(±lean_max)`. lean_max=0.5°.
  좌회전(yaw+)엔 왼쪽 기울기(lean−) → 앞의 음수.
- `v_fwd`=**body-forward**(v[0]cosψ+v[1]sinψ). world-x로 추종하면 선회 시 속도루프 헛돌아
  감속·전복 (실제 발생 → 수정).
- yaw_ref는 호출부에서 **슬루 3°/s** (완만 영역).

### 구조 (`mm_controller.py`)
```
yaw_ref → heading(yaw오차→lean_ref, ±lean_max 제한) → balance_mass(lean을 lean_ref로) → 힘
```
- `balance_mass(st,g,lean_ref)`: `-(K·(x−x_ref))`, x_ref=[lean_ref,0,0,0].
- `heading`: `lean_ref = -(k_yaw·yaw_err − kd_yaw·yaw_rate)`, `clip(±lean_max)`.
  부호: 좌회전(yaw+)엔 왼쪽 기울기(lean−) 필요 → 앞의 음수. lean_max=0.5°(0.0087rad).
- `v_fwd` = **body-forward** 속도(v[0]cosψ+v[1]sinψ). world-x로 추종하면 선회 시
  속도루프가 헛돌아 감속·전복 (실제로 발생 → 수정).
- yaw_ref 는 호출부에서 **슬루 제한 10°/s** (급격한 기준 반전이 전복 유발 — 아래).

### 발견/함정
1. **직진 K가 리밋사이클이면 조향에서 터진다.** 정지-검증용 공격적 K(4°,20N)는 즉시
   포화 bang-bang → 직진에서도 lean ±2° 리밋사이클. '생존'은 하나 yaw_ref 램프와 위상이
   맞으면 전복. → 운전점 재설계 **(10°,5N)**: v=2.0 직진에서 lean RMS 0.0°(진짜 수렴).
   교훈: 조향은 내부루프의 *정상상태 품질*을 드러낸다(생존≠수렴).
2. **급기준(슬루) 한계**: yaw_ref 슬루 25°/s까지 OK, 12°/s↓에서 오히려 전복 —
   느린 기준이 lean 을 한 방향으로 오래 물려 스트로크 소진. 10°/s 채택.
3. **선회는 본질적으로 완만**: lean_max=0.5°면 v=1 지속선회 R≈16.5m. 무게추 조향의
   근본 한계(지속 lean 유지 = 스트로크 상시 점유). 급선회엔 부족 — RL/큰 스트로크 여지.

### 검증 (v=2.0, 0.5° 초기섭동, slew 10°/s)
- +30° 스텝: 20s, 최종 yaw 28.6°, max lean 4.4°, F포화 87%(과도구간)
- S커브(+30/−30/0): 렌더 확인 (mm_scurve.mp4)
- 지속선회 R≈16.5m(v=1, lean_max 0.5°)

### 캐비앳
운전점 v=2.0·K(10°,5N)·lean_max 0.5°·slew 10°/s 한정. 정지~저속 조향은 미검증
(정지 균형 자체가 LQR 불가). heading gain(k_yaw=0.2)은 손튜닝 — MIMO/RL 로 흡수 여지.

### 파일 갱신
`mm_controller.py`(heading 추가), `mm_lqr.py`(운전점 (10°,5N)+heading gains 저장),
`mm_render.py`(MM_YAW 모드+슬루). 영상 `mm_turn30.mp4` `mm_scurve.mp4`(커밋 제외).

## 2026-07-07 — 플랜트 구축 + 4-state LQR: 저속 주행 균형 성공, 정지는 제약 지배로 실패

### 결론 (제안서 1단계 직결)
- **v ≥ 0.5~0.75 m/s에서 moving-mass 단독(free fork) 균형 20s 성공** — 무게추→lean→
  self-steering 동원이라는 제안서 핵심 메커니즘이 시뮬에서 작동함을 확인.
- **정지(v=0)는 전 파라미터 조합에서 LQR 실패** — 힘/스트로크 제약이 지배하는 구조적 한계.
- **설계 반직관 발견: 추는 가벼울수록 좋다** (힘 한계 고정 시). 하드웨어 설계에 직접 시사.

### 플랜트 (`moving_mass_bicycle.xml`, sanity 5/5 통과)
reaction-wheel 플랜트에서 파생: trail 74mm 포크·얇은 타이어·접촉제외 유지.
플라이휠/조향모터 제거(free fork), 안장 위(h=0.55m) 횡방향 슬라이더(기본 2kg, ±0.15m,
힘입력 ±20N, 비충돌 geom). M=13.2kg, h_CoM=0.519m, I0=4.213kg·m², 불안정 τ=0.250s.
`mm_model.build(m,d)`로 파라미터 변형. sanity: 인덱스/내부접촉(유령브레이크 검사)/trail/
역진자(2°→1.07s)/free-fork self-steering(기운 쪽 ✓).

### 4-state LQR (`mm_lqr.py`)
접촉선 기준 라그랑주: `[I0, −m·hm; −m·hm, m][θ̈;ÿ] = [Mghθ − mgy; F − mgθ − cẏ]`.
**부호 함정 3곳**(y+=왼쪽, lean+=오른쪽 규약): 오프셋 토크 −mgy, 결합항 −m·hm, 레일중력
−mgθ. 처음에 +로 썼다가 LQR이 추를 넘어지는 쪽으로 미는 증상(passive보다 빨리 낙하,
슬라이더 스톱 고정)으로 발각·수정. 시스템 ID로 모델 검증: 임펄스 응답(ẏ, roll̇) 오차
4~9%, 비최소위상 방향 일치.

### 왜 정지(v=0)는 안 되나 — 제약 지배 (모델 오류 아님)
- 0.1° 섭동에서도 선형 폐루프가 요구하는 스트로크 ≈ 0.21m > 0.15m 물리 한계.
- 불안정극 과도증폭(τ=0.25s) vs 추 이동시간의 경쟁 + 비최소위상 반작용 + 스톱 충돌.
- gain을 어떻게 잡아도 (강함→힘 포화 bang-bang / 약함→스트로크 초과) 실패.
  Bryson 4개 변형 전부 동일 낙하시간(1.0~2.7s)이 그 증거 (포화 지배 = gain 무관).
- 오답 노트: "무게추 고정 서보" 정역학 테스트는 검증이 아님 — 상수 오프셋은 넘어지는
  방향만 바꿈 (T2/T3에서 lean이 0을 지나 반대쪽으로 감 = 토크 권한 자체는 공식대로).
  fork-flop 가설도 기각 (포크 잠가도 동일).

### 설계영역 스윕 (0.5° 섭동, 20s 기준, per-config LQR 재설계, F=20N 고정)
| m\d | 0.10m | 0.15m | 0.20m |
|---|---|---|---|
| 1kg | v_min=0.75 | 0.75 | 0.75 |
| 2kg | 0.75 | **0.50** | — |
| 3kg | 0.75 | — | — |
| 4kg | — | — | — |

- **v=0 가능 조합 없음.** 최적: 2kg×0.15m (v_min=0.50).
- **무거울수록 나빠짐**: τ 내 가용 토크 `½F·g·τ²`는 질량 무관인데 반작용 관성(m·hm)은
  질량 비례 → 힘 한계 고정이면 가벼운 추가 우월. 스트로크 증가도 역효과(과도 시간↑).
- 하드웨어 시사: 슬라이더 구동력(벨트 장력/모터 토크) 예산이 추 질량보다 중요.

### 한계/캐비앳 (인용 시 주의)
F=20N·Bryson(4°,20N)·마운트 높이 0.55m 고정, 섭동 0.5°만, LQR 한정(MPC/RL은 v=0
경계를 밀어낼 수 있음 — 제약 인지 제어가 정확히 유리한 지점이라 RL 비교 명분 강화).

### 다음
- RL(PPO/SAC) 학습 → 동일 지표(v_min, 회복한계) LQR 대비 — 제안서 본론.
- F_max 축 추가한 3D authority 스윕, 마운트 높이 축.
- 정지 근처: MPC(제약 명시) 또는 fork lock/damper 하이브리드 검토.

### 파일
`moving_mass_bicycle.xml` `mm_model.py` `mm_sanity.py` `mm_lqr.py` `mm_controller.py`
`mm_lqr_gains.json` `mm_render.py` / 영상 `mm_ride_1ms.mp4`(커밋 제외)
