# moving-mass + free-fork (연구 타겟 플랜트)

## 2026-07-07 (정정) — 무게추 조향: heading을 "멈출" 수 없음. ~5° 미세보정만 유지, 진짜 선회 불가

### ⚠️ 앞선 "완만 선회 성공" 주장은 틀렸음 (긴 호라이즌 검증 누락)
20s 호라이즌으로 "생존"만 봐서 성공으로 오판했으나, **40s+ 로 늘려 보면 전부 오버슛 후
낙하**. 실제 결과:
- **heading 홀드 한계 ≈ 5°**: 5° 타겟만 50s 정착·유지(오버슛 0.7°). **10°↑는 전부 낙하**
  (10°→39.7° 반대로 넘어가며 42s, 20°→오버슛 22°→20s, 30·40°→18~25s 낙하).
- **원인**: 무게추가 선회를 *시작*은 해도 *멈추질* 못함. 선회 정지 = lean 반대로 넘기기
  = 방향반전과 동일 = authority 초과. yaw가 목표를 지나쳐 계속 돌다 전복.
- **yaw rate 상한**: v=2, lean_max 0.5°에서 g·tan(lean)/v = **2.45°/s**가 물리 한계.
  명령 slew 3°/s 는 이보다 빨라 애초에 못 따라감(지연 누적).
- **kd_yaw(yaw_rate 감쇠)는 역효과**: st.yaw_rate=v[5]가 lean 중엔 roll과 섞여 균형을
  깨뜨림 → kd_yaw>0 이면 2.4s 즉시 낙하. 현재 gains는 kd_yaw=0.
- **보낸 40° 영상들은 오해 유발**: 선회 진행 중(yaw 43°까지 상승) 24s에 끝났고 **25s에
  낙하**. "선회 완료"가 아니라 "낙하 직전 스냅샷"이었음.

### 함의 (하드웨어/연구)
2kg·±0.15m 슬라이더는 **직진 균형 + ~5° 미세 heading 보정**까지가 authority. 진짜 선회엔
훨씬 큰 CoM 이동권한 필요(실제 손놓고타기는 상체 ~30kg를 크게 씀) 또는 고속 self-stability.
→ "moving-mass authority 한계"의 결정적 데이터. RL/MPC가 이 물리 한계를 넘을 수 있는지는
별개 검증 대상(넘을 가능성은 낮음 — 제약이 physical).

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
