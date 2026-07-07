# 요약: 저속 자전거 균형 — 고전제어 baseline 완결 (2026-07 세션)

> 한 줄: **조향 모터 없이 무게추(moving mass)만으로, free-fork self-steering을 이용해
> 저속(≥0.65 m/s) 균형·선회·정지출발을 고전제어(LQR)로 다 풀었다.** 유일하게 막히는 곳은
> "여러 외란 결합 × 실차 액추에이터 지연"이며, 그것이 RL의 정량적 니치다.

세부 시간순 로그는 [progress/mm.md](progress/mm.md)(연구 플랜트), [progress/lqr.md](progress/lqr.md)·
[progress/plant.md](progress/plant.md)(리액션휠 스캐폴드). 이 문서는 결론만 정리.

---

## 1. 플랜트

| | reaction-wheel (스캐폴드) | **moving-mass + free-fork (연구 타겟)** |
|---|---|---|
| 파일 | `reaction_wheel_bicycle.xml` | `moving_mass_bicycle.xml` |
| 균형 액추에이터 | 플라이휠(리액션휠) | **좌우 이동 무게추 2kg, ±0.15m, 힘입력** |
| 조향 | 능동 steer 모터 | **free fork (무구동, self-steering)** |
| 공통 | 얇은 타이어(1.6cm)=역진자 / trail 74mm(caster 12°) / 뒷바퀴 구동 | 동일 |

공통 물리: M=13.2kg, h_CoM=0.52m, I_roll≈4.2, 불안정 시정수 τ=0.25s. 자가안정 속도창 v≥8 m/s.

**중요 모델 함정 2개** (초기 커밋부터 잠복, 발견·수정):
- frame↔front_wheel 조부모-손자 접촉 미제외 → downtube가 앞바퀴를 **1.9kN 유령 브레이크**로
  누름. → `<contact><exclude>`. (전진 불가·감속의 진범이었음)
- v1의 수직 조향축(trail=0) → self-steering 부재·포크 스톱 잼. → 12° caster로 trail 74mm.

## 2. 고전제어(LQR)가 달성한 것

설계: FD 선형화는 접촉 강성 아티팩트로 폐기 → **접촉선 기준 해석 진자 모델**(리액션휠은
3-state, moving-mass는 4-state [lean, roll̇, y_m, ẏ_m]) → 이산 리카티. 조향은 heading
캐스케이드(yaw오차→lean_ref), 속도는 뒷바퀴 PI.

| 기능 | 결과 |
|---|---|
| 정지출발 → 가속 → 순항 균형 | ✅ (초기 기울기 ≤1°, 0.78s에 안전속도 통과) |
| 저속 직진 균형 | ✅ **v ≥ 0.65 m/s** (pure balance; heading 루프 on 시 ~0.70) (사람다움 5/5, 30s+) |
| 선회 (무게추 유도 카운터스티어) | ✅ 완만 선회 slip<3.5° (slew≤7°/s), 40° 정착 |
| 개별 외란 | ✅ 옆경사~5°, 오르막~4°, 마찰 μ0.1(빙판), 범프 ≥5cm |
| **외란 결합 (지연 없이)** | ✅ **~s0.5**(2°/μ0.9/2cm+선회) 98% — 단 s0.75(3°/μ0.6/4cm)는 **20%**로 붕괴 (40 seeds) |
| **경사 코스 유지** (외곽루프) | ✅ heading PI+k_lat: \|crosstrack\| 중앙 **0.1m** (보강 전엔 내리막 항복, 30m 표류) |

## 3. 핵심 정량 결과 (제안서 control authority 곡선)

- **최소 균형속도 v_min ≈ 0.65 m/s.** 힘(60~200N)·스트로크(±0.15~1m) **무관** = self-steering
  한계(속도의존). v_min↓ 레버는 액추에이터가 아니라 **포크 trail/지오메트리**.
- **힘 임계 ≥50N.** 2kg·±0.15m에서 힘 20N=1/5(2Hz bang-bang), **≥50N=5/5 human-like**(0.1Hz,
  포화0). **20N 스펙이 이 세션의 거의 모든 병리(slam·선회chaos·비현실적motion)의 근본 원인.**
  질량↑은 힘 없이는 역효과. → **하드웨어 최소사양: 무게추 2kg·±0.15m·구동력 ≥50N.**
- **realism 지표**(`mm_metrics.py`): 진폭·한계접촉%·주파수·jerk·힘포화·lean으로 "사람다움/5" 채점.
- **결합×지연 생존율 맵**: `mm_envelope.py`(**공용 채점 하네스**, 40 seeds, 프로토콜 고정)
  → `envelope_lqr_klat.json`(정본; `_yawpi`=k_lat 어블레이션, 무접미=구 drift baseline)
  → `mm_plot_envelope.py` → `lqr_envelope_map.png`. RL도 같은 하네스로 채점해 맵을 겹친다.
  지연 격자는 물리스텝(4ms) 정수배만. **생존+코스 유지(ct_end/ct_max/yaw_end) 병기** —
  생존-only는 "내리막 항복" cheat 가능(구 baseline이 실제로 그랬음).

## 4. 고전제어의 한계 (2종)

**A. 물리 authority 한계 — RL도 못 넘음:**
- v=0 정적 균형 불가 (유한 스트로크 반작용질량 → travel 소진). v<0.65 불가 (self-steering 부재).
- 옆경사>5°(스트로크), 오르막>4°(드라이브 토크), 급선회(타이어 옆미끄럼). 큰 섭동 회복.

**B. 제어-설계 한계 — RL 니치 (40-seed, 외곽루프 보강 후 기준):**
- **결합 × 실차 지연 (주 타겟)**: 단독 지연은 40ms까지 OK(detune/Smith predictor로 60ms).
  그러나 s0.5 결합(2°/μ0.9/2cm+선회)에선 **98%@8ms → 55%@12ms → 5%@20ms** (50% 교차
  ~12ms, 완만한 붕괴). 실차 총지연(IMU 10~30ms + 50Hz + 액추에이터)이 이 대역 위.
- **강한 지형 결합 (추가 gap)**: s0.75(3°/μ0.6/4cm)는 **지연 0에서도 20%** — 지연 무관하게
  이미 붕괴 (μ0.6↔1.4 무차이 → 킬러는 경사+범프). RL이 개선하면 보너스, 못 하면 물리 한계.
- 위 둘은 lateral/heading 외곽루프(보강②)로도 그대로 → 쉬운 고전 보강으로는 안 닫힘.
- **왜 고전이 막히나**: Smith predictor(=지연-증강 LQR 최적)는 정확한 모델 필요 → 감축모델이
  지형·self-steering 무시 → 지형서 예측 틀림 → 보상 역효과. **model-based가 지형에 깨짐.**

## 5. RL 포지셔닝 (정직한 버전)

- RL은 "자전거를 세우려고" 필요한 게 아님 — nominal은 고전이 다 함 (그 자체가 결과).
- RL의 정당한 니치 = **결합×지연 하의 sim-to-real 강건성**. model-free라 지형+지연을 결합
  랜덤화(domain randomization)해 학습 → model-based 고전이 막히는 지점을 원리적으로 겨냥.
- 단 RL도 물리 authority 한계(v=0, sub-v_min 등)는 못 넘음.
- **비교 구도**: RL vs **강한** 고전 baseline(LQR+detune+Smith predictor). 대부분 RL 논문이
  약baseline과 비교하는 것과 반대 — 어느 쪽이 이겨도 정직한 기여.

## 6. 코드 맵

```
플랜트   reaction_wheel_bicycle.xml   moving_mass_bicycle.xml
공통     model.py / mm_model.py(build: 질량·스트로크·힘 변형)
제어     controller.py(리액션휠)   mm_controller.py(무게추: balance/steer/speed)
설계     lqr.py                    mm_lqr.py(4-state 해석 LQR)  *_gains.json
평가     rollout.py  mm_metrics.py(realism 채점)  mm_authority_sweep.py  mm_plot_envelope.py
검증     mm_sanity.py(회귀: ncon·trail·역진자·self-steer)
렌더     viz.py  render_bike.py  mm_render.py(MM_FORCE/LEANMAX/CAM/FLOOR/BUMP/V0/TURN_T)
```

## 7. 다음 단계

1. **RL 착수** (PPO/SAC + domain randomization, 지연·지형 포함) — 결합×지연 캡을 넘는지.
2. lateral/heading 외곽루프(k_lat) — 경사 드리프트 잡기(쉬운 고전 보강).
3. 결합×지연 맵 고해상도화 (seed↑), delay-aware를 steer 포함 모델로 개선.
4. 실기 system ID (조향마찰·접촉·지연 실측) 후 sim 보정 → 실전이.
