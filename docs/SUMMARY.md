# 요약: 저속 자전거 균형 — 고전제어 baseline 완결 (2026-07 세션)

> 한 줄: **조향 모터 없이 무게추(moving mass)만으로, free-fork self-steering을 이용해
> 저속(≥0.65 m/s) 균형·선회·정지출발을 고전제어(LQR)로 다 풀었다.** 결합×지연도 Smith
> predictor(정확 모델)면 거의 닫힌다 — 단 **모델오차 ±5%에 붕괴**(무보상 이하)하고, 강한
> 지형 결합(s0.75+)은 무엇으로도 안 열린다. **RL의 정량 니치 = 모델오차·지형 하 강건성.**

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
- **결합×지연 생존율 맵**: `mm_envelope.py`(**공용 채점 하네스**, 40 seeds, 프로토콜 고정,
  `--ctrl base|smith4|smith6`) → `envelope_lqr_smith4.json`(최강 고전, 정확모델) /
  `_klat.json`(무보상 base) / `_yawpi.json`(k_lat 어블) / 무접미(구 drift baseline)
  → `mm_plot_envelope.py` → `lqr_envelope_map.png`. RL도 같은 하네스로 채점해 맵을 겹친다.
  지연 격자는 물리스텝(4ms) 정수배만. **생존+코스 유지(ct_end/ct_max/yaw_end) 병기** —
  생존-only는 "내리막 항복" cheat 가능(구 baseline이 실제로 그랬음).
- **지연보상**: `mm_delay.py` — smith4(해석 4-state 예측기)/smith6(steer 포함 6-state
  DMDc sysid, `mm_sysid_6state.json`) + 예측오차 리포트. 모델오차 민감도는 §4B.

## 4. 고전제어의 한계 (2종)

**A. 물리 authority 한계 — RL도 못 넘음:**
- v=0 정적 균형 불가 (유한 스트로크 반작용질량 → travel 소진). v<0.65 불가 (self-steering 부재).
- 옆경사>5°(스트로크), 오르막>4°(드라이브 토크), 급선회(타이어 옆미끄럼). 큰 섭동 회복.

**B. 제어-설계 한계 — RL 니치 (40-seed, 보강①②③ 후 최종):**
- **결합 × 지연은 정확한 모델이 있으면 고전으로도 거의 닫힌다** (구 "10-15ms 캡" 주장
  정정 — 약한 baseline+3 seeds 아티팩트). 무보상 s0.5: 98@8ms→55@12ms→5@20ms인 것을
  해석모델 Smith predictor(smith4)가 **95@12ms·88@16ms·52@20ms**로 회복.
- **그러나 smith4는 취약**: 예측기 모델오차 **+5%에서 16ms 셀이 9/10→0/10** (±10~20%는
  무보상보다 나쁨). steer 포함 6-state sysid 예측기(smith6)는 롤포워드 오차 증폭으로
  더 못함(지형 12ms+ 붕괴). **model-based 지연보상은 정확도가 아니라 강건성이 병목** —
  실차 등가 모델오차(액추에이터 게인·마찰·타이어)는 5%를 쉽게 넘는다.
- **강한 지형 결합**: s0.75(3°/μ0.6/4cm)는 **지연 0에서도 20%** — 외곽루프·예측기 전부
  무효 (μ0.6↔1.4 무차이 → 킬러는 경사+범프). RL이 개선하면 보너스, 못 하면 물리 한계.
- **왜 고전이 막히나**: Smith predictor(=지연-증강 LQR 최적)는 정확한 모델 필요 → 감축모델이
  지형·self-steering 무시 → 지형서 예측 틀림 → 보상 역효과. **model-based가 지형에 깨짐.**

## 5. RL 포지셔닝 (정직한 버전, 보강①②③ 반영)

- RL은 "자전거를 세우려고" 필요한 게 아님 — nominal은 고전이 다 함 (그 자체가 결과).
  결합×지연조차 **정확한 모델의 Smith predictor면 거의 닫힘** (이것도 결과).
- RL의 정당한 니치(정밀화) = **모델오차·지형 하의 강건성**: ① smith4는 모델오차 ≥5%에
  무보상 이하로 붕괴 — DR(파라미터·지연·지형 결합 랜덤화)로 학습한 model-free 정책이
  이 취약성을 원리적으로 회피하는지 ② s0.75+ 순수 지형(모든 고전 무효, 20%) ③ s0.5×20ms
  꼬리(52%). 단 RL도 물리 authority 한계(v=0, sub-v_min 등)는 못 넘음.
- **비교 구도 (3-way)**: RL vs base(LQR+외곽루프) vs smith4(정확모델) — 명목 평가 +
  **파라미터 랜덤화 평가**(모델오차 ±5~20% 주입) 두 축. 명목에선 smith4가 이길 수 있음;
  RL의 승부처는 랜덤화 축. 생존율뿐 아니라 **코스 유지(crosstrack)** 동일 요구
  (생존-only는 "내리막 항복" cheat 가능 — 구 baseline이 실제로 그랬음).
- 대부분 RL 논문이 약baseline과 비교하는 것과 반대 — 어느 쪽이 이겨도 정직한 기여.

## 6. 코드 맵

```
플랜트   reaction_wheel_bicycle.xml   moving_mass_bicycle.xml
공통     model.py / mm_model.py(build: 질량·스트로크·힘 변형)
제어     controller.py(리액션휠)   mm_controller.py(무게추: balance/steer/speed)
설계     lqr.py                    mm_lqr.py(4-state 해석 LQR)  *_gains.json
평가     rollout.py  mm_metrics.py(realism 채점)  mm_authority_sweep.py
         mm_envelope.py(결합×지연 하네스)  mm_plot_envelope.py
지연보상  mm_delay.py(smith4/smith6 예측기 + sysid)  mm_sysid_6state.json
검증     mm_sanity.py(회귀: ncon·trail·역진자·self-steer)
렌더     viz.py  render_bike.py  mm_render.py(MM_FORCE/LEANMAX/CAM/FLOOR/BUMP/V0/TURN_T)
```

## 7. 다음 단계

1. **RL 착수** (PPO/SAC + DR: 지형·지연·**파라미터** 결합 랜덤화) — 3-way 비교
   (base/smith4/RL), 명목 + 모델오차 랜덤화 두 평가축, mm_envelope 하네스 채점.
2. 실기 system ID (조향마찰·접촉·지연 실측) 후 sim 보정 → 실전이. (smith4를 실전
   투입하려면 등가 모델오차 <5% 필요 — 달성 가능성 자체가 측정 대상)
3. passive 자가안정 속도창(≥8 m/s 확인됨) 정밀화, Whipple 비교.
