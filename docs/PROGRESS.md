# PROGRESS

자전거 자율균형 (제안서: 저속 moving-mass + free-fork RL vs 고전) — 진행 현황 단일 소스.

> **종합 정리는 [SUMMARY.md](SUMMARY.md)** (결론 중심, 제안서용). 아래는 현재 스냅샷 + 다음 할 일.

> **경로 안내 (2026-07-26 레포 정리):** 아래 로그의 파일명은 그대로 유효하지만 위치가 바뀌었다.
> 모듈 `mm_*.py` → `src/mm/`, 스캐폴드(`model.py`·`lqr.py` 등) → `src/rw/`,
> 플랜트 XML → `assets/`, 게인·sysid JSON → `params/`, `envelope_*.json` → `results/envelopes/`.
> 실행은 `python mm_envelope.py …` → **`python src/mm/mm_envelope.py …`** (CWD 무관.
> 경로는 `mm_model.ROOT` 가 레포 루트 기준으로 해결). 영문 개요는 [../README.md](../README.md).

## Open (현재 상태 스냅샷)

**고전제어(LQR) baseline 완결 + 40-seed 엔벨로프 확정(`mm_envelope.py`).** moving-mass +
free-fork 연구 플랜트에서:
- ✅ 정지출발→가속→순항 균형, **저속 v≥0.65 m/s** 균형, 무게추 선회, 개별 외란(경사5°·
  빙판·범프5cm), **중간 결합(~s0.5, 지연 없이)** 까지 LQR로 해결. 사람다움 5/5(힘≥50N).
- ❌ 물리 한계(RL도 못 넘음): v=0 정적, sub-v_min, authority 초과(급선회·큰경사).
- ⚠️ **제어-설계 벽 (보강①②③ 후 최종)**: smith4(해석모델 Smith predictor, 정확모델)는
  결합×지연을 **거의 닫음**(s0.5: 12ms 95%·16ms 88%·20ms 52% — 구 "10-15ms 캡"은 약한
  baseline 아티팩트로 **정정**). 남는 벽 3개 = ① **s0.75+ 순수 지형**(지연0에서 20%, 외곽
  루프·예측기 무효) ② **s0.5×20ms 꼬리**(52%) ③ **모델오차 강건성** — 예측기 모델오차
  ≥5%면 smith4가 무보상보다 나빠짐(16ms 9/10→0/10). → **RL 니치 = 모델오차·지형 하
  강건성** (DR이 원리적으로 겨냥하는 지점, 실차 등가오차는 5%를 쉽게 넘음).
- **보강② 완료**: heading 폭주(경사서 yaw -66°로 항복) 진단 → lean_max 3°+heading PI(ki_yaw)
  +k_lat 외곽루프(슬루 가드)+저속 스케줄. **코스유지 |ct|중앙 30.5m→0.1m, 생존율 동등**.
  생존-only 채점은 "내리막 항복" cheat 가능 → 하네스에 코스 지표(ct/yaw) 병기.
- 핵심 수치: v_min≈0.65(pure balance; heading on ~0.70), 힘 임계≥50N, 최소 2kg·±0.15m·≥50N.
- RL 채점은 **`mm_envelope.py` 하네스 고정** (같은 지형 seed·같은 판정·코스 유지 동일 요구).
  정본 baseline = `envelope_lqr_klat.json` (어블레이션: `_yawpi`, 구 drift baseline: 무접미).

**중대 버그 2개 수정**(초기 커밋부터 잠복): frame↔앞바퀴 1.9kN 유령브레이크(접촉제외), trail=0
포크잼(caster 12°=trail 74mm). 이전 수치 중 유령브레이크 시절 것은 재측정 완료.

### 다음 할 일 (RL 착수 전 보강 ①✅②③ 순서 진행 중)
- [x] ~~결합×지연 맵 고해상도화(seed↑)~~ → **완료**: `mm_envelope.py` 40-seed, 프로토콜 고정.
- [x] ~~보강②: lateral/heading 외곽루프~~ → **완료**: heading PI+k_lat+스케줄, 코스유지
      0.1m, 결합×지연 재측정 (`envelope_lqr_klat.json`). 니치 2개 유지 확인.
- [x] ~~보강③: delay-aware 개선/방어~~ → **완료(결론 반전)**: smith4가 명목 sim에서 갭을
      거의 닫음 — 단 모델오차 ±5%에 붕괴(무보상 이하), smith6(steer 포함 sysid)은 롤포워드
      발산. `mm_delay.py`, `envelope_lqr_smith{4,6}.json`. RL 명분 = 강건성으로 정밀화.
- [x] **RL 시뮬 비교 완결** ([상세](progress/rl.md)): 최종 = res_v2@잔차0.5.
      **명목: base 전 셀 격파**(s0.5×지연 55/10/5→78/52/22%, s0.75 20→38-48%).
      **파라미터 랜덤화 축: 오차 5%부터 RL 종합 1위, 오차와 함께 격차 확대**
      (20%: RL 35.4 vs smith4 30.1 vs base 26.8 — 제안서 핵심 질문의 답).
      교훈: POMDP→4프레임 스태킹 / 무앵커 잔차=과작동 국소최적→감쇠 배포 /
      커리큘럼에 지연 0 포함 / lean 벌점 캡 / 생존-only 채점은 cheat 가능.
- [ ] 실기 system ID(조향마찰·접촉·지연 실측)→sim 보정→실전이.
- [ ] passive 자가안정 속도창(≥8 m/s 확인됨) 정밀화, Whipple 비교.

## Index (상세 — 최신 항목은 각 파일 상단)

- [SUMMARY](SUMMARY.md) — **결론 종합** (플랜트·성과·정량결과·한계·RL포지셔닝·코드맵·다음).
- [rl](progress/rl.md) — RL 파이프라인: MJX env·자체 PPO·num_envs 탐색·보상 캡 교훈·
  cold start·sim2sim. 다음 = residual RL.
- [mm](progress/mm.md) — 연구 타겟(moving-mass+free-fork) 시간순 상세: 설계·조향·힘임계·v_min·
  realism·slip·정지출발·stress-test(지연/지형/결합)·40-seed 엔벨로프 정정·외곽루프(코스유지)·
  **Smith predictor 반전(결합×지연 닫힘, 단 모델오차 ±5%에 붕괴)**·
  **액추에이터 수요 실측→NEMA17 부적합, 60N/3m/s/150-250W BLDC+FOC 권장**.
- [lqr](progress/lqr.md) — (스캐폴드) LQR 플라이휠 균형 설계 과정.
- [plant](progress/plant.md) — (스캐폴드) 유령 브레이크 제거·trail 74mm·재측정.
