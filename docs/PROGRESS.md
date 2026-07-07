# PROGRESS

자전거 자율균형 (제안서: 저속 moving-mass + free-fork RL vs 고전) — 진행 현황 단일 소스.

> **종합 정리는 [SUMMARY.md](SUMMARY.md)** (결론 중심, 제안서용). 아래는 현재 스냅샷 + 다음 할 일.

## Open (현재 상태 스냅샷)

**고전제어(LQR) baseline 완결 + 40-seed 엔벨로프 확정(`mm_envelope.py`).** moving-mass +
free-fork 연구 플랜트에서:
- ✅ 정지출발→가속→순항 균형, **저속 v≥0.65 m/s** 균형, 무게추 선회, 개별 외란(경사5°·
  빙판·범프5cm), **중간 결합(~s0.5, 지연 없이)** 까지 LQR로 해결. 사람다움 5/5(힘≥50N).
- ❌ 물리 한계(RL도 못 넘음): v=0 정적, sub-v_min, authority 초과(급선회·큰경사).
- ⚠️ **제어-설계 벽 (40 seeds 확정)**: ① **s0.5 결합 × 지연** — 88%@8ms→48%@12ms→0%@20ms
  (실차 지연대에 마진 없음, model-based 지연보상은 지형에 깨짐) ② **s0.75+ 순수 지형 결합**
  — 지연 0에서도 25% (신규 발견, 구 3-seed 맵의 66%는 착시). → **RL 니치 2개**.
- 핵심 수치: v_min≈0.65(힘·스트로크 무관), 힘 임계≥50N, 하드웨어 최소 2kg·±0.15m·≥50N.
- RL 채점은 **`mm_envelope.py` 하네스 고정** (같은 지형 seed·같은 판정으로 맵 겹치기).

**중대 버그 2개 수정**(초기 커밋부터 잠복): frame↔앞바퀴 1.9kN 유령브레이크(접촉제외), trail=0
포크잼(caster 12°=trail 74mm). 이전 수치 중 유령브레이크 시절 것은 재측정 완료.

### 다음 할 일 (RL 착수 전 보강 ①✅②③ 순서 진행 중)
- [x] ~~결합×지연 맵 고해상도화(seed↑)~~ → **완료**: `mm_envelope.py` 40-seed, 프로토콜 고정.
- [ ] **보강②**: lateral/heading 외곽루프(k_lat) — 경사 드리프트 잡기 → 결합×지연 셀 재확인.
- [ ] **보강③**: delay-aware(Smith predictor)를 steer 포함 모델로 개선 — 또는 지형 하
      예측오차 직접 증거로 "model-based 캡" 주장 방어.
- [ ] **RL 착수** (PPO/SAC + domain randomization, 지형·지연 결합) — s0.5×지연 및 s0.75
      지형 gap을 넘는지. 채점은 mm_envelope.py 하네스.
- [ ] 실기 system ID(조향마찰·접촉·지연 실측)→sim 보정→실전이.
- [ ] passive 자가안정 속도창(≥8 m/s 확인됨) 정밀화, Whipple 비교.

## Index (상세 — 최신 항목은 각 파일 상단)

- [SUMMARY](SUMMARY.md) — **결론 종합** (플랜트·성과·정량결과·한계·RL포지셔닝·코드맵·다음).
- [mm](progress/mm.md) — 연구 타겟(moving-mass+free-fork) 시간순 상세: 설계·조향·힘임계·v_min·
  realism·slip·정지출발·stress-test(지연/지형/결합)·delay-aware 상한·**40-seed 엔벨로프 정정**.
- [lqr](progress/lqr.md) — (스캐폴드) LQR 플라이휠 균형 설계 과정.
- [plant](progress/plant.md) — (스캐폴드) 유령 브레이크 제거·trail 74mm·재측정.
