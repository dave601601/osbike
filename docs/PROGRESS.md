# PROGRESS

리액션휠 자전거 자율균형 — 진행 현황 단일 소스. (git=무엇이 바뀌었나, 여기=지금 어디인가)

## Open (현재 상태 스냅샷)

- **동작하는 균형 컨트롤러 확보**: LQR 플라이휠 균형(roll 3-state) + 조향 yaw-hold(10/2).
  섭동 5°까지 정지·주행(2 m/s) 20s 유지. gain=`lqr_gains.json`, `viz.py` 우선 로드.
- **고전 방법론 진행 중**: PID(캐스케이드) → 랜덤서치(sweep) → **LQR(현재)**. 다음은 MIMO LQR.

### 열린 이슈 / 다음 할 일
- [ ] **MIMO LQR**: 조향을 상태+입력으로 포함 → yaw-hold 손튜닝 제거, roll↔yaw 결합 명시 처리.
- [ ] 큰 섭동(>8°) 회복 한계 — 토크 예산 또는 카운터스티어 주도 재설계.
- [ ] **속도 추종 약함**: 구동 시 뒷바퀴 토크 98% 포화(+4Nm)인데 v_fwd≈0.2m/s (목표 2). 바퀴가 거의 안 굴러감 — weaving 측면 스크럽/구름저항 의심. 균형과는 별개.
- [ ] lateral 위치 외곽루프(k_lat) 복원해 직진선 추종.
- [ ] `sweep.py` 정합성: balance gain 부호범위 음수로, IC를 섭동배치로 (랜덤서치 쓸 경우).
- [ ] 연구제안서(`2026 Work Station ...`) 내용과 코드 매핑 정리.

## Index (태스크별 상세 — 최신 항목은 각 파일 상단)

- [lqr](progress/lqr.md) — LQR 균형+yaw-hold 로 20s 성공. 다음은 MIMO LQR.
- [plant](progress/plant.md) — 무제어 낙하 확인, 타이어 폭 basin≈atan(w/h) 검증.
