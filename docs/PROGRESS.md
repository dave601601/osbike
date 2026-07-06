# PROGRESS

리액션휠 자전거 자율균형 — 진행 현황 단일 소스. (git=무엇이 바뀌었나, 여기=지금 어디인가)

## Open (현재 상태 스냅샷)

- **플랜트 v2 (trail 74mm)**: 조향축 12° caster. self-steering 존재, 포크잼 해소.
- **주행 균형 달성(크롤링)**: 플라이휠 LQR + lean→steer PD(센터링 포함)로 20s STABLE
  (발사/정지출발 모두, max|steer|≤17°). 단 v_fwd≈0.25 m/s — 2 m/s 추종은 미해결.
- **주의**: `lqr_gains.json`+`viz.py`의 yaw-hold 설정은 **plant v2에서 넘어짐**(6.1s).
  새 조향법칙은 steer각 피드백이 필요해 controller.py 구조 개편 or MIMO LQR로 흡수 예정.
- **고전 방법론 진행 중**: PID(캐스케이드) → 랜덤서치(폐기) → **LQR(플라이휠 검증)** → 다음 MIMO LQR.

### 열린 이슈 / 다음 할 일
- [ ] **MIMO LQR**: [lean,roll̇,fw,steer,steeṙ(,v)] 상태로 플라이휠+조향 동시 설계.
  yaw-hold·k_ls 손튜닝 제거. plant v2 기준으로.
- [ ] **속도 추종**: STABLE 셀도 크롤링(0.25m/s). 감속 원인(스크럽/구름저항/구동한계) 규명.
- [ ] controller.heading()을 steer각 피드백 구조로 개편 + lqr_gains.json 갱신 (viz 복구).
- [ ] free-fork 연구 타겟 전환: steer 모터 제거 + moving-mass 추가 (trail은 준비됨).
- [ ] 조향 감쇠/frictionloss 현실화 (passive 포크가 스톱까지 오버슛 — B 실험).
- [ ] 큰 섭동(>8°) 회복 한계 — 토크 예산 또는 카운터스티어 주도 재설계.
- [ ] 연구제안서(`2026 Work Station ...`) 내용과 코드 매핑 정리.

## Index (태스크별 상세 — 최신 항목은 각 파일 상단)

- [lqr](progress/lqr.md) — LQR 균형+yaw-hold 로 20s 성공. 다음은 MIMO LQR.
- [plant](progress/plant.md) — trail 74mm 추가로 self-steering 생성·주행균형 20s 달성(크롤링). 속도추종 미해결.
