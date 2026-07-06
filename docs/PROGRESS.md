# PROGRESS

리액션휠 자전거 자율균형 — 진행 현황 단일 소스. (git=무엇이 바뀌었나, 여기=지금 어디인가)

## Open (현재 상태 스냅샷)

- **플랜트 v2.1**: trail 74mm(caster 12°) + **frame↔front_wheel 접촉 제외**(1.9kN 유령
  브레이크 제거 — 초기 커밋부터 있던 버그, 모든 구동 문제의 진범이었음).
- **정지+주행 균형+속도추종 전부 동작**: 플라이휠 LQR + 조향(센터링+lean PD) + 속도 PI로
  정지 균형 20s, 정지출발→4s에 2 m/s 도달 후 추종, 주행 균형 20s STABLE (k_ls=0도 됨).
- **주의**: 유령 브레이크 이전의 정량 수치(8° 한계, passive basin, self-steering 응답,
  trail 배터리 A~D)는 오염 — 인용 전 재측정. `lqr_gains.json`+`viz.py`는 여전히 stale
  (구 yaw-hold 법칙) → controller.py 개편 필요.
- **고전 방법론**: PID 캐스케이드+LQR 균형이 전 시나리오 동작. 다음 후보: MIMO LQR(선택),
  MPC, 그리고 RL 비교 준비.

### 열린 이슈 / 다음 할 일
- [x] ~~controller.py 조향법칙 정착~~ → `steer()` = 예측 lean + 센터링, 실코드 경로 3종 검증.
- [x] ~~오염 수치 재측정~~ → 정지한계 7°(이론 8.4° 정합), **주행 2m/s는 10°+ 회복**,
  self-steering 물리 정상(기운 쪽), 타이어 basin 결론 유지.
- [ ] free-fork 연구 타겟 전환: steer 모터 제거 + moving-mass 추가 (trail 준비됨).
  새 body 추가 시 d.ncon 덤프로 내부 접촉 확인 (유령 브레이크 재발 방지).
- [ ] MIMO LQR (선택): 손튜닝 항(k_ls, 센터링) 원리적 대체 + 성능 한계 탐색용.
- [ ] 조향 감쇠/frictionloss 현실화 (제안서 2단계 system ID 항목과 연결).
- [ ] passive 자가안정 속도창 스윕 (2m/s엔 없음 — 3~6m/s 탐색, Whipple 비교).
- [ ] 연구제안서(`2026 Work Station ...`) 내용과 코드 매핑 정리.

## Index (태스크별 상세 — 최신 항목은 각 파일 상단)

- [lqr](progress/lqr.md) — LQR 균형+yaw-hold 로 20s 성공. 다음은 MIMO LQR.
- [plant](progress/plant.md) — 유령 브레이크(frame↔앞바퀴 1.9kN) 제거로 속도추종까지 전부 해결. trail 74mm.
