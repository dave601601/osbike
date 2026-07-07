# PROGRESS

자전거 자율균형 (제안서: 저속 moving-mass + free-fork RL vs 고전) — 진행 현황 단일 소스.

## Open (현재 상태 스냅샷)

- **연구 타겟 플랜트 가동**: `moving_mass_bicycle.xml` (free fork + 2kg/±0.15m 슬라이더).
  4-state LQR로 **v≥0.5 m/s 주행 균형 20s** + **무게추 조향(heading 캐스케이드)으로
  완만 선회** 성공. self-steering 동원 확인.
  **v=0은 힘/스트로크 제약 지배로 LQR 불가** — MPC/RL 명분. 설계영역 1차: 추는
  가벼울수록 유리(반직관), 최적 2kg×0.15m. 상세: [mm](progress/mm.md).
- **무게추 조향은 완만해야만 robust** (CPU 기준): slew ≤3°/s면 목표 heading 60°까지
  lean 3~4°로 clean. slew↑는 marginal(lean 25°)·전복, **방향반전(S커브)은 authority 초과**.
- **수치 marginality 경보**: 내부루프 F포화 87% ≈ bang-bang → **CPU-JAX vs GPU-JAX 궤적
  상이**. 성능은 배포경로(CPU)로만 주장. 학습(GPU)→실기(CPU) sim2sim 갭 주의.
- 다음 본론: **PPO/SAC 학습 → LQR과 동일 지표 비교** (v_min, 회복한계, 조향추종, 외란).

### reaction-wheel 스캐폴드 (완료 상태)

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

- [mm](progress/mm.md) — **연구 타겟**: free-fork+moving-mass, v≥0.5 균형 성공 / v=0 제약 지배 실패 / 설계영역 1차.
- [lqr](progress/lqr.md) — (스캐폴드) LQR 플라이휠 균형 설계 과정.
- [plant](progress/plant.md) — (스캐폴드) 유령 브레이크 제거·trail 74mm·재측정.
