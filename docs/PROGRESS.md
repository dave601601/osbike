# PROGRESS

자전거 자율균형 (제안서: 저속 moving-mass + free-fork RL vs 고전) — 진행 현황 단일 소스.

## Open (현재 상태 스냅샷)

- **연구 타겟 플랜트 가동**: `moving_mass_bicycle.xml` (free fork + 2kg/±0.15m 슬라이더).
  4-state LQR로 **v≥0.5 m/s 직진 주행 균형 20s** 성공. self-steering 동원 확인.
  **v=0은 힘/스트로크 제약 지배로 LQR 불가** — MPC/RL 명분. 설계영역 1차: 추는
  가벼울수록 유리(반직관), 최적 2kg×0.15m. 상세: [mm](progress/mm.md).
- **무게추 조향(최종 확정): 직진·~5°보정은 robust, ≥10° 선회는 chaotic.** 약한 액추에이터
  → LQR 거대게인 → 힘 85~96% 포화(bang-bang). 선회 영역에선 초기섭동 0.0001° 차이로
  낙하시각 25s↔9s 급변 = 비예측. → **LQR이 이 플랜트에 부적합함을 보이는 데이터.**
- **realism 지표(`mm_metrics.py`)**: 사람다움 채점(/5). LQR 1/5(2Hz slam), 소게인PD 2/5(sloppy).
- **★ Authority 스윕 핵심**: **힘 부족(20N)이 근본 원인**. 2kg·±0.15m·**≥50N이면 5/5
  human-like**(0.1Hz·포화0). 질량↑은 힘없이 역효과. 하드웨어 최소사양 **2kg·±0.15m·≥50N**.
- **★ 재검토(60N) 결과**:
  - ✅ **선회 chaos 해소** — 40°/20° 선회가 5/5·결정론적으로 정착. "선회 chaotic"은 20N 아티팩트.
  - ❌ **v=0 정지균형은 근본 불가** — 힘 120N·스트로크 ±1m라도 미달(travel 소진).
- **★★ 최소 균형속도 v_min ≈ 0.65 m/s** — **v≥0.65에서 human-like(5/5) 저속 균형 30s+**.
  **힘(60~200N)·스트로크(±0.15~1m) 무관** = 순전히 self-steering 한계(속도의존). v_min을 내리는
  레버는 액추에이터 아니라 **포크 trail/지오메트리**. → 고전제어로 저속(0.65 m/s)까지 다 풀림.
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
