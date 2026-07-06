# 플랜트 / passive 안정성

## 2026-07-07 — 조향축 12° caster 추가 (trail 74mm): self-steering 생성, 주행균형 달성

### 결론
v1의 수직 조향축(trail=0)이 포크잼·주행균형 실패·self-steering 부재의 공통 원인이었다.
조향축을 12° 후경(`axis="-0.2079 0 0.9781"`) → **trail 74.4mm 실측**(실차 투어링급).

### 검증 (2° 섭동, 20s 호라이즌)
1. **Self-steering 생성 [B]**: 완전 무제어에서 포크가 lean 쪽으로 스스로 꺾임
   (0.2s에 +14~46°, v0=0~4 전부 부호 일치). 제안서 핵심 메커니즘이 시뮬에 존재하게 됨.
   단 조향 감쇠가 작아(0.02) 스톱(45°)까지 오버슛 → passive 단독으론 여전히 낙하
   (자가안정 속도창 없음). 조향 frictionloss/damper 추가 검토 여지.
2. **주행균형 달성 [C]**: lean→steer PD(+센터링+감쇠) `u_st=k_ls·lean+(k_ls/4)·roll̇−2·steer−0.5·steeṙ`
   + 플라이휠 LQR + 속도PI 조합, **여러 셀이 5000/5000 (20s) STABLE**, max|steer|≤17°(잼 없음):
   발사 v0=2: k_ls=−2, +2 STABLE / 정지출발: k_ls=−5, −2 STABLE.
   trail 이전 최고는 11.4s+포크 46° 포화였음.
3. **포크잼 해소 [D]**: 구 드라이브 설정에서 steer가 더는 45°에 안 박힘(max 34°),
   v_fwd 0.18→0.55. 단 stale yaw-hold gain 때문에 9.1s에 낙하.
4. **회귀 [A]**: 구 정지균형 설정(LQR+yaw-hold 10/2)은 새 플랜트에서 **6.1s로 퇴행**
   (wheel-flop과 yaw-hold가 상충). → yaw-hold 손 gain은 폐기 대상.
   같은 정지 조건에서 [C]의 새 조향법칙(k_ls=−2)은 20s STABLE — 대체 확인.

### 미해결
- **속도 추종 여전히 실패**: STABLE 셀들도 v_fwd≈0.25m/s로 크롤링(목표 2). 발사해도 감속.
  균형과 분리된 구동/저항 문제 (스크럽·구름저항 의심).
- `lqr_gains.json`+`viz.py`의 yaw-hold 설정은 새 플랜트에서 넘어짐 — 조향법칙을
  steer각 피드백 구조로 바꿔야 (controller.heading() 개편 or MIMO LQR).

## 2026-07-06 — 무제어 낙하 + 타이어 폭의 영향 (baseline)

### 결론
무제어(ctrl=0)에서 현재 타이어(반폭 0.8cm / 전폭 1.6cm)는 **0.5° 섭동에도 1.7초 안에 넘어짐**
= 실질적 역진자. 타이어 폭을 넓히면 정적 안정 basin이 **정확히 `atan(반폭/h_CoM)`** 대로 커짐.
→ "얇은 타이어라 진짜 균형 문제"라는 설계 전제 정량 검증. LQR이 실제 일을 하고 있음이 확인됨.

### 무제어 낙하 시간 [초] (5000스텝=20s, up_z<0.7 기준)
| 반폭(전폭) | 0.5° | 1° | 2° | 5° | 10° | 정적한계 atan(w/h) |
|---|---|---|---|---|---|---|
| 0.8cm(1.6cm) ← 현재 | 1.72 | 1.39 | 1.12 | 0.80 | 0.57 | 0.9° |
| 2.0cm(4cm) | STABLE | STABLE | STABLE | 0.89 | 0.60 | 2.2° |
| 4.0cm(8cm) | STABLE | STABLE | STABLE | 1.74 | 0.70 | 4.4° |
| 8.0cm(16cm) | STABLE | STABLE | STABLE | STABLE | 0.89 | 8.7° |
| 15cm(30cm) | STABLE | STABLE | STABLE | STABLE | STABLE | 16.1° |

STABLE/낙하 경계가 예측 basin `atan(w/h)`와 거의 정확히 일치 (h=0.52m).
예: 4cm반폭 한계 4.4° → 2°는 유지, 5°는 낙하. 8cm 한계 8.7° → 5° 유지, 10° 낙하.

### 메커니즘
실린더 타이어를 forward축(x) 기준으로 기울이면 접지가 타이어 가장자리(±반폭)로 이동,
CoM 투영이 ±반폭 안이면 정적 복원. lean > atan(w/h) 넘으면 tipping. 얇을수록 basin≈0.

### 참고
- 완벽 직립(0°)에서는 대칭이라 무제어로도 안 넘어짐(불안정 평형에 정확히 얹힘) — 섭동 필요.
- 질량은 1.2kg 고정, 폭만 변수(관성만 미세 변화) → 통제된 비교.

### 파일
- `tire_test.py` — 폭×섭동 그리드, XML fromto 문자열 주입 (디스크 XML 불변)
- `render_bike.py` — `NOCONTROL=1`, `TIRE_HALFWIDTH=` env 지원 (격자바닥)
- `nocontrol_fall_16mm.mp4` — 현재 타이어 무제어 낙하 (2°→1.1s, 커밋 제외)
