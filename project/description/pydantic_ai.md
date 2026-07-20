# PydanticAI 백엔드 — 궁극적으로 무엇이 달라졌나

## 한 줄 요약

> **사용자가 보는 결과는 하나도 달라지지 않았고, 그 결과를 만드는 코드가 짧고 안전해졌다.**

RA·PV 어시스턴트의 LLM 모드에는 같은 일을 하는 두 개의 구현이 있다:

- **`sdk` (기본값)** — anthropic SDK 로 tool-use 루프를 **바닥부터 직접 구현** (`src/agent/agent.py` `_chat_llm()`, 약 230줄)
- **`pydantic_ai`** — 같은 루프를 **PydanticAI 프레임워크에 위임** (`src/agent/pydantic_agent.py`, 핵심 로직 약 60줄)

어느 쪽으로 실행해도 답변·출처 카드·검증 배지·PII 마스킹은 동일하다.
"달라진 게 없어 보이는 것"이 정확히 의도다 — 백엔드 교체는 **루프 구현의 교체**이지,
안전장치의 교체가 아니어야 하기 때문이다.

비유하면 **수동변속 → 자동변속**이다. 목적지(답변 품질·안전장치)는 같은데,
기어 조작(루프·메시지 포장·재시도)을 내가 하지 않는다.

---

## 1. 좋아진 것 — 손으로 짜던 코드 4덩어리가 사라졌다

| sdk 백엔드에서 직접 짠 코드 | pydantic_ai 백엔드에서는 |
|---|---|
| **에이전트 루프** — `for step in range(6):` 안에서 Claude 응답을 받고, "도구를 불러달라"(stop_reason=tool_use)인지 "답변 확정"인지 분기하고, 도구 결과를 tool_result 블록으로 포장해 대화록에 붙여 재호출 | **`agent.run()` 한 줄.** 루프가 프레임워크 안에 있다 |
| **스키마 변환기** — MCP 도구 스키마를 Anthropic tools 포맷으로 바꾸는 `_to_anthropic_tools()` | **불필요.** `MCPToolset` 이 인메모리 FastMCP 서버를 그대로 읽는다 — 변환 코드 자체가 없다 |
| **에러 재시도 배선** — 도구 실패를 `is_error=True` 로 모델에 되먹여 자가 정정을 유도 | **자동.** `ToolError`→`ModelRetry` 변환이 내장, 설정 하나(`retries=2`)로 끝 |
| **테스트 대역** — 가짜 anthropic 모듈을 손수 만들어 `sys.modules` 에 주입·유지 | **공식 제공.** `TestModel`/`FunctionModel` 로 API 키 없이 전 경로 테스트 |

덤: 모델을 다른 프로바이더로 바꿀 일이 생기면 모델 문자열 하나만 바꾸면 된다
(sdk 백엔드라면 호출 코드를 다시 짜야 한다).

## 2. 안 달라진 것 — 안전장치는 여전히 엔지니어 몫이다

이 프로젝트 고유의 안전장치는 **프레임워크가 대신해 주지 않는다.** PydanticAI 에서도
전부 직접 심어야 했고, 심는 위치만 달라졌다 — sdk 루프 본문에 흩어져 있던 것이
`_process_tool_call` 훅(도구 호출 1건마다 지나는 단일 관문) 한 곳으로 모였다.

| 안전장치 | 두 백엔드 공통 (같은 헬퍼 재사용) |
|---|---|
| 환각 검증 게이트 | 모든 답변이 `_finalize()` 통과 — 수치·날짜·인용 버전을 도구 출력과 대조 |
| 신뢰 소스 규율 | 사용자 입력 에코(query·as_of)는 근거로 승격 금지, 에러 계약(`{"error":...}`) 제외 |
| grounded 판정 | '내용 있는 도구 출력'이 있어야 근거 배지 — 빈 검색 성공은 근거가 아니다 |
| 폐지본 인용 통제 | 이력 검색이 실제 반환한 문서만 경고 면제 |
| PII 마스킹 | 백엔드 분기 **이전** 공통 입구에서 — 백엔드는 마스킹본만 받는다 |
| 실패 계약 | API 실패 시 크래시 대신 안내문, 예외 타입명만 노출 |

"같은 계약"은 말이 아니라 테스트로 고정되어 있다: `tests/test_agent_llm.py`(sdk)와
`tests/test_agent_pydantic_ai.py`(pydantic_ai)가 **같은 시나리오**를 각자 실행으로 검증한다.

## 3. 결론 

> **"프레임워크는 반복되는 배관 작업(루프·스키마 변환·재시도)을 없애주지만,
> 도메인 안전장치(검증·신뢰소스·PII)는 여전히 엔지니어 몫이다.

이것이 병렬 백엔드를 만든 궁극적 목적이다: 직접 구현으로 **원리**를 증명하고,
PydanticAI 적용으로 **프레임워크 활용**을 증명하며, 두 구현이 같은 계약 테스트를
통과하는 것으로 "프레임워크가 해주는 일과 내가 해야 하는 일의 경계"를 코드로 보인다.

---

## 4. 사용법

```bash
# .env 또는 셸에서 (LLM 모드 전용 — ANTHROPIC_API_KEY 필요)
AGENT_BACKEND=pydantic_ai   # PydanticAI 백엔드
AGENT_BACKEND=sdk           # (또는 미설정) 직접 구현 루프
```

| ANTHROPIC_API_KEY | AGENT_BACKEND | 실행 경로 |
|---|---|---|
| 없음 | (무관) | 오프라인 규칙 라우터 — 백엔드 선택은 의미 없음 |
| 있음 | 없음 또는 `sdk` | 직접 구현 루프 (`_chat_llm`) |
| 있음 | `pydantic_ai` | PydanticAI (`chat_llm_pydantic`) |

**적용 확인 3가지:**
1. 기동 배너(및 `/health`의 `banner`)에 `· backend=pydantic_ai` 표기
2. 흐름 로그의 진입 함수: `_chat_llm()` ↔ `chat_llm_pydantic()`
3. trace 의 llm 스팬에 `"backend": "pydantic_ai"`

가장 효과적인 시연: 같은 질문을 두 백엔드로 한 번씩 던져, **로그(만드는 과정)는 갈리고
응답(출처 카드·검증 배지)은 동일**한 것을 나란히 보여주는 것.

## 5. 알아두면 좋은 세부 차이

- **루프 상한 계산**: sdk 는 '왕복 6회' 단일 카운터, pydantic_ai 는 모델 요청 상한
  (`UsageLimits(request_limit=6)`)과 도구별 재시도(`retries=2`)가 분리. "상한이 있다"는
  계약과 초과 시 안내 문구는 같지만 회계 방식은 프레임워크 관례를 따랐다.
- **상한 초과여도 증거는 보존**: 초과 시점까지 쌓인 출처·신뢰 소스는 실패 답변에도 붙는다.
- **대화 이력**: 두 백엔드 모두 이전 턴의 텍스트만 모델에 이월한다 (pydantic_ai 는
  dict → `ModelMessage` 객체 변환을 거침).

## 6. 파일 지도

| 파일 | 역할 |
|---|---|
| `src/config.py` (`AGENT_BACKEND`) | 백엔드 선택 스위치 (기본 `sdk`) |
| `src/agent/agent.py` (`RaAgent.chat`) | 공통 입구(PII 마스킹)·백엔드 분기·공유 헬퍼 정의처 |
| `src/agent/agent.py` (`_chat_llm`) | sdk 백엔드 — 직접 구현 루프 |
| `src/agent/pydantic_agent.py` | pydantic_ai 백엔드 — 증거 바구니(`PaDeps`)·북키핑 훅(`_process_tool_call`)·진입점 |
| `tests/test_agent_llm.py` · `tests/test_agent_pydantic_ai.py` | 두 백엔드의 동일 계약 테스트 |
| `requirements.txt` | `pydantic-ai-slim[anthropic,mcp]==2.13.0` (선택 의존성) |
