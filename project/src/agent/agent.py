"""RA 어시스턴트 에이전트 (Agentic Workflow + Function Calling).

에이전트가 MCP 서버의 도구를 자율적으로 호출해 사용자의 규제업무 질문에 답한다.

두 가지 실행 경로 (동일 인터페이스):
  - LLM 모드 (ANTHROPIC_API_KEY 있음): 실제 Claude가 tool-use 루프로
    "관찰→생각→도구호출→관찰" 을 반복하며 스스로 도구를 선택·조합한다. (진짜 Agentic)
  - 오프라인 모드 (키 없음): 규칙 기반 라우터가 적절한 MCP 도구를 호출하고,
    검색 근거에 기반한 추출형 답변을 조립한다. (환각 없는 grounded 답변)

두 경로 모두 '도구는 MCP를 통해' 호출한다 — 즉 GC의 Hey.GC 2.0 구조와 동일하게
모델과 도구가 MCP 규격으로 분리되어 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from .. import config
from ..mcp_server.server import mcp
from ..observability import Trace, timed
from ..pv.redactor import redact
from ..rag.synonyms import expand_query
from ..rag.textutil import tokenize
from ..verify.verifier import verify_answer, warning_text

# 근거 부족(abstention) 판단 임계값 — 제약 규제 도메인의 환각 안전장치.
# 두 신호를 함께 본다: (1) 최상위 근거의 검색 관련도 점수, (2) 질의-근거 토큰 커버리지.
# 둘 다 문턱 아래일 때만("확실히 근거 없음") 답을 지어내지 않고 회피한다.
# AND 조건을 쓰는 이유: 콜로퀄한 범위내 질문은 점수는 낮아도 커버리지가 남아있어
# 과도한 회피(over-abstention)를 피할 수 있다.
#
# 문턱값은 범위내(qa_dataset)/범위밖(abstention_dataset) 분포를 실측해 마진
# 중앙으로 보정한 값이다(리랭커 공식이 바뀌면 점수 스케일이 바뀌므로 재보정 필수):
#   범위밖 최대  score=0.167, cov=0.250  → 이보다 위에 문턱
#   범위내 최소  score=0.156(cov 0.314) / cov=0.227(score 0.201)
#   → SCORE_FLOOR ∈ (0.167, 0.201), COVERAGE_FLOOR ∈ (0.250, 0.314)
# 리랭커 v3(섹션 타입 prior) 반영 후 재실측(32문항): 페널티는 대조/서두 섹션에만
# 작용해 경계 분포가 사실상 불변(범위내 최소 0.156/0.227, 범위밖 최대 0.167/0.250)
# — 문턱 유지. 재보정 절차 자체는 리랭커를 바꿀 때마다 반복한다.
SCORE_FLOOR = 0.19       # 최상위 근거 관련도 하한
COVERAGE_FLOOR = 0.28    # (확장)질의 토큰 커버리지 하한

SYSTEM_PROMPT = (
    "당신은 제약회사 RA(인허가·규제업무)·PV(약물감시) 담당자를 돕는 어시스턴트다. "
    "반드시 제공된 MCP 도구로 사내 규제문서와 업무 데이터를 조회한 뒤, "
    "그 근거에 기반해서만 답한다. 근거가 없으면 모른다고 답하고 추측하지 않는다. "
    "규제 정보(기한·절차)는 정확성이 생명이므로 항상 출처(문서명·섹션)를 함께 제시한다. "
    "답변은 한국어로, 간결한 실무체로 작성한다."
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result_summary: str


@dataclass
class AgentResult:
    answer: str
    mode: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    grounded: bool = True          # 검색 근거로 뒷받침된 답변인지(abstention이면 False)
    trace: list[dict] = field(default_factory=list)   # 스텝별 지연·성패(관측성)
    latency_ms: float = 0.0        # 총 처리 지연
    redactions: list[dict] = field(default_factory=list)  # PII 마스킹 내역(유형·건수만)
    verification: dict = field(default_factory=dict)  # 사후 검증 결과(수치 대조·버전 점검)


# ---------------------------------------------------------------------------
# 공통: MCP 도구 스키마 → Anthropic 도구 포맷
# ---------------------------------------------------------------------------
def _to_anthropic_tools(mcp_tools) -> list[dict]:
    tools = []
    for t in mcp_tools:
        tools.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
        )
    return tools


def _collect_citations(tool_calls: list[ToolCall], raw_results: list[dict]) -> list[dict]:
    """search_regulations 결과에서 출처 목록을 모은다."""
    cites: list[dict] = []
    seen = set()
    for res in raw_results:
        for r in res.get("results", []):
            key = (r.get("source"), r.get("section"))
            if key in seen:
                continue
            seen.add(key)
            cites.append(
                {
                    "doc_id": r.get("doc_id"),
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "section": r.get("section"),
                    "version": r.get("version"),
                    "effective_date": r.get("effective_date"),
                    "status": r.get("status"),  # 사후 검증(폐지본 인용 감지)에 사용
                    "score": r.get("score"),
                }
            )
    return cites


def _strip_query_echo(data: Any) -> Any:
    """도구 출력에서 질의 에코 필드(query)를 재귀 제거한 사본을 만든다.

    신뢰 소스는 '근거 문단 ∪ 도구 출력'인데, search 계열 도구는 받은 질의를
    출력에 에코한다. 이를 그대로 신뢰하면 **사용자 질문 속 수치가 신뢰 소스로
    승격**되어, 사용자가 틀린 수치를 전제로 물었을 때 모델이 맞장구쳐도
    검증을 통과하는 구멍이 생긴다. 케이스 서술(case)은 남긴다 — 그것은
    검증할 규정 클레임이 아니라 사용자가 제공한 '사실'이다.
    """
    if isinstance(data, dict):
        return {k: _strip_query_echo(v) for k, v in data.items() if k != "query"}
    if isinstance(data, list):
        return [_strip_query_echo(v) for v in data]
    return data


def _finalize(
    result: AgentResult,
    trusted_texts: list[str],
    allow_superseded: bool = False,
    question: str = "",
) -> AgentResult:
    """모든 응답이 통과하는 사후 검증 게이트 — 답변 속 수치·날짜·방향 한정어를
    신뢰 소스(검색 근거 + 결정론적 도구 출력, 질문 에코 제외)와 대조하고
    폐지본 인용을 점검한다.

    실패해도 답변을 차단하지 않는다 — 경고를 본문에 부착하고 verification
    필드로 노출한다(시끄러운 실패, 최종 확정은 사람). 오프라인 모드는 추출형이라
    통과가 정상이며, 통과 자체가 '포매터가 근거 밖 수치를 만들지 않는다'는
    회귀 가드가 된다. LLM 모드는 생성 답변이므로 이 게이트가 실질 방어선이다.
    question 을 넘기는 이유: 미확인 수치가 질문에 있던 값이면 '환각'이 아니라
    '전제 확인 필요'로 경고 문구를 조정한다(정정 답변의 오탐 완화).
    """
    v = verify_answer(result.answer, trusted_texts, result.citations, allow_superseded, question=question)
    if not v.ok:
        result.answer = f"{result.answer}\n\n{warning_text(v)}"
    result.verification = v.summary()
    return result


class RaAgent:
    """RA 어시스턴트. chat() 하나로 두 모드를 투명하게 처리."""

    async def chat(self, message: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        trace = Trace()
        # PII 마스킹은 '에이전트 입구'에서 — 이후의 모든 경로(외부 LLM API·검색·
        # 로그·트레이스)에 원문 개인정보가 흘러들지 않는다. 원 값은 보존하지 않는다.
        red = redact(message)
        message = red.text
        history = _redact_history(history)  # 클라이언트가 보낸 이전 턴에도 PII가 있을 수 있다
        with timed(trace, "chat", "agent", {"mode": "llm" if config.LLM_AVAILABLE else "offline"}):
            if config.LLM_AVAILABLE:
                result = await self._chat_llm(message, history, trace)
            else:
                result = await self._chat_offline(message, history, trace)
        result.trace = trace.to_list()
        result.latency_ms = trace.total_ms
        result.redactions = red.summary()
        return result

    # ---- LLM 모드: 실제 tool-use 루프 ----
    async def _chat_llm(self, message: str, history: list[dict], trace: Trace) -> AgentResult:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        tool_calls: list[ToolCall] = []
        raw_search_results: list[dict] = []
        trusted_texts: list[str] = []   # 사후 검증의 신뢰 소스(성공한 도구 출력 전체)
        allow_superseded = False        # 사용자가 명시적으로 이력(폐지본) 조회를 했는가

        async with Client(mcp) as mcp_client:
            tools = _to_anthropic_tools(await mcp_client.list_tools())
            messages: list[dict] = [*history, {"role": "user", "content": message}]

            # 에이전트 루프 (최대 6스텝 — 무한루프 방지)
            for step in range(6):
                with timed(trace, f"llm.step{step}", "llm", {"model": config.LLM_MODEL}):
                    resp = await client.messages.create(
                        model=config.LLM_MODEL,
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        tools=tools,
                        messages=messages,
                    )
                if resp.stop_reason != "tool_use":
                    text = "".join(
                        b.text for b in resp.content if b.type == "text"
                    ).strip()
                    # 생성 답변은 매번 다르므로 평가셋이 아니라 '이 답변'을 검증한다
                    return _finalize(
                        AgentResult(
                            answer=text,
                            mode="llm",
                            tool_calls=tool_calls,
                            citations=_collect_citations(tool_calls, raw_search_results),
                        ),
                        trusted_texts,
                        allow_superseded,
                        question=message,
                    )

                # 도구 호출 실행
                messages.append({"role": "assistant", "content": resp.content})
                tool_results_content = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    data, is_error = await self._safe_tool_call(
                        mcp_client, block.name, dict(block.input), trace
                    )
                    if not is_error:
                        # 도구 출력 = 검증 신뢰 소스. 단 질의 에코(query)는 제외 —
                        # 사용자 전제가 신뢰 소스로 승격되는 구멍을 막는다.
                        trusted_texts.append(_stringify(_strip_query_echo(data)))
                    if not is_error and isinstance(data, dict):
                        if block.name == "search_regulations":
                            raw_search_results.append(data)
                            if block.input.get("include_superseded") or block.input.get("as_of"):
                                allow_superseded = True  # 이력 조회 의도 — 폐지본 인용은 결함 아님
                        elif block.name in ("assess_adverse_event", "draft_ae_report") and data.get("basis"):
                            raw_search_results.append(data["basis"])  # 트리아지/초안 근거 규정도 출처로
                    tool_calls.append(
                        ToolCall(
                            name=block.name,
                            args=dict(block.input),
                            result_summary=("오류: " + str(data)) if is_error else _summarize(data),
                        )
                    )
                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _stringify(data),
                            # 도구가 실패해도 크래시 대신 에러를 모델에 되먹여
                            # 스스로 복구(다른 도구/인자로 재시도)하게 한다.
                            "is_error": is_error,
                        }
                    )
                messages.append({"role": "user", "content": tool_results_content})

            # 루프 한계 도달
            return _finalize(
                AgentResult(
                    answer="(도구 호출이 반복되어 응답을 확정하지 못했습니다. 질문을 좁혀 다시 시도해 주세요.)",
                    mode="llm",
                    tool_calls=tool_calls,
                    citations=_collect_citations(tool_calls, raw_search_results),
                ),
                trusted_texts,
                allow_superseded,
                question=message,
            )

    async def _safe_tool_call(self, mcp_client, name: str, args: dict, trace: Trace):
        """MCP 도구를 지연 측정과 함께 호출하고, 실패해도 크래시하지 않는다.

        Returns: (data, is_error). 실패 시 data는 에러 메시지 문자열.
        """
        try:
            with timed(trace, f"tool.{name}", "tool", {"args": args}):
                call = await mcp_client.call_tool(name, args)
            return call.data, False
        except Exception as e:  # noqa: BLE001 - 도구 실패는 응답으로 흡수(견고성)
            return f"도구 '{name}' 실행 실패: {type(e).__name__}: {e}", True

    # ---- 오프라인 모드: 규칙 라우팅 + grounded 추출 답변 ----
    async def _chat_offline(self, message: str, history: list[dict], trace: Trace) -> AgentResult:
        tool_calls: list[ToolCall] = []
        # 멀티턴: 짧은 후속질문("그럼 그건 며칠?")은 직전 사용자 발화와 병합해 맥락 복원
        resolved = _resolve_followup(message, history)
        async with Client(mcp) as mcp_client:
            intent = _route_intent(resolved)

            if intent == "ae_report":
                # 케이스 서술 + '보고서 작성' 요청 → ICSR 초안 도구(트리아지+인과성+코딩+최소요건)
                args = {"case_description": resolved}
                with timed(trace, "tool.draft_ae_report", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("draft_ae_report", args)).data
                tool_calls.append(ToolCall("draft_ae_report", args, _summarize(data)))
                return _finalize(
                    AgentResult(
                        answer=_format_report(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(_strip_query_echo(data))],
                    question=resolved,
                )

            if intent == "ae_triage":
                # 구체적 케이스 서술 → PV 트리아지 도구(중대성 판정+기한 계산)
                args = {"case_description": resolved}
                with timed(trace, "tool.assess_adverse_event", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("assess_adverse_event", args)).data
                tool_calls.append(ToolCall("assess_adverse_event", args, _summarize(data)))
                return _finalize(
                    AgentResult(
                        answer=_format_triage(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(_strip_query_echo(data))],
                    question=resolved,
                )

            if intent == "deadlines":
                args = {"within_days": 30}
                with timed(trace, "tool.get_ra_deadlines", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_ra_deadlines", args)).data
                tool_calls.append(ToolCall("get_ra_deadlines", args, _summarize(data)))
                answer = _format_deadlines(data)
                return _finalize(
                    AgentResult(answer=answer, mode="offline", tool_calls=tool_calls),
                    [_stringify(data)],
                    question=resolved,
                )

            if intent == "checklist":
                category = _guess_category(resolved)
                args = {"category": category}
                with timed(trace, "tool.get_submission_checklist", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_submission_checklist", args)).data
                tool_calls.append(ToolCall("get_submission_checklist", args, _summarize(data)))
                answer = _format_checklist(data)
                return _finalize(
                    AgentResult(answer=answer, mode="offline", tool_calls=tool_calls),
                    [_stringify(data)],
                    question=resolved,
                )

            # 기본: 규제문서 검색 후 근거 기반 답변
            args = {"query": resolved, "top_n": 3}
            with timed(trace, "tool.search_regulations", "tool", {"args": args}):
                data = (await mcp_client.call_tool("search_regulations", args)).data
            tool_calls.append(ToolCall("search_regulations", args, _summarize(data)))

            # 근거 충분성(grounding) 판정 → 부족하면 환각 대신 abstention
            results = data.get("results", [])
            top_score = results[0].get("score", 0.0) if results else 0.0
            coverage = _grounding_coverage(resolved, results)
            if top_score < SCORE_FLOOR and coverage < COVERAGE_FLOOR:
                return _finalize(
                    AgentResult(
                        answer=(
                            "제공된 사내 규제문서에서 이 질문에 답할 근거를 찾지 못했습니다. "
                            "질문을 규제업무(허가·변경·GMP·라벨링·약물감시·임상) 범위로 좁혀 다시 시도해 주세요."
                        ),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=[],
                        grounded=False,
                    ),
                    [],  # 회피 답변은 수치 클레임이 없어 자명하게 통과 — 그래야 정상
                    question=resolved,
                )
            answer = _format_search_answer(data)
            return _finalize(
                AgentResult(
                    answer=answer,
                    mode="offline",
                    tool_calls=tool_calls,
                    citations=_collect_citations(tool_calls, [data]),
                    grounded=True,
                ),
                [_stringify(_strip_query_echo(data))],
                question=resolved,
            )


# ---------------------------------------------------------------------------
# 보조 함수 (오프라인 포매팅 · 라우팅)
# ---------------------------------------------------------------------------
_FOLLOWUP_MARKERS = ("그건", "그거", "그럼", "그게", "이건", "위", "방금", "아까", "그 경우", "그 때")


def _redact_history(history: list[dict]) -> list[dict]:
    """대화 이력의 문자열 콘텐츠에서 PII를 마스킹한다(외부 API로 나가기 전 방어)."""
    out: list[dict] = []
    for turn in history:
        content = turn.get("content")
        if isinstance(content, str):
            out.append({**turn, "content": redact(content).text})
        else:
            out.append(turn)
    return out


def _last_user_text(history: list[dict]) -> str:
    """대화 이력에서 마지막 사용자 발화 텍스트를 추출(멀티턴 맥락 복원용)."""
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # anthropic 블록 형식 대비
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                return " ".join(parts)
    return ""


def _resolve_followup(message: str, history: list[dict]) -> str:
    """짧거나 지시대명사로 시작하는 후속질문을 직전 발화와 병합한다.

    예) 이전: "변경허가 처리기한은?" / 현재: "그럼 변경신고는?"
        → 검색 질의를 "변경허가 처리기한은? 그럼 변경신고는?" 로 확장해 맥락 유지.
    오프라인 규칙 라우팅에서도 대명사 후속질문이 동작하게 하는 경량 처리.
    """
    if not history:
        return message
    m = message.strip()
    is_short = len(m) <= 12
    has_marker = any(mark in m for mark in _FOLLOWUP_MARKERS)
    if is_short or has_marker:
        prev = _last_user_text(history)
        if prev:
            return f"{prev} {message}".strip()
    return message


def _grounding_coverage(query: str, results: list[dict]) -> float:
    """질의 토큰 중 검색된 근거 텍스트에 실제로 등장한 비율(0~1).

    낮으면 검색이 답할 '재료'를 못 찾은 것 → abstention 신호.
    (LLM 모드에서 생성된 답의 faithfulness 를 사후 점검하는 데도 같은 지표를 쓴다.)

    커버리지는 '확장 질의' 기준으로 계산한다. 검색 자체가 동의어 확장
    ("부작용"→"이상사례")으로 정답을 찾는데, 회피 판정만 원 질의 토큰으로
    보면 정답을 찾아 놓고도 "근거 없음"이라 답하는 자기모순(과회피)이 생긴다
    — 범위밖 질문은 도메인 용어가 없어 확장되지 않으므로 이 보정은
    범위밖 회피 정확도를 해치지 않는다(faithfulness eval 로 검증).
    """
    q_tokens = set(tokenize(expand_query(query)))
    if not q_tokens:
        return 0.0
    ctx = "\n".join(r.get("text", "") for r in results)
    ctx_tokens = set(tokenize(ctx))
    return len(q_tokens & ctx_tokens) / len(q_tokens)


_AE_CASE_MARKERS = ("환자", "복용 후", "투여 후", "복용했", "투여했", "접종 후")
_AE_EVENT_MARKERS = (
    "사망", "입원", "생명", "쇼크", "아나필락시스", "중환자실", "기형", "장애",
    "부작용", "이상사례", "이상반응",
)


_AE_REPORT_MARKERS = ("보고서", "초안", "kaers", "icsr", "보고서 작성", "보고서 만들")


def _route_intent(message: str) -> str:
    m = message.lower()
    # AE 트리아지: '구체적 케이스 서술'일 때만 (환자/복용 맥락 + 사건 어휘).
    # "중대한 이상사례는 며칠 안에 보고?" 같은 '규정 질문'은 search 로 남긴다.
    if any(k in m for k in _AE_CASE_MARKERS) and any(k in m for k in _AE_EVENT_MARKERS):
        # 같은 케이스라도 '보고서 작성' 요청이면 초안 도구로(트리아지는 판정만).
        if any(k in m for k in _AE_REPORT_MARKERS):
            return "ae_report"
        return "ae_triage"
    if any(k in m for k in ["마감", "기한", "일정", "언제까지", "d-day", "디데이", "며칠", "남은"]):
        # '보고 기한' 같은 규정 질문과 구분: 업무 일정 신호가 강할 때만
        if any(k in m for k in ["마감", "일정", "남은", "이번 주", "오늘", "디데이", "d-day"]):
            return "deadlines"
    if any(k in m for k in ["체크리스트", "준비물", "구비서류", "무엇이 필요", "뭐가 필요", "준비할"]):
        return "checklist"
    return "search"


def _guess_category(message: str) -> str:
    m = message
    if "변경" in m:
        return "변경허가"
    if any(k in m for k in ["안전", "이상사례", "부작용", "보고"]):
        return "안전성보고"
    return "품목허가"


def _format_deadlines(data: dict) -> str:
    if not data.get("deadlines"):
        return f"오늘({data.get('today')}) 기준 임박한 규제 업무 마감이 없습니다."
    lines = [f"오늘({data['today']}) 기준 다가오는 규제 업무 마감입니다:", ""]
    for d in data["deadlines"]:
        dday = d["d_day"]
        tag = "🔴 지남/오늘" if dday <= 0 else ("🟠 임박" if dday <= 7 else "🟢")
        lines.append(f"- {tag} D{dday:+d} · {d['due_date']} · {d['item']} ({d['type']}, {d['owner']}, {d['status']})")
    return "\n".join(lines)


def _format_checklist(data: dict) -> str:
    if "error" in data:
        return f"{data['error']}. 지원 항목: {', '.join(data['available'])}"
    lines = [f"[{data['category']}] 준비 체크리스트:", ""]
    lines += [f"{i}. {item}" for i, item in enumerate(data["items"], 1)]
    return "\n".join(lines)


def _format_triage(data: dict) -> str:
    lines = ["📋 이상사례(AE) 트리아지 결과", ""]
    if data.get("is_serious"):
        lines.append(f"- 중대성 판정: **중대(Serious)** — 충족 기준: {', '.join(data.get('criteria_met', []))}")
    else:
        lines.append("- 중대성 판정: 비중대 (중대성 기준 미감지)")
    lines.append(f"- 보고 경로: {data.get('route', '')}")
    if data.get("deadline_date"):
        lines.append(
            f"- 보고 기한: {data['deadline_date']} (인지일 {data.get('awareness_date')} 기준"
            + (", 지체 없이)" if data.get("deadline_days") == 0 else f", {data.get('deadline_days')}일 이내)")
        )
    lines.append(f"- 판정 사유: {data.get('rationale', '')}")
    if data.get("coded_terms"):
        coded = " · ".join(f"{t['verbatim']}→{t['pt']}({t['pt_en']})" for t in data["coded_terms"])
        lines.append(f"- 표준 용어 코딩(MedDRA 방식): {coded}")
    cz = data.get("causality") or {}
    if cz.get("suggested"):
        lines.append(f"- 인과성(WHO-UMC) 제안: {cz['suggested']} — {cz.get('rationale', '')}")
        if cz.get("missing_info"):
            lines.append(f"  · 보고자 확인 필요: {'; '.join(cz['missing_info'][:2])} 등")
    if data.get("pii_masked"):
        masked = ", ".join(f"{m['type']} {m['count']}건" for m in data["pii_masked"])
        lines.append(f"- 🔒 개인정보 마스킹: {masked}")
    for c in data.get("caveats", []):
        lines.append(f"- ⚠ {c}")
    return "\n".join(lines)


def _format_report(data: dict) -> str:
    """draft_ae_report 결과 → 초안 본문 + 보완 안내."""
    lines = [data.get("draft_markdown", "").strip()]
    followups = data.get("followups", [])
    if followups:
        lines += ["", "📌 보고 확정 전 확인/보완할 항목:"]
        lines += [f"- {f}" for f in followups]
    if data.get("pii_masked"):
        masked = ", ".join(f"{m['type']} {m['count']}건" for m in data["pii_masked"])
        lines += ["", f"🔒 개인정보 마스킹: {masked} (초안에는 비식별 서술만 포함)"]
    return "\n".join(lines)


def _format_search_answer(data: dict) -> str:
    results = data.get("results", [])
    if not results:
        return "관련 규정을 찾지 못했습니다. 질문을 바꿔 다시 시도해 주세요."
    # grounded 추출: 최상위 근거를 중심으로 요약 제시(오프라인이라 생성 대신 발췌).
    # 질문을 본문에 재인쇄하지 않는다 — 답변 본문은 검증 대상 클레임 공간이라,
    # 사용자 전제(질문 속 수치)를 그대로 옮기면 그것도 클레임이 된다.
    lines = [
        "※ 오프라인 모드: LLM 없이 검색 근거를 발췌해 제시합니다 "
        "(ANTHROPIC_API_KEY 설정 시 근거를 종합한 자연어 답변으로 자동 전환).",
        "",
        "가장 관련 높은 규정 근거:",
    ]
    for i, r in enumerate(results, 1):
        body = r["text"].split("]\n", 1)[-1].strip()
        snippet = body[:260] + ("…" if len(body) > 260 else "")
        lines.append(f"\n[근거 {i}] {r['title']} > {r['section']} (출처: {r['source']})\n{snippet}")
    return "\n".join(lines)


def _summarize(data: Any) -> str:
    if isinstance(data, dict):
        if "reportable" in data:
            n_missing = len(data.get("missing", []))
            return "초안 완성(요건 충족)" if data["reportable"] else f"초안 생성(보완 {n_missing}건 필요)"
        if "is_serious" in data:
            return "중대 → " + data.get("route", "") if data["is_serious"] else "비중대 → PSUR"
        if "results" in data:
            return f"{len(data['results'])}건 검색"
        if "deadlines" in data:
            return f"{data.get('count', 0)}건 마감"
        if "items" in data:
            return f"체크리스트 {len(data['items'])}항목"
        if "documents" in data:
            return f"문서 {data.get('count', 0)}건"
    return str(data)[:80]


def _stringify(data: Any) -> str:
    import json

    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(data)
