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
from ..rag.textutil import tokenize

# 근거 부족(abstention) 판단 임계값 — 제약 규제 도메인의 환각 안전장치.
# 두 신호를 함께 본다: (1) 최상위 근거의 검색 관련도 점수, (2) 질의-근거 토큰 커버리지.
# 둘 다 문턱 아래일 때만("확실히 근거 없음") 답을 지어내지 않고 회피한다.
# AND 조건을 쓰는 이유: 콜로퀄한 범위내 질문은 점수는 낮아도 커버리지가 남아있어
# 과도한 회피(over-abstention)를 피할 수 있다. 문턱값은 범위내/범위밖 분포로 보정.
SCORE_FLOOR = 0.22       # 최상위 근거 관련도 하한
COVERAGE_FLOOR = 0.26    # 질의 토큰 커버리지 하한

SYSTEM_PROMPT = (
    "당신은 제약회사 RA(인허가·규제업무) 담당자를 돕는 어시스턴트다. "
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
                    "score": r.get("score"),
                }
            )
    return cites


class RaAgent:
    """RA 어시스턴트. chat() 하나로 두 모드를 투명하게 처리."""

    async def chat(self, message: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        trace = Trace()
        with timed(trace, "chat", "agent", {"mode": "llm" if config.LLM_AVAILABLE else "offline"}):
            if config.LLM_AVAILABLE:
                result = await self._chat_llm(message, history, trace)
            else:
                result = await self._chat_offline(message, history, trace)
        result.trace = trace.to_list()
        result.latency_ms = trace.total_ms
        return result

    # ---- LLM 모드: 실제 tool-use 루프 ----
    async def _chat_llm(self, message: str, history: list[dict], trace: Trace) -> AgentResult:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        tool_calls: list[ToolCall] = []
        raw_search_results: list[dict] = []

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
                    return AgentResult(
                        answer=text,
                        mode="llm",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, raw_search_results),
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
                    if not is_error and block.name == "search_regulations" and isinstance(data, dict):
                        raw_search_results.append(data)
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
            return AgentResult(
                answer="(도구 호출이 반복되어 응답을 확정하지 못했습니다. 질문을 좁혀 다시 시도해 주세요.)",
                mode="llm",
                tool_calls=tool_calls,
                citations=_collect_citations(tool_calls, raw_search_results),
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

            if intent == "deadlines":
                args = {"within_days": 30}
                with timed(trace, "tool.get_ra_deadlines", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_ra_deadlines", args)).data
                tool_calls.append(ToolCall("get_ra_deadlines", args, _summarize(data)))
                answer = _format_deadlines(data)
                return AgentResult(answer=answer, mode="offline", tool_calls=tool_calls)

            if intent == "checklist":
                category = _guess_category(resolved)
                args = {"category": category}
                with timed(trace, "tool.get_submission_checklist", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_submission_checklist", args)).data
                tool_calls.append(ToolCall("get_submission_checklist", args, _summarize(data)))
                answer = _format_checklist(data)
                return AgentResult(answer=answer, mode="offline", tool_calls=tool_calls)

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
                return AgentResult(
                    answer=(
                        "제공된 사내 규제문서에서 이 질문에 답할 근거를 찾지 못했습니다. "
                        "질문을 규제업무(허가·변경·GMP·라벨링·약물감시·임상) 범위로 좁혀 다시 시도해 주세요."
                    ),
                    mode="offline",
                    tool_calls=tool_calls,
                    citations=[],
                    grounded=False,
                )
            answer = _format_search_answer(resolved, data)
            return AgentResult(
                answer=answer,
                mode="offline",
                tool_calls=tool_calls,
                citations=_collect_citations(tool_calls, [data]),
                grounded=True,
            )


# ---------------------------------------------------------------------------
# 보조 함수 (오프라인 포매팅 · 라우팅)
# ---------------------------------------------------------------------------
_FOLLOWUP_MARKERS = ("그건", "그거", "그럼", "그게", "이건", "위", "방금", "아까", "그 경우", "그 때")


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
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    ctx = "\n".join(r.get("text", "") for r in results)
    ctx_tokens = set(tokenize(ctx))
    return len(q_tokens & ctx_tokens) / len(q_tokens)


def _route_intent(message: str) -> str:
    m = message.lower()
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


def _format_search_answer(query: str, data: dict) -> str:
    results = data.get("results", [])
    if not results:
        return "관련 규정을 찾지 못했습니다. 질문을 바꿔 다시 시도해 주세요."
    # grounded 추출: 최상위 근거를 중심으로 요약 제시(오프라인이라 생성 대신 발췌)
    lines = [
        "※ 오프라인 모드: LLM 없이 검색 근거를 발췌해 제시합니다 "
        "(ANTHROPIC_API_KEY 설정 시 근거를 종합한 자연어 답변으로 자동 전환).",
        "",
        f"질문: {query}",
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
