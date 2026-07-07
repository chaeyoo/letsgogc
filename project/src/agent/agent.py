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
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "section": r.get("section"),
                    "score": r.get("score"),
                }
            )
    return cites


class RaAgent:
    """RA 어시스턴트. chat() 하나로 두 모드를 투명하게 처리."""

    async def chat(self, message: str, history: list[dict] | None = None) -> AgentResult:
        if config.LLM_AVAILABLE:
            return await self._chat_llm(message, history or [])
        return await self._chat_offline(message)

    # ---- LLM 모드: 실제 tool-use 루프 ----
    async def _chat_llm(self, message: str, history: list[dict]) -> AgentResult:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        tool_calls: list[ToolCall] = []
        raw_search_results: list[dict] = []

        async with Client(mcp) as mcp_client:
            tools = _to_anthropic_tools(await mcp_client.list_tools())
            messages: list[dict] = [*history, {"role": "user", "content": message}]

            # 에이전트 루프 (최대 6스텝 — 무한루프 방지)
            for _ in range(6):
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
                    call = await mcp_client.call_tool(block.name, dict(block.input))
                    data = call.data
                    if block.name == "search_regulations" and isinstance(data, dict):
                        raw_search_results.append(data)
                    tool_calls.append(
                        ToolCall(
                            name=block.name,
                            args=dict(block.input),
                            result_summary=_summarize(data),
                        )
                    )
                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _stringify(data),
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

    # ---- 오프라인 모드: 규칙 라우팅 + grounded 추출 답변 ----
    async def _chat_offline(self, message: str) -> AgentResult:
        tool_calls: list[ToolCall] = []
        async with Client(mcp) as mcp_client:
            intent = _route_intent(message)

            if intent == "deadlines":
                call = await mcp_client.call_tool("get_ra_deadlines", {"within_days": 30})
                data = call.data
                tool_calls.append(ToolCall("get_ra_deadlines", {"within_days": 30}, _summarize(data)))
                answer = _format_deadlines(data)
                return AgentResult(answer=answer, mode="offline", tool_calls=tool_calls)

            if intent == "checklist":
                category = _guess_category(message)
                call = await mcp_client.call_tool("get_submission_checklist", {"category": category})
                data = call.data
                tool_calls.append(ToolCall("get_submission_checklist", {"category": category}, _summarize(data)))
                answer = _format_checklist(data)
                return AgentResult(answer=answer, mode="offline", tool_calls=tool_calls)

            # 기본: 규제문서 검색 후 근거 기반 답변
            call = await mcp_client.call_tool("search_regulations", {"query": message, "top_n": 3})
            data = call.data
            tool_calls.append(ToolCall("search_regulations", {"query": message, "top_n": 3}, _summarize(data)))
            answer = _format_search_answer(message, data)
            return AgentResult(
                answer=answer,
                mode="offline",
                tool_calls=tool_calls,
                citations=_collect_citations(tool_calls, [data]),
            )


# ---------------------------------------------------------------------------
# 보조 함수 (오프라인 포매팅 · 라우팅)
# ---------------------------------------------------------------------------
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
