"""RA·PV 어시스턴트의 PydanticAI 백엔드 (AGENT_BACKEND=pydantic_ai 선택 시).

기본 백엔드(agent.py `_chat_llm`)가 anthropic SDK 로 직접 구현한 tool-use
루프를, 같은 계약 그대로 PydanticAI 프레임워크 위에서 실행하는 병렬 구현이다.
두 백엔드는 항상 같은 것을 보장해야 한다:

  - 도구는 MCP 를 통해 호출한다 — `MCPToolset` 이 인메모리 FastMCP 서버
    (src/mcp_server/server.py 의 `mcp`)에 직접 붙으므로, 도구 이름·설명·스키마의
    출처가 direct 백엔드와 동일하다(스키마 이중 유지 없음).
  - 도구 호출 1건마다의 북키핑(신뢰 소스·출처·grounded·이력 허용 집합)은
    `process_tool_call` 훅에서 direct 루프 본문과 같은 헬퍼로 수행한다.
  - 모든 답변은 direct 백엔드와 동일 인자로 `_finalize()` 검증 게이트를 지난다.

의도된 차이(프레임워크의 계산 방식): direct 루프는 '왕복 6회'를 한 카운터로
세지만, 여기서는 모델 요청 상한(UsageLimits.request_limit=6)과 도구 재시도
상한(Agent retries=2)이 분리되어 있다 — 상한이 존재한다는 계약은 같고,
초과 시 사용자 문구도 같다. 도구 실패의 자가 정정 되먹임은 MCPToolset 이
ToolError→ModelRetry 변환으로 대신한다(direct 루프의 is_error=True 되먹임과 동치).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from .. import config
from ..mcp_server.server import mcp
from ..observability import Trace, flow, timed
from .agent import (
    SYSTEM_PROMPT,
    AgentResult,
    ToolCall,
    _collect_citations,
    _finalize,
    _has_evidence,
    _is_contract_error,
    _question_context,
    _split_user_facts,
    _stringify,
    _summarize,
)

MAX_LLM_REQUESTS = 6  # direct 루프의 6-step 상한과 같은 역할(모델 요청 횟수 상한)


@dataclass
class PaDeps:
    """per-run 가변 상태 — direct 루프의 '증거 바구니' 지역변수들과 1:1 대응.

    direct 루프는 함수 지역변수로 증거를 쌓지만, PydanticAI 에서는 루프가
    프레임워크 안에 있으므로 RunContext.deps 로 같은 바구니를 흘려보낸다.
    """

    trace: Trace
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_search_results: list[dict] = field(default_factory=list)
    trusted_texts: list[str] = field(default_factory=list)
    user_facts: list[str] = field(default_factory=list)
    history_doc_ids: set[str] = field(default_factory=set)
    grounded_evidence: bool = False


def _normalize(data: Any) -> Any:
    """call_tool 반환을 fastmcp Client(.data) 와 동형인 dict/list 로 정규화.

    MCPToolset 은 구조화 출력(dict)을 그대로 돌려주므로 보통 무변환이다 —
    구조화 출력이 없는 예외 형태(콘텐츠 블록 리스트)만 방어적으로 편다.
    """
    if isinstance(data, (list, tuple)) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return data


async def _process_tool_call(
    ctx: RunContext[PaDeps], call_tool: CallToolFunc, name: str, tool_args: dict[str, Any]
) -> ToolResult:
    """도구 호출 1건의 관문 — direct 루프 본문의 북키핑을 같은 헬퍼로 수행한다."""
    deps = ctx.deps
    flow(
        "_process_tool_call()",
        "PydanticAI 의 도구 실행 요청 — 실행은 MCPToolset 이 인메모리 FastMCP 로 대신한다",
        tool=name, args=dict(tool_args),
        next="call_tool() — 성공 출력은 신뢰 소스로 축적, 실패는 ModelRetry 로 모델에 되먹임",
    )
    try:
        with timed(deps.trace, f"tool.{name}", "tool", {"args": dict(tool_args)}):
            data = _normalize(await call_tool(name, tool_args))
    except Exception as e:
        # 실패도 UI 표시용으로 기록하고 그대로 재-raise 한다 — MCPToolset 의
        # tool_error_behavior='retry'(기본)가 에러 문구를 ModelRetry 로 모델에
        # 되먹여 자가 정정시킨다(direct 루프의 is_error=True 되먹임과 동치).
        deps.tool_calls.append(
            ToolCall(name, dict(tool_args), f"오류: 도구 '{name}' 실행 실패: {type(e).__name__}: {e}")
        )
        raise
    # ── direct 루프(agent.py _chat_llm)와 동일한 북키핑 ──
    if not _is_contract_error(data):
        # 성공 출력만 신뢰 소스로. 입력 에코(query·as_of)는 버리고 케이스
        # 서술(case)은 별도 계층(user_facts)으로 — 근거 승격 구멍 차단.
        stripped, facts = _split_user_facts(data)
        deps.trusted_texts.append(_stringify(stripped))
        deps.user_facts.extend(facts)
        if _has_evidence(stripped):  # 결과 0건짜리 '성공'은 근거로 안 친다
            deps.grounded_evidence = True
    if isinstance(data, dict):
        if name == "search_regulations" and "results" in data:
            deps.raw_search_results.append(data)
            if tool_args.get("include_superseded") or tool_args.get("as_of"):
                # 이력 검색이 '실제로 반환한' 문서만 폐지본 경고 면제(문서 단위)
                deps.history_doc_ids |= {
                    r["doc_id"] for r in data["results"] if r.get("doc_id")
                }
        elif name in ("assess_adverse_event", "draft_ae_report") and data.get("basis"):
            deps.raw_search_results.append(data["basis"])  # 트리아지/초안의 근거 규정도 출처로
    deps.tool_calls.append(ToolCall(name, dict(tool_args), _summarize(data)))
    return data


def _build_model():
    """실행 시점에만 모델을 만든다 — 키 없이도 모듈 import 가 가능해야 하고,
    테스트는 이 봉합선을 FunctionModel/TestModel 로 바꿔 끼운다."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicModel(
        config.LLM_MODEL, provider=AnthropicProvider(api_key=config.ANTHROPIC_API_KEY)
    )


def _build_agent() -> Agent[PaDeps, str]:
    """호출마다 toolset+Agent 를 새로 만든다 — direct 백엔드가 호출마다
    `Client(mcp)` 를 여는 것과 동형(이벤트 루프 간 세션 공유 없음)."""
    toolset = MCPToolset(mcp, process_tool_call=_process_tool_call)
    return Agent(
        deps_type=PaDeps,
        instructions=SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )


def _to_model_messages(history: list[dict]) -> list[ModelRequest | ModelResponse]:
    """anthropic 스타일 role/content dict 이력 → PydanticAI ModelMessage.

    텍스트만 이월한다(direct 백엔드도 이력의 텍스트 표기만 실질 사용) —
    도구 블록 등 비텍스트 표기는 이전 턴의 내부 산출물이라 재전송하지 않는다.
    """
    out: list[ModelRequest | ModelResponse] = []
    for turn in history:
        content = turn.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        else:
            continue
        if not text:
            continue
        if turn.get("role") == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        else:
            out.append(ModelResponse(parts=[TextPart(content=text)]))
    return out


async def chat_llm_pydantic(message: str, history: list[dict], trace: Trace) -> AgentResult:
    """PydanticAI 백엔드의 LLM 모드 진입점 — 반환 계약은 `_chat_llm` 과 동일.

    PII 마스킹은 호출부(RaAgent.chat)에서 이미 끝났고, 여기서는 에이전트 실행과
    북키핑·검증 게이트만 담당한다. mode 는 "llm" 그대로 — API/UI 는 백엔드를
    구분하지 않는다(배너·trace 로만 노출).
    """
    deps = PaDeps(trace=trace)
    flow(
        "chat_llm_pydantic()",
        "PydanticAI 백엔드 시작 — MCPToolset(인메모리 FastMCP)+UsageLimits 로 tool-use 루프를 프레임워크에 위임",
        model=config.LLM_MODEL, request_limit=MAX_LLM_REQUESTS,
        next="agent.run() — 도구 호출마다 _process_tool_call 훅이 신뢰 소스를 축적한다",
    )
    grounded = False
    try:
        agent = _build_agent()
        with timed(trace, "llm.run", "llm", {"model": config.LLM_MODEL, "backend": "pydantic_ai"}):
            result = await agent.run(
                message,
                deps=deps,
                model=_build_model(),
                message_history=_to_model_messages(history),
                usage_limits=UsageLimits(request_limit=MAX_LLM_REQUESTS),
            )
        answer = str(result.output).strip()
        grounded = deps.grounded_evidence
        flow(
            "chat_llm_pydantic()",
            "PydanticAI 실행 완료 — 답변 확정, 사후 검증 게이트로",
            answer_len=len(answer), tool_calls=len(deps.tool_calls), grounded=grounded,
            next="_finalize() — direct 백엔드와 동일 인자로 답변 속 수치·인용을 신뢰 소스와 대조",
        )
    except UsageLimitExceeded:
        # 요청 상한 초과 — direct 루프가 6바퀴를 다 쓴 경우와 같은 문구로 마감.
        # deps 에 쌓인 증거(출처·신뢰 소스)는 그대로 살려 반환한다.
        answer = "(도구 호출이 반복되어 응답을 확정하지 못했습니다. 질문을 좁혀 다시 시도해 주세요.)"
        flow(
            "chat_llm_pydantic()",
            "요청 상한 초과(UsageLimitExceeded) — 확정 실패 안내로 마감(direct 6-step 상한과 같은 계약)",
            tool_calls=len(deps.tool_calls),
            next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
        )
    except Exception as e:  # noqa: BLE001 - 외부 API 실패는 명시적 안내로 흡수
        # direct 백엔드와 같은 규율: 예외 '타입명'만 싣고 메시지 원문은 싣지 않는다
        # (외부 라이브러리 에러 문구에는 요청 내용이 에코될 수 있다).
        answer = (
            f"(LLM API 호출에 실패했습니다: {type(e).__name__}. "
            "ANTHROPIC_API_KEY 가 유효한지, 네트워크와 LLM_MODEL 설정을 "
            "확인해 주세요. 키를 비우면 오프라인 모드로 동작합니다.)"
        )
        flow(
            "chat_llm_pydantic()",
            "LLM API 호출 실패 — 크래시(HTTP 500) 대신 확인 안내문으로 응답(원인 타입만 노출)",
            error=type(e).__name__,
            next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
        )
    return _finalize(
        AgentResult(
            answer=answer,
            mode="llm",
            tool_calls=deps.tool_calls,
            citations=_collect_citations(deps.tool_calls, deps.raw_search_results),
            grounded=grounded,
        ),
        deps.trusted_texts,
        question=_question_context(message, history),
        allowed_superseded_ids=deps.history_doc_ids,
        user_facts=deps.user_facts,
    )
