"""RA·PV 어시스턴트 에이전트 (Agentic Workflow + Function Calling).

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

import re
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from .. import config
from ..mcp_server.server import mcp
from ..observability import Trace, record_verification, timed
from ..pv.coding import symptom_keywords
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
    # 도구 근거(검색 결과·결정론 도구 출력)로 뒷받침된 답변인지. 오프라인 모드는
    # abstention 이면 False. LLM 모드는 문턱 판정이 없으므로(회피 판정은 오프라인
    # 검색 경로 전용) '성공한 도구 출력이 신뢰 소스로 확보되었는가'로 계산한다 —
    # 모델이 도구를 한 번도 부르지 않고 답하면 False(근거 보증이 아니라는 신호).
    grounded: bool = True
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


_ECHO_KEYS = ("query", "as_of")  # 도구 출력에 에코되는 '사용자 입력' 필드


def _strip_query_echo(data: Any) -> Any:
    """도구 출력에서 사용자 입력 에코 필드(query·as_of)를 재귀 제거한 사본을 만든다.

    신뢰 소스는 '근거 문단 ∪ 도구 출력'인데, search 계열 도구는 받은 입력을
    출력에 에코한다. 이를 그대로 신뢰하면 **사용자 입력 속 수치·날짜가 신뢰
    소스로 승격**되어, 사용자가 틀린 값을 전제로 물었을 때 모델이 맞장구쳐도
    검증을 통과하는 구멍이 생긴다. query 만 제거하고 as_of(기준일 에코)를
    남기는 것은 같은 구멍을 반쪽만 막은 비대칭이었다 — 사용자가 지정한
    기준일이 답변 날짜 클레임의 '근거'가 되어 버린다. 케이스 서술(case)은
    남긴다 — 그것은 검증할 규정 클레임이 아니라 사용자가 제공한 '사실'이다.
    """
    if isinstance(data, dict):
        return {k: _strip_query_echo(v) for k, v in data.items() if k not in _ECHO_KEYS}
    if isinstance(data, list):
        return [_strip_query_echo(v) for v in data]
    return data


_USER_FACT_KEYS = ("case",)  # 도구 출력 중 '사용자 제공 사실' 에코 필드


def _split_user_facts(data: Any) -> tuple[Any, list[str]]:
    """도구 출력에서 입력 에코를 제거하고 사용자 사실 필드(case)를 분리한다.

    반환: (에코 제거된 사본, 사용자 사실 텍스트 목록). query·as_of 는 종전대로
    버리고(전제의 승격 차단), case 는 버리는 대신 **별도 계층**으로 분리한다 —
    검증기가 지지 근거로는 인정하되 from_case 라벨을 붙일 수 있도록.
    '케이스는 사실이므로 신뢰 소스'라는 종전 규칙에는 "사실이라는 성질이 그
    값을 규정 클레임의 근거로 승격시키지는 않는다"는 반례가 있었다(케이스의
    "30일간 복용"이 답변의 "보고 기한 30일"을 지지하는 조용한 통과).
    한계의 명시: draft_ae_report 의 draft_markdown 처럼 케이스 서술이 다른
    필드 '안에' 재조립된 경우는 이 분리가 잡지 못한다(필드 단위 분리의 경계).
    """
    facts: list[str] = []

    def _strip(d: Any) -> Any:
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                if k in _ECHO_KEYS:
                    continue
                if k in _USER_FACT_KEYS and isinstance(v, str):
                    facts.append(v)
                    continue
                out[k] = _strip(v)
            return out
        if isinstance(d, list):
            return [_strip(v) for v in d]
        return d

    return _strip(data), facts


def _is_contract_error(data: Any) -> bool:
    """도구의 명시적 에러 계약({"error", ...}) 응답인지 판정한다.

    에러 계약은 에이전트의 자가 정정용 되먹임이지 **근거가 아니다** — 에러
    문구에는 사용자 입력이 에코된다(예: "as_of '2024/06/01' 가 … 형식이 아님").
    이를 신뢰 소스에 넣으면 잘못된 입력 값이 검증의 '근거'로 승격된다.
    query·as_of 키만 걷어내는 필드 단위 접근으로는 못 막는 경로라(값이 error
    문자열 안에 들어 있다), 에러 계약 응답은 통째로 신뢰 소스에서 제외한다.
    """
    return isinstance(data, dict) and "error" in data


def _captured_evidence(data: Any) -> bool:
    """도구 출력이 실제 '근거'를 담았는가 — grounded 신호 계산용(v10).

    검색이 0건이면 그 출력이 신뢰 소스로 직렬화돼도('{"results": []}') 근거는
    아니다: grounded=bool(trusted_texts) 는 이 빈 봉투를 참으로 세어, 도구를
    부르되 아무것도 못 찾은(출처 0건) 답변이 근거 배지를 달고 나가게 했다 —
    v7 이 '도구 미호출' 경로에서 막은 바로 그 '출처 0건 grounded' 오탐이
    '도구 호출·빈 결과' 경로에 잔존한 형태(안전장치가 한쪽 경로에만 배선된
    비대칭). 오프라인 모드는 같은 무근거 질의를 abstention 으로 grounded=False
    로 라벨링하는데 LLM 모드만 True 였다. 검색류(results 키)는 결과가 비지
    않을 때만 근거로 세고, results 키가 없는 결정론 도구(트리아지·마감·초안)의
    출력은 그 자체가 근거다.
    """
    if isinstance(data, dict) and "results" in data:
        return bool(data["results"])
    return True


def _finalize(
    result: AgentResult,
    trusted_texts: list[str],
    allow_superseded: bool = False,
    question: str = "",
    allowed_superseded_ids: set[str] | None = None,
    user_facts: list[str] | None = None,
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
    v = verify_answer(
        result.answer,
        trusted_texts,
        result.citations,
        allow_superseded,
        question=question,
        allowed_superseded_ids=allowed_superseded_ids,
        user_fact_texts=user_facts,
    )
    if not v.ok:
        result.answer = f"{result.answer}\n\n{warning_text(v)}"
    result.verification = v.summary()
    # 운영 계기판 집계 + 감사 로그 — 경고율(alert fatigue 조기 신호)을 /health 로 상시 노출
    record_verification(result.verification)
    return result


class RaAgent:
    """RA·PV 어시스턴트. chat() 하나로 두 모드를 투명하게 처리."""

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
        evidence_captured = False       # 실제 근거(비어 있지 않은 도구 출력)를 확보했는가 — grounded 신호
        # 이력(as_of·include_superseded) 검색이 '실제로 반환한' 문서 집합 —
        # 이 문서들의 폐지본 인용만 경고에서 면제한다. 전역 bool 로 켜면 같은
        # 턴의 현행 검색에 (상류 결함으로) 섞여 든 폐지본 인용까지 경고가 꺼져,
        # 게이트의 버전 축이 이력 턴 동안 통째로 무장해제된다 — 안전장치를
        # 끄는 스위치의 면적은 근거가 성립하는 문서 단위로 좁힌다.
        history_doc_ids: set[str] = set()
        user_facts: list[str] = []      # 도구 출력 속 사용자 사실 에코(case) — 2계층 신뢰 소스

        async with Client(mcp) as mcp_client:
            tools = _to_anthropic_tools(await mcp_client.list_tools())
            messages: list[dict] = [*history, {"role": "user", "content": message}]

            # 에이전트 루프 (최대 6스텝 — 무한루프 방지)
            for step in range(6):
                try:
                    with timed(trace, f"llm.step{step}", "llm", {"model": config.LLM_MODEL}):
                        resp = await client.messages.create(
                            model=config.LLM_MODEL,
                            max_tokens=1024,
                            system=SYSTEM_PROMPT,
                            tools=tools,
                            messages=messages,
                        )
                except Exception as e:  # noqa: BLE001 - 외부 API 실패는 명시적 안내로 흡수
                    # 잘못된 키(401)·네트워크 오류·존재하지 않는 모델명은 가장 흔한
                    # 온보딩 실패 경로다 — 예외를 그대로 전파하면 HTTP 500 이라는
                    # 불투명한 실패가 된다(조용하진 않지만 원인을 말해주지 않는다).
                    # 명시적 안내로 답한다. 예외 '메시지'는 답변에 싣지 않는다 —
                    # 외부 라이브러리 에러 문구에는 요청 정보가 에코될 수 있다
                    # (에러 계약 응답을 신뢰 소스에서 빼는 것과 같은 계열의 규율).
                    return _finalize(
                        AgentResult(
                            answer=(
                                f"(LLM API 호출에 실패했습니다: {type(e).__name__}. "
                                "ANTHROPIC_API_KEY 가 유효한지, 네트워크와 LLM_MODEL 설정을 "
                                "확인해 주세요. 키를 비우면 오프라인 모드로 동작합니다.)"
                            ),
                            mode="llm",
                            tool_calls=tool_calls,
                            citations=_collect_citations(tool_calls, raw_search_results),
                            grounded=False,  # 실패 안내문 — 근거 보증이 아니다
                        ),
                        trusted_texts,
                        question=_question_context(message, history),
                        allowed_superseded_ids=history_doc_ids,
                        user_facts=user_facts,
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
                            # LLM 모드에는 문턱 기반 abstention 이 없다(오프라인 검색
                            # 경로 전용) — grounded 는 '성공한 도구 출력이 실제 근거를
                            # 확보했는가'다. 모델이 도구 없이 답하거나(도구 미호출)
                            # 검색이 0건이면(빈 결과) False: 이전에는 dataclass 기본값
                            # True(v7 전) 와 bool(trusted_texts)(v7~v9) 가 각각
                            # '도구 미호출'·'빈 결과' 답변에 근거 배지를 달았다 —
                            # _captured_evidence 가 빈 검색 봉투를 근거에서 뺀다(v10).
                            grounded=evidence_captured,
                        ),
                        trusted_texts,
                        question=_question_context(message, history),
                        allowed_superseded_ids=history_doc_ids,
                        user_facts=user_facts,
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
                    if not is_error and not _is_contract_error(data):
                        # 도구 출력 = 검증 신뢰 소스. 단 입력 에코(query·as_of)와
                        # 에러 계약 응답은 제외 — 사용자 전제가 신뢰 소스로
                        # 승격되는 구멍을 막는다(에러 문구 속 에코 포함).
                        # 케이스 에코(case)는 별도 계층으로 분리 — 지지 근거로는
                        # 인정하되 from_case 라벨이 붙는다(사용자 서술의 승격 가시화).
                        stripped, facts = _split_user_facts(data)
                        trusted_texts.append(_stringify(stripped))
                        user_facts.extend(facts)
                        if _captured_evidence(data):
                            evidence_captured = True
                    if not is_error and isinstance(data, dict):
                        if block.name == "search_regulations" and "results" in data:
                            # 성공한 검색만 집계한다 — 형식 오류로 실패한 as_of
                            # 호출이 허용 집합을 채우면, 시점 검색이 한 번도
                            # 성공하지 않았는데 폐지본 인용 경고만 꺼진다.
                            raw_search_results.append(data)
                            if block.input.get("include_superseded") or block.input.get("as_of"):
                                # 이력 조회 의도 — 단, 이 검색이 '실제로 반환한'
                                # 문서의 폐지본 인용만 결함이 아니다(문서 단위 허용).
                                history_doc_ids |= {
                                    r["doc_id"] for r in data["results"] if r.get("doc_id")
                                }
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
                    grounded=False,  # 확정 실패 안내문 — 근거 보증이 아니다
                ),
                trusted_texts,
                question=_question_context(message, history),
                allowed_superseded_ids=history_doc_ids,
                user_facts=user_facts,
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
                if aware := _extract_awareness_date(resolved):
                    args["awareness_date"] = aware
                with timed(trace, "tool.draft_ae_report", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("draft_ae_report", args)).data
                tool_calls.append(ToolCall("draft_ae_report", args, _summarize(data)))
                stripped, facts = _split_user_facts(data)
                return _finalize(
                    AgentResult(
                        answer=_format_report(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(stripped)],
                    question=resolved,
                    user_facts=facts,
                )

            if intent == "ae_triage":
                # 구체적 케이스 서술 → PV 트리아지 도구(중대성 판정+기한 계산)
                args = {"case_description": resolved}
                if aware := _extract_awareness_date(resolved):
                    args["awareness_date"] = aware
                with timed(trace, "tool.assess_adverse_event", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("assess_adverse_event", args)).data
                tool_calls.append(ToolCall("assess_adverse_event", args, _summarize(data)))
                stripped, facts = _split_user_facts(data)
                return _finalize(
                    AgentResult(
                        answer=_format_triage(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(stripped)],
                    question=resolved,
                    user_facts=facts,
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

            checklist_note = ""
            if intent == "checklist":
                category = _guess_category(resolved)
                args = {"category": category}
                with timed(trace, "tool.get_submission_checklist", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_submission_checklist", args)).data
                tool_calls.append(ToolCall("get_submission_checklist", args, _summarize(data)))
                if "error" not in data:
                    answer = _format_checklist(data)
                    return _finalize(
                        AgentResult(answer=answer, mode="offline", tool_calls=tool_calls),
                        [_stringify(data)],
                        question=resolved,
                    )
                # 카테고리 매칭 실패 — 엉뚱한 체크리스트를 자신 있게 내는 대신,
                # 지원 목록을 안내하고 규정 검색으로 폴백한다. LLM 모드가
                # error+available 을 되먹여 받아 자가 정정하는 것과 같은 계약을
                # 오프라인 라우터도 따르는 것(계약을 라우터가 우회하지 않는다).
                # 문구는 검색 결과를 약속하지 않는다 — 폴백 검색이 회피(abstention)로
                # 끝나면 "아래는 검색 결과입니다 / 근거를 찾지 못했습니다"가 한 답변에
                # 공존하는 자기모순이 된다(결과를 본 뒤에만 말할 수 있는 것을 미리
                # 말하지 않는다).
                available = ", ".join(data.get("available", []))
                checklist_note = (
                    f"※ 요청과 일치하는 체크리스트 유형을 찾지 못했습니다 (지원: {available}).\n\n"
                )

            # 기본: 규제문서 검색 후 근거 기반 답변.
            # 이력 조회 의도(예전/구판/개정 이력)를 감지하면 폐지본을 포함해 검색한다
            # — "이력을 요청하면 구판도 노출"이라는 사용 계약이 LLM 모드(모델이
            # include_superseded 를 스스로 지정)에서만 참이고 오프라인 라우터에는
            # 그 경로 자체가 없던 불일치의 해소.
            history_intent = any(k in resolved for k in _HISTORY_MARKERS)
            args = {"query": resolved, "top_n": 3}
            if history_intent:
                args["include_superseded"] = True
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
                        answer=checklist_note
                        + (
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
            answer = checklist_note + _format_search_answer(data)
            return _finalize(
                AgentResult(
                    answer=answer,
                    mode="offline",
                    tool_calls=tool_calls,
                    citations=_collect_citations(tool_calls, [data]),
                    grounded=True,
                ),
                [_stringify(_strip_query_echo(data))],
                allow_superseded=history_intent,  # 이력 조회에서 폐지본 인용은 결함이 아니라 목적
                question=resolved,
            )


# ---------------------------------------------------------------------------
# 보조 함수 (오프라인 포매팅 · 라우팅)
# ---------------------------------------------------------------------------
_FOLLOWUP_MARKERS = ("그건", "그거", "그럼", "그게", "이건", "위", "방금", "아까", "그 경우", "그 때")

# 이력(폐지본 포함) 조회 의도 — **명시적 이력 요청 구문만** 좁게 잡는다.
# 단독 "이력"은 현행 질문의 일상 어휘("접수·처리 이력 관리 요건")와, 단독
# "폐지"는 현행 상태 질문("이 규정 폐지됐나요?")과 겹친다 — 이력 의도 오탐은
# 폐지본을 검색에 섞을 뿐 아니라 allow_superseded 로 **버전 경고까지 끈다**.
# 안전장치를 끄는 스위치의 오탐은 일반 라우팅 오탐보다 비싸므로, 확신 없는
# 어휘는 넣지 않는다(코딩 사전과 같은 철학).
_HISTORY_MARKERS = ("예전", "구판", "개정 이력", "개정 전", "이전 버전", "과거 규정", "당시 규정")


def _redact_history(history: list[dict]) -> list[dict]:
    """대화 이력의 콘텐츠에서 PII를 마스킹한다(외부 API로 나가기 전 방어).

    문자열 콘텐츠뿐 아니라 **anthropic 블록 리스트 표기**(content=[{"type":
    "text", "text": …}])도 마스킹한다 — 블록 표기는 이 파일의 _user_texts 가
    명시적으로 처리하는 '예상된 입력 형태'인데, 마스킹만 문자열 표기에서
    성립하는 것은 표기의 우회였다(정규식 \\b 경계가 한글 직결 표기에서만
    깨지던 v6 발견 C 와 동형 — 이력 리스트형 콘텐츠 속 개인정보가 마스킹
    없이 외부 LLM API 로 나가는 경로, v7 발견). 블록의 텍스트 표기는 두 키다
    — text(텍스트 블록)와 **content(tool_result 블록의 문자열 본문)**: raw
    transcript 를 이력으로 되돌려 보내는 사용에서 tool_result 표기만 빠지면
    같은 우회가 한 키 이름 차이로 반복된다(v7 리뷰 잔여분). 그 외 타입
    (비문자열 스칼라 등)은 텍스트가 아니므로 그대로 둔다.
    """

    def _redact_block(b: Any) -> Any:
        if not isinstance(b, dict):
            return b
        out_b = dict(b)
        if isinstance(out_b.get("text"), str):
            out_b["text"] = redact(out_b["text"]).text
        if isinstance(out_b.get("content"), str):  # tool_result 블록의 문자열 본문
            out_b["content"] = redact(out_b["content"]).text
        elif isinstance(out_b.get("content"), list):  # 중첩 블록 리스트
            out_b["content"] = [_redact_block(x) for x in out_b["content"]]
        return out_b

    out: list[dict] = []
    for turn in history:
        content = turn.get("content")
        if isinstance(content, str):
            out.append({**turn, "content": redact(content).text})
        elif isinstance(content, list):
            out.append({**turn, "content": [_redact_block(b) for b in content]})
        else:
            out.append(turn)
    return out


def _user_texts(history: list[dict]) -> list[str]:
    """대화 이력에서 사용자 발화 텍스트만 순서대로 추출한다."""
    out: list[str] = []
    for turn in history:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):  # anthropic 블록 형식 대비
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                out.append(" ".join(parts))
    return out


def _last_user_text(history: list[dict]) -> str:
    """대화 이력에서 마지막 사용자 발화 텍스트를 추출(멀티턴 맥락 복원용)."""
    texts = _user_texts(history)
    return texts[-1] if texts else ""


def _question_context(message: str, history: list[dict]) -> str:
    """검증의 from_question 라벨링용 질문 맥락 — 직전 사용자 발화까지만 포함한다.

    LLM 모드 답변은 이전 턴의 전제("보고 기한이 30일 맞지?" → 다음 턴 "그럼
    그 30일은…")를 이어받아 재서술할 수 있는데, 현재 턴 질문만 대조하면 그 값이
    '환각' 경고로 라벨링된다 — 실제로 맞는 경고는 '전제 확인 필요'다(둘 다
    경고는 붙는다 — 라벨은 경고의 종류를 조정할 뿐, 면제하지 않는다).

    범위를 두 방향으로 제한한다:
    - **사용자 발화만**: 어시스턴트의 이전 답변까지 넣으면 한 번 새어 나간
      미확인 수치가 다음 턴부터 '전제'로 완화 라벨링되는 자기 강화 루프가 생긴다.
    - **직전 턴까지만**(전체 이력 아님): 대화 전체를 누적하면 10턴 전 무관한
      맥락의 수치("점유율 90%")가 이후 모든 환각을 '전제'로 완화 라벨링한다 —
      완화 라벨의 면적은 실제 전제 이월이 일어나는 창(직전 턴, 오프라인
      후속질문 병합과 같은 창)으로 최소화한다.
    신뢰 소스(trusted_texts)에는 어느 쪽도 넣지 않는 것은 동일하다.
    """
    prev = _last_user_text(history)
    return f"{prev}\n{message}" if prev else message


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
# 사건 어휘 = 중대·일반 총칭 + PV 코딩 사전의 증상 표면형 전체(v8).
# 이전에는 중대 어휘만 있어 "두드러기·발진" 같은 일반 증상 케이스가
# 검색으로 빠져 회피됐다 — 코딩 사전과 어휘를 공유해 사전 갱신이 라우팅에
# 자동 반영된다. 케이스 맥락(_AE_CASE_MARKERS)과의 AND 조건은 유지하므로
# "두드러기 보고 기한은?" 같은 규정 질문은 여전히 검색으로 간다(과잉 매칭 방지).
_AE_EVENT_MARKERS = (
    "사망", "입원", "생명", "쇼크", "아나필락시스", "중환자실", "기형", "장애",
    "부작용", "이상사례", "이상반응",
) + symptom_keywords()


_AE_REPORT_MARKERS = ("보고서", "초안", "kaers", "icsr", "보고서 작성", "보고서 만들")


# 인지일 표기 — 값 부분은 일부러 느슨하게 잡는다(아래 docstring 참고).
_AWARENESS_RE = re.compile(r"인지일\s*[:은는]?\s*(\d{4}[-./]\d{1,2}[-./]\d{1,2})")


def _extract_awareness_date(message: str) -> str:
    """서술 속 인지일을 도구 인자로 승격한다 — 없으면 "" (v8).

    오프라인 라우터가 case_description 만 전달하면 awareness_date 는 항상
    빈 값이라, 사용자가 "인지일은 2026-07-01" 이라고 명시해도 도구가 경고
    없이 '오늘'로 폴백했다 — 15일 기한 케이스라면 기한이 그만큼 밀린 값이
    caveat 도 없이 나가는, '조용한 폴백 금지' 원칙의 라우터 계층 위반.

    값 표기는 느슨하게 잡아 그대로 전달한다 — "2026-7-1"·"2026/07/01" 같은
    형식 오류의 판정과 caveat 부착은 도구(triage)의 계약이므로, 라우터가
    여기서 정규화하거나 걸러 그 계약을 우회하지 않는다(_guess_category 와
    같은 규율: 안전장치를 만든 층 아래에서 무력화하지 않는다).
    """
    m = _AWARENESS_RE.search(message)
    return m.group(1) if m else ""


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
    """체크리스트 카테고리 추측 — 확신 있는 신호가 있을 때만 답한다.

    이전에는 아무 신호가 없어도 '품목허가'를 기본 추측으로 반환했다.
    그러면 "GMP 체크리스트 줘" 같은 미지원 요청에 엉뚱한 품목허가 체크리스트가
    자신 있게 나간다 — 도구에는 error+available 계약(조용한 빈 결과 금지)을
    만들어 놓고, 라우터가 항상 유효한 카테고리를 추측해 그 계약을 우회하고
    있었다(안전장치를 만든 층 아래에서 무력화하는 형태의 사각지대).
    확신 없으면 "" 를 반환하고, 호출부가 지원 목록 안내 + 검색 폴백으로 처리한다.
    """
    m = message
    if "변경" in m:
        return "변경허가"
    if any(k in m for k in ["안전", "이상사례", "부작용", "보고"]):
        return "안전성보고"
    if any(k in m for k in ["품목허가", "허가", "신약", "신규"]):
        return "품목허가"
    return ""


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
    # 에러 계약은 호출부(_chat_offline)가 먼저 걸러 검색 폴백으로 처리한다 —
    # 여기 도달하는 data 는 항상 정상 체크리스트다.
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
