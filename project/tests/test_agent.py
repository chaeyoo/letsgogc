"""에이전트 테스트: 인텐트 라우팅 · abstention · 멀티턴 · 도구 에러 견고성 · 트레이스.

오프라인 모드(ANTHROPIC_API_KEY 없음)를 기준으로 검증한다.
"""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent
from src.observability import Trace


@pytest.mark.asyncio
async def test_intent_routing():
    agent = RaAgent()
    r1 = await agent.chat("품목허가 심사 기간은?")
    assert r1.tool_calls[0].name == "search_regulations"
    r2 = await agent.chat("이번 주 마감 임박한 업무 알려줘")
    assert r2.tool_calls[0].name == "get_ra_deadlines"
    r3 = await agent.chat("변경허가 체크리스트 알려줘")
    assert r3.tool_calls[0].name == "get_submission_checklist"
    assert all(r.answer for r in (r1, r2, r3))


@pytest.mark.asyncio
async def test_in_scope_is_grounded_and_cited():
    agent = RaAgent()
    r = await agent.chat("중대한 이상사례는 며칠 이내에 보고하나요?")
    assert r.grounded is True
    assert r.citations, "범위내 답변에는 출처가 있어야 함"


@pytest.mark.asyncio
async def test_out_of_scope_abstains():
    agent = RaAgent()
    for q in ["비트코인 지금 사도 될까요?", "우리 팀 회식 장소 추천해줘"]:
        r = await agent.chat(q)
        assert r.grounded is False, f"범위밖 '{q}' 은 abstain 해야 함"
        assert not r.citations


@pytest.mark.asyncio
async def test_offline_multiturn_followup():
    agent = RaAgent()
    history = [
        {"role": "user", "content": "변경허가 처리기한은?"},
        {"role": "assistant", "content": "..."},
    ]
    r = await agent.chat("그럼 변경신고는?", history)
    # 후속질문이 직전 맥락과 병합되어 변경 관련 문서로 검색되어야 함
    assert "변경허가" in r.tool_calls[0].args["query"]
    assert any(c["doc_id"] == "REG-002" for c in r.citations)


@pytest.mark.asyncio
async def test_trace_is_populated():
    agent = RaAgent()
    r = await agent.chat("GMP 적합판정 유효기간은?")
    assert r.trace, "트레이스 span 이 기록되어야 함"
    assert r.latency_ms > 0
    assert any(s["kind"] == "tool" for s in r.trace)


class _FailingClient:
    async def call_tool(self, name, args):
        raise RuntimeError("simulated MCP failure")


@pytest.mark.asyncio
async def test_tool_failure_is_absorbed_not_crashed():
    """MCP 도구가 예외를 던져도 크래시 대신 (에러메시지, is_error=True) 로 흡수."""
    agent = RaAgent()
    data, is_error = await agent._safe_tool_call(
        _FailingClient(), "search_regulations", {"query": "x"}, Trace()
    )
    assert is_error is True
    assert "실행 실패" in data
