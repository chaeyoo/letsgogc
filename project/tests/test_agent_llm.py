"""LLM 모드 tool-use 루프의 계약 테스트 — 스텁 LLM 으로 실제 루프를 태운다.

LLM 모드의 안전 배선(신뢰 소스 수집·이력 허용 집합·검증 게이트 통과)은
오프라인 모드와 코드 경로가 다른데, 실 API 키 없이는 실행 자체가 안 되어
테스트 커버리지가 0이었다 — '두 모드는 같은 계약'이라는 문서 주장이 한쪽
모드에서는 실행으로 확인된 적이 없던 셈이다. anthropic 클라이언트만 스텁으로
바꾸고 MCP·RAG·검증 게이트는 전부 실물을 태워, LLM 모드 경로의 계약을
실행 가능한 형태로 고정한다.
"""
from __future__ import annotations

import sys
import types

import pytest

from src import config
from src.agent.agent import RaAgent


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stub_anthropic(monkeypatch, responses: list) -> None:
    """messages.create 가 준비된 응답을 순서대로 돌려주는 가짜 anthropic 모듈."""
    state = {"i": 0}

    class _Messages:
        async def create(self, **kwargs):
            r = responses[state["i"]]
            state["i"] += 1
            return r

    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = lambda api_key=None: types.SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")


@pytest.mark.asyncio
async def test_llm_history_search_allows_only_returned_superseded_docs(monkeypatch):
    """as_of(이력) 검색이 '실제로 반환한' 폐지본 인용은 경고에서 면제된다 —
    허용이 응답 전역 스위치가 아니라 문서 단위 집합으로 작동하는지의 배선 확인."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "중대한 이상사례 보고 기한", "as_of": "2025-01-01"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            # 기준일(as_of)을 답변에 재서술하지 않는 형태 — as_of 는 사용자 입력
            # 에코라 신뢰 소스에서 제거되므로, 재서술하면 미확인 날짜 경고가
            # 붙는 것이 의도된 보수 동작이다(사용자 지정 기준일은 근거가 아니다).
            _Block(type="text",
                   text="그 시점 당시 현행이던 구판 기준으로 보고 기한은 30일 이내였습니다."),
        ]),
    ])
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert any(c.get("doc_id") == "REG-013" for c in r.citations), r.citations
    # 이력 검색이 반환한 문서의 폐지본 인용 — 경고 면제, 수치(30일)도 구판 근거로 지원
    assert r.verification["superseded_cited"] == []
    assert r.verification["ok"], r.verification


@pytest.mark.asyncio
async def test_llm_case_echo_support_is_labeled(monkeypatch):
    """모델이 케이스 서술의 수치를 재서술하면 — 지지 근거가 도구 출력의 케이스
    에코뿐이므로 from_case(case_origin) 라벨이 붙는다. 규정·도구 계산에서 온
    수치(15일)는 라벨이 붙지 않는다: '사용자 서술 근거'와 '규정 근거'의 구분."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="assess_adverse_event",
                   input={"case_description": "환자가 A정을 30일간 복용 후 두드러기로 입원했습니다"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text",
                   text="복용 기간 30일의 입원 케이스로 중대에 해당하며, 보고 기한은 15일 이내입니다."),
        ]),
    ])
    r = await RaAgent().chat("이 케이스 언제까지 보고해야 하나요?")
    assert r.mode == "llm"
    assert r.verification["ok"], r.verification          # 경고는 없다
    assert r.verification["case_origin"] == ["30일"]     # 케이스 유래 지지 라벨
    assert "15일" not in r.verification["case_origin"]   # 규정(REG-005) 근거 지지


@pytest.mark.asyncio
async def test_llm_failed_as_of_does_not_open_allowance(monkeypatch):
    """형식이 틀린 as_of 호출(에러 계약)은 허용 집합을 채우지 못한다 — 시점
    검색이 성공한 적 없는데 폐지본 경고만 꺼지는 경로의 차단. 에러 계약 응답이
    신뢰 소스로 승격되지 않는 것도 함께 고정한다(에러 문구 속 사용자 입력 에코)."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "이상사례 보고 기한", "as_of": "2025/01/01"}),  # 형식 오류
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="보고 기한 규정을 확인하지 못했습니다."),
        ]),
    ])
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []          # 성공한 검색이 없다
    assert r.verification["ok"]       # 수치 클레임 없는 안내 — 자명 통과
    # 에러 계약이 신뢰 소스로 승격됐다면 '2025' 같은 에코 값이 지원 근거가 됐을 것
    assert all(not c["supported"] for c in r.verification["checks"]) or not r.verification["checks"]
