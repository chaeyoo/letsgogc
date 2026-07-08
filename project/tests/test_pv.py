"""PV(약물감시) 모듈 테스트 — AE 트리아지 · PII 비식별화 · 질의 확장."""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent, _route_intent
from src.mcp_server.server import assess_adverse_event
from src.pv.redactor import redact
from src.pv.triage import assess_case
from src.rag.synonyms import expand_query


# ---------------------------------------------------------------------------
# AE 트리아지 (규칙 기반 중대성 판정 + 기한 계산)
# ---------------------------------------------------------------------------
def test_triage_death_is_immediate():
    t = assess_case("환자가 투여 3일 후 사망", awareness_date="2026-07-08")
    assert t.is_serious and "사망" in t.criteria_met
    assert t.deadline_days == 0
    assert t.deadline_date == "2026-07-08"  # 지체 없이 = 인지일 당일


def test_triage_hospitalization_is_15_days():
    t = assess_case(
        "복용 후 두드러기가 생겨 입원했고, 허가사항에 없는 예상치 못한 반응",
        awareness_date="2026-07-01",
    )
    assert t.is_serious and t.expectedness == "unexpected"
    assert t.deadline_days == 15
    assert t.deadline_date == "2026-07-16"  # 인지일 + 15일


def test_triage_non_serious_goes_to_psur():
    t = assess_case("가벼운 두통이 있었으나 회복")
    assert not t.is_serious
    assert t.deadline_date is None
    assert "PSUR" in t.route


def test_triage_unknown_expectedness_is_conservative():
    """예상 여부를 모르면 '예상치 못한 사례'로 보수 적용(안전한 실패)."""
    t = assess_case("복용 후 입원", awareness_date="2026-07-01")
    assert t.expectedness == "unknown"
    assert t.deadline_days == 15  # 완화하지 않고 15일 트래킹
    assert any("보수" in c for c in t.caveats)


def test_triage_always_flags_human_confirmation():
    for case in ["환자가 사망", "가벼운 두통"]:
        t = assess_case(case)
        assert any("담당자가 확정" in c for c in t.caveats)


# ---------------------------------------------------------------------------
# PII 비식별화
# ---------------------------------------------------------------------------
def test_redact_masks_common_pii():
    r = redact("환자 김철수님(750101-1234567, 010-1234-5678, kim@test.com) 입원")
    assert "750101" not in r.text and "010-1234" not in r.text and "kim@test.com" not in r.text
    assert "김철수" not in r.text
    assert {"주민등록번호", "전화번호", "이메일", "이름(호칭)"} <= set(r.counts)


def test_redact_report_never_contains_original_values():
    r = redact("연락처 010-9876-5432")
    assert "9876" not in str(r.summary())  # 요약에는 유형·건수만


def test_redact_skips_common_titles():
    """'선생님/담당자님' 같은 일반 호칭은 이름으로 오탐하지 않는다."""
    r = redact("선생님, 담당자님께 전달해 주세요")
    assert r.text == "선생님, 담당자님께 전달해 주세요"
    assert not r.redacted


def test_redact_clean_text_untouched():
    r = redact("품목허가 심사 기간은 얼마나 걸리나요?")
    assert not r.redacted and r.text == "품목허가 심사 기간은 얼마나 걸리나요?"


# ---------------------------------------------------------------------------
# MCP 도구 (assess_adverse_event)
# ---------------------------------------------------------------------------
def test_assess_tool_returns_grounded_basis():
    out = assess_adverse_event("환자가 복용 후 아나필락시스로 입원", awareness_date="2026-07-01")
    assert out["is_serious"] and out["deadline_days"] == 0
    # 판정 근거 규정 문단(REG-005)이 출처와 함께 부착된다
    ids = [r["doc_id"] for r in out["basis"]["results"]]
    assert "REG-005" in ids


def test_assess_tool_masks_pii_in_result():
    out = assess_adverse_event("환자 박영희님(010-1111-2222)이 복용 후 사망")
    assert "박영희" not in out["case"] and "1111" not in out["case"]
    assert out["pii_masked"]


# ---------------------------------------------------------------------------
# 질의 확장 (도메인 동의어)
# ---------------------------------------------------------------------------
def test_expand_query_bridges_vocabulary_mismatch():
    q = expand_query("부작용 보고 기한은?")
    assert "이상사례" in q
    assert q.startswith("부작용 보고 기한은?")  # 원 질의는 항상 보존


def test_expand_query_no_hit_returns_original():
    q = "품목허가 심사 기간"
    assert expand_query(q) == q


def test_expansion_improves_colloquial_retrieval(pipeline):
    """'설명서'(구어) → '첨부문서'(문서 용어) 확장이 정답 문서를 회수한다."""
    q = "제품 설명서에 경고 문구는 어디에 표시하나요?"
    with_exp = pipeline.retriever.retrieve(q, top_k=8, rerank_n=1, expand=True)
    assert with_exp[0].chunk.doc_id == "REG-004"


# ---------------------------------------------------------------------------
# 에이전트 통합 (라우팅 + 입구 마스킹)
# ---------------------------------------------------------------------------
def test_route_case_description_to_triage():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?") == "ae_triage"


def test_route_regulation_question_stays_search():
    # 케이스 서술이 아닌 '규정 질문'은 문서 검색으로
    assert _route_intent("중대한 이상사례는 며칠 안에 보고해야 하나요?") == "search"


@pytest.mark.asyncio
async def test_agent_triage_end_to_end():
    agent = RaAgent()
    r = await agent.chat("환자 김철수님(010-1234-5678)이 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?")
    assert [t.name for t in r.tool_calls] == ["assess_adverse_event"]
    assert r.grounded and any(c["doc_id"] == "REG-005" for c in r.citations)
    # 입구에서 마스킹: 답변·트레이스 어디에도 원 PII가 없다
    assert "김철수" not in r.answer and "1234" not in r.answer
    assert "김철수" not in str(r.trace)
    assert {x["type"] for x in r.redactions} == {"전화번호", "이름(호칭)"}
