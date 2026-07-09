"""PV 워크플로 심화 테스트 — 인과성(WHO-UMC) · 용어 코딩 · ICSR 보고서 초안.

트리아지 이후의 PV 업무(인과성 평가 → 코딩 → 보고서 작성)가
규칙 기반으로 재현 가능하게 동작하는지 보증한다.
"""
from __future__ import annotations

from src.agent.agent import RaAgent, _route_intent
from src.mcp_server.server import assess_adverse_event, draft_ae_report
from src.pv.causality import (
    CERTAIN,
    POSSIBLE,
    PROBABLE,
    UNASSESSABLE,
    UNLIKELY,
    assess_causality,
)
from src.pv.coding import code_terms
from src.pv.report import build_report


# ---------------------------------------------------------------------------
# 인과성 평가 (WHO-UMC 제안)
# ---------------------------------------------------------------------------
def test_causality_certain_needs_all_signals():
    c = assess_causality("복용 후 두드러기 발생, 중단 후 호전, 재투여 후 재발")
    assert c.suggested == CERTAIN


def test_causality_probable_without_rechallenge():
    c = assess_causality("복용 후 두드러기 발생, 중단 후 호전됨")
    assert c.suggested == PROBABLE
    # 재투여 정보가 없으므로 되물을 질문에 포함된다
    assert any("재투여" in q for q in c.missing_info)


def test_causality_alternative_cause_caps_at_possible():
    """대체 원인(병용약)이 있으면 dechallenge 가 있어도 Possible 이상 못 올라간다."""
    c = assess_causality("복용 후 발진, 중단 후 호전. 다만 항생제를 병용 중이었음")
    assert c.suggested == POSSIBLE
    assert c.signals["대체 원인 가능성(병용약·기저질환)"]


def test_causality_unlikely_and_unassessable():
    assert assess_causality("기저질환 악화로 입원").suggested == UNLIKELY
    assert assess_causality("두통이 있었다").suggested == UNASSESSABLE


def test_causality_is_suggestion_not_verdict():
    """정보가 없는 요소는 충족으로 치지 않는다(등급은 보수적으로만)."""
    c = assess_causality("복용 후 구토")
    assert c.suggested == POSSIBLE  # 시간관계만으로 Probable 로 올려주지 않는다
    assert len(c.missing_info) == 3  # dechallenge·rechallenge·대체원인 전부 확인 필요


# ---------------------------------------------------------------------------
# 표준 용어 코딩 (MedDRA 방식)
# ---------------------------------------------------------------------------
def test_coding_maps_colloquial_to_pt():
    coded = code_terms("복용 후 숨쉬기 힘들고 온몸에 두드러기가 났다")
    pts = {t.pt for t in coded}
    assert {"호흡곤란", "두드러기"} <= pts
    dysp = next(t for t in coded if t.pt == "호흡곤란")
    assert dysp.pt_en == "Dyspnoea" and "호흡기" in dysp.soc


def test_coding_dedupes_same_pt():
    """같은 PT 를 가리키는 표현이 여러 번 나와도 1건으로 집계한다."""
    coded = code_terms("어지러움과 현기증, 어지럼 증상")
    assert len([t for t in coded if t.pt == "어지러움"]) == 1


def test_coding_preserves_narrative_order():
    coded = code_terms("두통이 먼저 왔고 이후 구토를 했다")
    assert [t.pt for t in coded] == ["두통", "구토"]


def test_coding_no_match_returns_empty():
    assert code_terms("품목허가 심사 기간은?") == []


# ---------------------------------------------------------------------------
# ICSR 보고서 초안 (최소보고요건 + 조립)
# ---------------------------------------------------------------------------
def test_report_complete_case_is_reportable():
    r = build_report(
        "45세 남성 환자가 A정 복용 후 아나필락시스로 입원",
        reporter="의사", awareness_date="2026-07-01",
    )
    assert r.reportable and not r.missing
    assert r.triage.is_serious and r.triage.deadline_days == 0
    assert "최소보고요건 충족" in r.draft_markdown


def test_report_detects_missing_minimum_criteria():
    """의심약·보고자가 없으면 reportable=False + 보완 항목 안내(ICH E2D 4요소)."""
    r = build_report("복용 후 두드러기가 생겨 입원")
    assert not r.reportable
    assert any("환자" in m for m in r.missing)
    assert any("보고자" in m for m in r.missing)
    assert any("의심 의약품" in m for m in r.missing)
    assert any("최소보고요건 보완" in f for f in r.followups)


def test_report_draft_contains_full_pv_workflow():
    r = build_report(
        "환자가 B캡슐 복용 후 두드러기 발생으로 입원, 중단 후 호전",
        reporter="약사", awareness_date="2026-07-01",
    )
    d = r.draft_markdown
    assert "중대성" in d and "2026-07-16" in d          # 트리아지(15일 기한)
    assert PROBABLE in d                                  # 인과성 제안
    assert "Urticaria" in d                               # 용어 코딩
    assert "PV 담당자" in d                               # 사람 확정 caveat 강제


# ---------------------------------------------------------------------------
# MCP 도구 계약
# ---------------------------------------------------------------------------
def test_draft_tool_contract_and_basis():
    out = draft_ae_report(
        "환자가 복용 후 아나필락시스로 입원", reporter="의사", suspected_drug="A정",
        awareness_date="2026-07-01",
    )
    for key in ["reportable", "missing", "followups", "draft_markdown",
                "causality", "coded_terms", "pii_masked", "basis"]:
        assert key in out
    assert "REG-005" in [r["doc_id"] for r in out["basis"]["results"]]


def test_draft_tool_masks_pii_before_drafting():
    out = draft_ae_report("환자 김철수님(010-1234-5678)이 C정 복용 후 사망", reporter="의사")
    assert "김철수" not in out["draft_markdown"] and "1234" not in out["draft_markdown"]
    assert out["pii_masked"]
    # 마스킹 후에도 '환자 존재 신호'([이름]님)는 남아 최소요건 판정이 가능하다
    assert not any("환자" in m for m in out["missing"])


def test_assess_tool_now_includes_causality_and_coding():
    out = assess_adverse_event("복용 후 두드러기로 입원, 중단 후 호전", awareness_date="2026-07-01")
    assert out["causality"]["suggested"] == PROBABLE
    assert any(t["pt"] == "두드러기" for t in out["coded_terms"])


# ---------------------------------------------------------------------------
# 에이전트 라우팅 + 오프라인 E2E
# ---------------------------------------------------------------------------
def test_route_report_request_to_draft_tool():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. KAERS 보고서 초안 작성해줘") == "ae_report"


def test_route_case_without_report_request_stays_triage():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?") == "ae_triage"


async def test_agent_report_end_to_end():
    agent = RaAgent()
    r = await agent.chat(
        "환자 박영희님(010-1111-2222)이 D정 복용 후 두드러기로 입원했습니다. 보고서 초안 만들어줘"
    )
    assert [t.name for t in r.tool_calls] == ["draft_ae_report"]
    assert "개별사례보고(ICSR) 초안" in r.answer and "Urticaria" in r.answer
    assert r.grounded and any(c["doc_id"] == "REG-005" for c in r.citations)
    # 입구 마스킹: 초안·트레이스 어디에도 원 PII가 없다
    assert "박영희" not in r.answer and "1111" not in str(r.trace)


# ---------------------------------------------------------------------------
# 코딩 2·3계층 — 후보(LLT 참조) 제시와 미코딩 감지, 보고요건 연쇄 차단
# ---------------------------------------------------------------------------
def test_candidate_tier_suggests_but_never_confirms():
    """1계층 미수록 표현('청력 상실')은 후보로만 제시된다 — 확정 목록에
    섞이면 시그널 집계가 오염되므로 타입/필드 수준에서 분리를 보증한다."""
    from src.pv.coding import suggest_candidates

    case = "환자가 항결핵제O정을 복용 후 청력 상실이 발생했습니다."
    coded = code_terms(case)
    cands = suggest_candidates(case, coded)
    assert all(t.pt != "난청" for t in coded), "후보가 확정에 섞이면 안 된다"
    assert [c.pt for c in cands] == ["난청"]
    assert cands[0].needs_confirmation


def test_candidate_tier_skips_already_confirmed_pt():
    """확정 사전이 이미 잡은 PT는 후보로 중복 제시하지 않는다(이중 집계 방지)."""
    from src.pv.coding import suggest_candidates

    case = "복용 후 오심과 함께 속이 울렁거린다고 합니다."  # 오심은 1계층 수록
    coded = code_terms(case)
    assert any(t.pt == "오심" for t in coded)
    assert suggest_candidates(case, coded) == []


def test_uncoded_flag_detects_specific_symptom_only():
    """3계층은 '구체적 증상 서술'만 감지한다 — 막연한 서술("몸이 좋지 않다")은
    ICH E2D의 specificity 요구상 ④요건 미충족이 올바른 판정이라 잡지 않는다."""
    from src.pv.coding import flag_uncoded_expressions

    hit = flag_uncoded_expressions("복용 후 손발이 저릿저릿하다고 호소", [], [])
    assert hit and "저릿" in hit[0]
    assert flag_uncoded_expressions("지인이 약을 먹고 몸이 좋지 않다고 합니다", [], []) == []


def test_report_candidate_satisfies_event_criterion():
    """④요건의 본질은 '구체적 이상사례 서술의 존재'다 — 1계층 코딩이 실패해도
    후보(2계층)가 있으면 reportable 이고, 확정은 follow-up 으로 요청된다."""
    r = build_report(
        "환자가 위장약M정을 복용 후 속이 울렁거리고 힘들다고 호소했습니다. 약사가 보고했습니다."
    )
    assert r.coded_terms == [] and [c.pt for c in r.candidate_terms] == ["오심"]
    assert r.reportable, r.missing
    assert any("후보 승인/기각" in f for f in r.followups)
    assert "후보(승인/기각 필요)" in r.draft_markdown


def test_report_uncoded_signal_satisfies_event_criterion():
    """어느 사전에도 없는 심층 롱테일도 '증상 서술 감지'로 ④요건은 충족 —
    코딩 실패가 '보고 불가' 오판으로 연쇄되지 않는다(PT 부여는 사람 몫)."""
    r = build_report(
        "환자가 진통제R정을 복용 후 손발이 저릿저릿하다고 호소했습니다. 의사가 보고했습니다."
    )
    assert r.coded_terms == [] and r.candidate_terms == []
    assert r.uncoded_expressions and r.reportable
    assert any("PT 부여 필요" in f for f in r.followups)


def test_report_vague_case_still_fails_event_criterion():
    """보수성 유지 가드: 막연한 서술만 있는 케이스는 여전히 ④요건 미충족 —
    후보/감지 계층이 '아무거나 통과'로 변질되지 않았음을 고정한다."""
    r = build_report("지인이 약을 먹고 몸이 좋지 않다고 합니다.")
    assert not r.reportable
    assert any("④" in m for m in r.missing)


def test_tool_exposes_candidate_and_uncoded_layers():
    """MCP 도구 계약: 확정/후보/미코딩 3계층이 응답 스키마로 구분 노출된다."""
    out = draft_ae_report(
        "환자가 당뇨약C정을 복용 후 심한 저혈당으로 입원했습니다. 약사가 보고했습니다."
    )
    assert out["coded_terms"] == []
    assert [c["pt"] for c in out["candidate_terms"]] == ["저혈당"]
    assert all(c["needs_confirmation"] for c in out["candidate_terms"])
