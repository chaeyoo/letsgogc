"""MCP 도구 계약(contract) 테스트: 반환 스키마·에러 경로.

에이전트가 도구를 신뢰하려면 반환 형태가 안정적이어야 한다(FDE 관점의 인터페이스 보증).
"""
from __future__ import annotations

from src.mcp_server.server import (
    get_ra_deadlines,
    get_submission_checklist,
    list_regulation_documents,
    pv_case_intake,
    search_regulations,
)


def test_search_regulations_contract():
    out = search_regulations("품목허가 심사 기간", top_n=3)
    assert set(["query", "results"]).issubset(out)
    assert 1 <= len(out["results"]) <= 3
    for r in out["results"]:
        for key in ["text", "title", "source", "section", "version", "effective_date", "score"]:
            assert key in r


def test_search_as_of_returns_then_active_version():
    """시점 조회 계약: as_of 시점에 시행 중이던 버전(폐지 여부 무관)을 반환한다."""
    out = search_regulations("중대한 이상사례 보고 기한", top_n=1, as_of="2025-01-01")
    assert out["results"], "시점 조회가 0건 — 당시 현행 버전이 걸러졌다"
    assert out["results"][0]["doc_id"] == "REG-013"
    assert out["results"][0]["status"] == "superseded"  # 출처에 폐지 상태가 그대로 노출


def test_search_bad_as_of_is_explicit_error():
    """형식이 틀린 as_of 를 조용히 무시하고 현행 기준으로 답하면 '그 시점 규정'을
    받았다고 믿게 되는 자신 있는 오답 — 명시적 에러 계약으로 답한다(다른 도구의
    error+available 계약과 같은 원칙)."""
    bad = search_regulations("보고 기한", as_of="2025/01/01")
    assert "error" in bad and "expected" in bad
    assert "results" not in bad


def test_get_ra_deadlines_contract():
    out = get_ra_deadlines(within_days=365)
    assert "today" in out and "deadlines" in out
    assert out["count"] == len(out["deadlines"])
    # 마감일 오름차순 정렬 보장
    dates = [d["due_date"] for d in out["deadlines"]]
    assert dates == sorted(dates)
    assert all("d_day" in d for d in out["deadlines"])


def test_get_ra_deadlines_type_filter():
    out = get_ra_deadlines(within_days=365, task_type="안전관리")
    assert all(d["type"] == "안전관리" for d in out["deadlines"])


def test_deadlines_unknown_type_is_explicit_error():
    """오타 난 유형 필터에 '마감 없음'(자신 있는 오답)이 아니라 에러+가용 목록으로
    답한다 — 에이전트가 available 을 보고 스스로 정정 재시도할 수 있는 계약."""
    bad = get_ra_deadlines(within_days=365, task_type="존재하지않는유형")
    assert "error" in bad and bad["available"]
    ok = get_ra_deadlines(within_days=365, task_type=bad["available"][0])
    assert "deadlines" in ok and "error" not in ok


def test_checklist_known_and_unknown():
    ok = get_submission_checklist("품목허가")
    assert ok["category"] == "품목허가" and ok["items"]
    bad = get_submission_checklist("존재하지않는유형")
    assert "error" in bad and "available" in bad


def test_pv_intake_prompt_encodes_sop():
    """MCP Prompt: 케이스 처리 SOP(도구 호출 순서)가 프롬프트에 배포된다."""
    p = pv_case_intake("환자가 복용 후 입원")
    assert "환자가 복용 후 입원" in p
    for tool in ["assess_adverse_event", "search_regulations", "draft_ae_report"]:
        assert tool in p


def test_list_documents_contract():
    out = list_regulation_documents()
    assert out["count"] >= 12
    assert all("doc_id" in d and "title" in d for d in out["documents"])
