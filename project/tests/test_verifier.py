"""답변 사후 검증(src/verify) 테스트 — 수치 대조 · 버전 점검 · 에이전트 통합."""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent
from src.verify.verifier import extract_claims, verify_answer, warning_text


# ---------------------------------------------------------------------------
# 클레임 추출
# ---------------------------------------------------------------------------
def test_extract_normalizes_spacing_and_zeros():
    nums, dates = extract_claims("보고는 15 일 이내, 유효기간 03년, 마감 2026-07-25")
    assert ("15", "일") in nums and ("3", "년") in nums
    assert "2026-07-25" in dates


def test_extract_ignores_bare_numbers():
    nums, _ = extract_claims("근거 3건 중 REG-005 문서 1순위")  # 단위 없는 숫자·코드
    assert ("5", "일") not in nums and not any(u == "일" for _, u in nums)


# ---------------------------------------------------------------------------
# 수치 대조
# ---------------------------------------------------------------------------
def test_supported_claim_passes():
    v = verify_answer("중대 이상사례는 15일 이내 보고합니다.", ["규정: 인지일로부터 15일 이내 신속보고"])
    assert v.ok and v.summary()["checked"] == 1


def test_unsupported_claim_is_flagged():
    v = verify_answer("중대 이상사례는 30일 이내 보고합니다.", ["규정: 인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "30일" in v.unsupported
    assert "30일" in warning_text(v)


def test_unit_paraphrase_is_flagged():
    """'15일'을 '약 2주'로 환산하면 근거에 없는 값 — 마감일 환산 오차는 리스크다."""
    v = verify_answer("약 2주 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "2주" in v.unsupported


def test_date_claim_checked_against_tool_output():
    trusted = ['{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25"}']
    ok = verify_answer("보고 기한은 2026-07-25 입니다.", trusted)
    bad = verify_answer("보고 기한은 2026-07-30 입니다.", trusted)
    assert ok.ok and not bad.ok


def test_no_claims_is_trivially_ok():
    v = verify_answer("근거를 찾지 못했습니다.", [])
    assert v.ok and v.summary()["checked"] == 0


# ---------------------------------------------------------------------------
# 인용 버전 점검 (폐지본 감지)
# ---------------------------------------------------------------------------
def test_superseded_citation_is_flagged():
    cites = [{"doc_id": "REG-013", "status": "superseded"}]
    v = verify_answer("이상사례는 30일 이내 보고", ["30일"], cites)
    assert not v.ok and v.superseded_cited == ["REG-013"]
    assert "REG-013" in warning_text(v)


def test_history_mode_allows_superseded():
    cites = [{"doc_id": "REG-013", "status": "superseded"}]
    v = verify_answer("구판 기준은 30일이었다", ["30일"], cites, allow_superseded=True)
    assert v.ok


def test_active_citation_not_flagged():
    v = verify_answer("15일 이내 보고", ["15일"], [{"doc_id": "REG-005", "status": "active"}])
    assert v.ok


# ---------------------------------------------------------------------------
# 에이전트 통합 — 모든 응답에 verification 이 부착되고, 정상 경로는 통과한다
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_attaches_verification_and_passes():
    agent = RaAgent()
    for q in [
        "신약 품목허가 심사 며칠 걸려?",                       # 검색 경로
        "환자가 복용 후 아나필락시스로 입원했어요. 언제까지 보고?",  # 트리아지 경로
        "이번 주 마감 임박한 규제 업무는?",                     # 마감일 경로
    ]:
        r = await agent.chat(q)
        assert r.verification, f"verification 누락: {q}"
        assert r.verification["ok"], f"정상 경로 오탐: {q} → {r.verification}"
        assert "⚠ 자동 검증 경고" not in r.answer


@pytest.mark.asyncio
async def test_agent_abstention_has_trivial_verification():
    agent = RaAgent()
    r = await agent.chat("오늘 점심 메뉴 추천해줘")
    assert not r.grounded
    assert r.verification["ok"] and r.verification["checked"] == 0
