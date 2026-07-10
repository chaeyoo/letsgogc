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


def test_extract_working_days_as_distinct_unit():
    """'근무일'은 '일'과 다른 단위 — 코퍼스 핵심 처리기한의 검증 사각지대였다."""
    nums, _ = extract_claims("신약: 신청일로부터 **120 근무일** 이내")
    assert ("120", "근무일") in nums
    assert ("120", "일") not in nums  # '일'로 오분류되면 환산 오류를 못 잡는다


def test_extract_range_lower_bound():
    """'15~30일'의 하한(15일)도 클레임이다 — 주 정규식은 상한만 잡는다."""
    nums, _ = extract_claims("접수 후 15~30일 소요")
    assert ("15", "일") in nums and ("30", "일") in nums


def test_extract_comma_grouped_number():
    """'1,000회'가 '000회'로 오추출되면 값 대조가 무의미해진다."""
    nums, _ = extract_claims("수수료는 1,000회 기준")
    assert ("1000", "회") in nums and not any(n == "0" for n, _ in nums)


def test_extract_native_numerals_symmetric():
    """고유어 수사(보름=15일)는 답변·근거 양쪽에서 같은 canonical 로 추출된다."""
    nums, _ = extract_claims("보름 이내 보고, 한 달 뒤 재심사, 일주일 관찰")
    assert ("15", "일") in nums and ("1", "개월") in nums and ("7", "일") in nums


def test_extract_week_suffix_normalized():
    nums, _ = extract_claims("2주일 이상 지속")
    assert ("2", "주") in nums


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


def test_working_day_paraphrase_is_flagged():
    """'120 근무일'을 '120일'로 옮기면 실제 달력 기한이 달라진다 — 단위 환산 오류."""
    v = verify_answer("심사는 120일 이내입니다.", ["신청일로부터 **120 근무일** 이내"])
    assert not v.ok and "120일" in v.unsupported


def test_native_numeral_equivalence_passes():
    """보름=15일은 환산이 아니라 표기 변형 — 오탐을 내면 안 된다(alert fatigue)."""
    v = verify_answer("보름 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert v.ok


def test_wrong_native_numeral_is_flagged():
    """값이 다른 고유어(열흘=10일)는 v1에서 아예 추출되지 않아 조용히 통과했다."""
    v = verify_answer("열흘 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "열흘" in v.unsupported


def test_supported_claim_carries_evidence_snippet():
    """지원된 클레임에는 근거 위치 스니펫이 붙는다 — 사람의 대조를 빠르게."""
    v = verify_answer("15일 이내 보고합니다.", ["규정 제5조: 인지일로부터 15일 이내 신속보고한다."])
    check = next(c for c in v.checks if c.claim == "15일")
    assert check.supported and "15일 이내 신속보고" in check.evidence


# ---------------------------------------------------------------------------
# 방향 한정어 — 수치가 맞아도 방향이 뒤집히면 컴플라이언스 오류
# ---------------------------------------------------------------------------
def test_direction_flip_is_flagged():
    v = verify_answer("90일 이후에 제출하면 됩니다.", ["보완 회신은 통상 **90일** 이내에 제출한다."])
    assert not v.ok and "90일 이후" in v.direction_conflicts
    assert "방향" in warning_text(v)


def test_direction_same_class_passes():
    """이내↔이하는 같은 방향(상한) — 동의 표현을 충돌로 오탐하지 않는다."""
    v = verify_answer("90일 이하로 제출합니다.", ["보완 회신은 통상 90일 이내에 제출한다."])
    assert v.ok


def test_direction_without_source_qualifier_not_flagged():
    """근거에 한정어가 없으면 판단 근거가 없다 — 보수적으로 플래그하지 않는다."""
    v = verify_answer("90일 이후에 제출합니다.", ["보완 회신 기간은 90일로 한다."])
    assert v.ok


def test_direction_matching_lower_bound_passes():
    v = verify_answer("6개월 이상 수행합니다.", ["안정성시험은 6개월 이상 수행한다."])
    assert v.ok


# ---------------------------------------------------------------------------
# 질문 에코 — 사용자 전제는 신뢰 소스가 아니라 '전제 확인' 라벨
# ---------------------------------------------------------------------------
def test_question_origin_claim_labeled_separately():
    v = verify_answer(
        "30일이 아니라 15일 이내입니다.",
        ["인지일로부터 15일 이내 신속보고"],
        question="보고 기한이 30일 맞나요?",
    )
    assert not v.ok and "30일" in v.question_origin
    w = warning_text(v)
    assert "전제 확인" in w and "환각" not in w


def test_question_does_not_promote_claim_to_supported():
    """질문에 있던 수치라도 supported 로 승격되지는 않는다 — 라벨만 다르다."""
    v = verify_answer("답은 30일입니다.", ["15일 이내"], question="30일인가요?")
    assert not v.ok and "30일" in v.unsupported


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
