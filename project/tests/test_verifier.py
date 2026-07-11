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


def test_date_direction_flip_is_flagged():
    """근거의 "…까지"(상한) 기한 날짜를 "… 이후"로 뒤집는 왜곡 — 날짜는 근거에
    실존해 존재 대조를 통과하므로, 방향 축이 날짜에도 있어야 잡힌다(수치에만
    방향 축이 있던 비대칭의 해소)."""
    v = verify_answer(
        "보완자료는 2026-07-25 이후에 제출하면 됩니다.",
        ["보완자료는 2026-07-25까지 제출한다."],
    )
    assert not v.ok and "2026-07-25 이후" in v.direction_conflicts
    assert "방향" in warning_text(v)


def test_date_direction_korean_notation_symmetric():
    """근거가 한국어 날짜 표기여도 정규화 후 대칭으로 대조된다."""
    v = verify_answer(
        "2026-07-25 이후 제출 가능합니다.",
        ["보완자료는 2026년 7월 25일까지 제출한다."],
    )
    assert not v.ok and "2026-07-25 이후" in v.direction_conflicts


def test_date_direction_same_class_passes():
    """까지↔이내는 같은 방향(상한) — 동의 표현을 충돌로 오탐하지 않는다."""
    v = verify_answer("2026-07-25까지 제출합니다.", ["제출 기한은 2026-07-25 이내로 한다."])
    assert v.ok


def test_date_direction_without_source_qualifier_not_flagged():
    """근거에 그 날짜의 한정어가 없으면(도구 라벨 JSON 등) 판단하지 않는다 — 보수성.
    "마감일 이후에는 지연보고로 처리된다" 같은 정당한 '이후' 용례를 오탐하지 않기
    위한 규칙이기도 하다(수치 방향 대조와 동일)."""
    v = verify_answer(
        "2026-07-25 이후에는 지연보고로 처리됩니다.",
        ['{"deadline_date": "2026-07-25"}'],
    )
    assert v.ok and not v.direction_conflicts


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
# 케이스 에코 — 사용자 사실은 지지 근거로 인정하되 from_case 라벨로 구분한다
# ---------------------------------------------------------------------------
def test_case_only_support_is_labeled_not_warned():
    """케이스의 "30일간 복용"이 답변의 "보고 기한 30일"을 지지하는 조용한 통과 —
    차단(재서술 오탐)도 침묵(승격 은폐)도 아닌 라벨로 가시화한다."""
    v = verify_answer(
        "케이스상 복용 기간은 30일이며, 보고 기한 규정은 15일 이내입니다.",
        ["규정: 인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 A정을 30일간 복용 후 두드러기 발생"],
    )
    assert v.ok                                   # 경고는 아니다
    assert v.case_origin == ["30일"]              # 그러나 등급이 다름을 라벨로
    assert "15일" not in v.case_origin            # 규정 지지 클레임은 라벨 없음
    check = next(c for c in v.checks if c.claim == "30일")
    assert check.supported and check.from_case


def test_regulation_supported_claim_not_labeled_from_case():
    """같은 값이 규정 근거에도 있으면 케이스 라벨을 붙이지 않는다(strict 우선)."""
    v = verify_answer(
        "보고 기한은 15일 이내입니다.",
        ["인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 15일 전부터 복용"],
    )
    assert v.ok and v.case_origin == []


def test_claim_nowhere_is_still_unsupported_with_facts_present():
    """케이스 계층이 있어도 어디에도 없는 값은 종전대로 미확인 경고다."""
    v = verify_answer(
        "보고 기한은 45일 이내입니다.",
        ["인지일로부터 15일 이내"],
        user_fact_texts=["환자가 30일간 복용"],
    )
    assert not v.ok and "45일" in v.unsupported


# ---------------------------------------------------------------------------
# 날짜 역할 대조 — 두 날짜가 모두 근거에 있어도 역할(기한↔인지일)이 뒤바뀌면 잡는다
# ---------------------------------------------------------------------------
_TOOL_OUT = '{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25", "deadline_days": 15}'


def test_role_swap_detected_even_when_both_dates_exist():
    """존재 대조를 '정의상' 통과하는 변조 — 역할 라벨 대조 축의 존재 이유."""
    v = verify_answer("보고 기한: 2026-07-10 (인지일 2026-07-25 기준)", [_TOOL_OUT])
    assert not v.ok
    assert "기한 2026-07-10" in v.role_conflicts and "인지일 2026-07-25" in v.role_conflicts
    assert not v.unsupported  # 두 날짜 모두 신뢰 소스에 실존 — 존재 축은 통과
    assert "날짜 역할" in warning_text(v)


def test_correct_roles_pass():
    v = verify_answer(
        "보고 기한: 2026-07-25 (인지일 2026-07-10 기준, 15일 이내)", [_TOOL_OUT, "15일 이내"]
    )
    assert v.ok and not v.role_conflicts


def test_role_check_needs_tool_labels():
    """검색 근거만 있으면(역할 라벨 없음) 판단 근거가 없다 — 플래그하지 않는다."""
    v = verify_answer("보고 기한은 2025-04-01 이후 적용", ["시행일 2025-04-01 명시"])
    assert v.ok and not v.role_conflicts


def test_role_keyword_must_be_adjacent():
    """'기한 규정은 <날짜> 시행'처럼 키워드가 날짜에 직접 붙지 않으면 역할 주장이 아니다."""
    v = verify_answer("기한 규정은 2026-07-10 시행 문서를 참조하세요", [_TOOL_OUT])
    assert not v.role_conflicts


def test_unsupported_date_near_keyword_is_existence_issue_not_role():
    """근거에 아예 없는 날짜는 존재 대조 축이 잡는다 — 역할 축과 중복 경고하지 않는다."""
    v = verify_answer("보고 기한: 2026-08-01", [_TOOL_OUT])
    assert "2026-08-01" in v.unsupported and not v.role_conflicts


# ---------------------------------------------------------------------------
# 검증 게이트 운영 계기판 (경고율 집계 — alert fatigue 조기 신호)
# ---------------------------------------------------------------------------
def test_gate_stats_records_and_snapshots():
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "unsupported": [], "checked": 2})
    gs.record({"ok": False, "unsupported": ["30일"], "role_conflicts": ["기한 2026-07-10"], "checked": 3})
    snap = gs.snapshot()
    assert snap["responses"] == 2 and snap["warned"] == 1 and snap["warn_rate"] == 0.5
    assert snap["by_axis"] == {"unsupported": 1, "role_conflicts": 1}


def test_gate_stats_checked_rate_is_immune_to_traffic_mix():
    """warn_rate 는 회피·무클레임 응답이 분모에 섞여 트래픽 믹스에 따라 착시를
    만든다 — warn_rate_checked 는 '검증할 것이 있던 응답'만 분모로 잡는다."""
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "checked": 2})                       # 클레임 있음·통과
    gs.record({"ok": False, "unsupported": ["30일"], "checked": 1})  # 클레임 있음·경고
    for _ in range(8):
        gs.record({"ok": True, "checked": 0})                   # 회피/무클레임 홍수
    snap = gs.snapshot()
    assert snap["warn_rate"] == 0.1          # 믹스에 희석된 값
    assert snap["checked_responses"] == 2
    assert snap["warn_rate_checked"] == 0.5  # 믹스와 무관한 실질 경고율


@pytest.mark.asyncio
async def test_agent_chat_feeds_gate_stats():
    """모든 chat 응답이 계기판에 집계된다 — /health 노출의 데이터 소스."""
    from src.observability import gate_stats

    before = gate_stats.snapshot()["responses"]
    r = await RaAgent().chat("신약 품목허가 심사 며칠 걸려?")
    assert r.verification
    assert gate_stats.snapshot()["responses"] == before + 1


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


def test_superseded_allowance_is_per_document():
    """경고를 끄는 스위치의 면적은 문서 단위 — 이력 검색이 반환한 문서(REG-013)의
    폐지본 인용만 면제되고, 같은 응답의 다른 폐지본 인용(현행 검색에 상류
    결함으로 섞여 든 문서)은 계속 경고된다. 전역 bool 이면 이력 턴 동안 버전
    축이 통째로 꺼진다."""
    cites = [
        {"doc_id": "REG-013", "status": "superseded"},
        {"doc_id": "REG-099", "status": "superseded"},
    ]
    v = verify_answer("구판 기준은 30일이었다", ["30일"], cites,
                      allowed_superseded_ids={"REG-013"})
    assert not v.ok and v.superseded_cited == ["REG-099"]
    # 허용 집합이 모든 인용을 덮으면 통과
    v2 = verify_answer("구판 기준은 30일이었다", ["30일"], cites,
                       allowed_superseded_ids={"REG-013", "REG-099"})
    assert v2.ok


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


def test_korean_date_notation_matches_iso_evidence():
    """한국어 날짜 표기("2026년 7월 25일")는 ISO 근거(2026-07-25)와 표기 정규화로
    대조된다 — 정규화가 없으면 (1) 올바른 답변이 미확인 날짜 오탐을 받고,
    (2) 날짜의 '25일' 성분이 기간 클레임으로 오추출된다."""
    v = verify_answer(
        "보고 기한은 2026년 7월 25일입니다.",
        ['"deadline_date": "2026-07-25", "awareness_date": "2026-07-10"'],
    )
    assert v.ok, v.summary()
    # 역방향(근거가 한국어 표기, 답변이 ISO)도 대칭으로 통과
    v2 = verify_answer("보고 기한은 2026-07-25입니다.", ["마감일은 2026년 7월 25일이다"])
    assert v2.ok, v2.summary()


def test_korean_date_in_question_gets_premise_label():
    """질문에 한국어 표기로 들어온 날짜를 답변이 ISO 로 재서술한 경우 —
    '환각'이 아니라 '전제 확인 필요'(from_question)로 라벨링되어야 한다."""
    v = verify_answer(
        "2024-06-01 기준으로는 심사 기간 규정이 없습니다.",
        ["심사 기간은 120 근무일"],
        question="2024년 6월 1일 기준 심사 기간은?",
    )
    assert "2024-06-01" in v.question_origin


def test_korean_date_component_not_extracted_as_duration():
    """"7월 15일"의 '15일'이 기간 클레임으로 추출되면, 근거의 '15일 이내'(기간)와
    우연히 일치해 통과하는 오염 경로가 생긴다 — 날짜 문맥은 날짜로만 읽는다."""
    from src.verify.verifier import extract_claims

    nums, dates = extract_claims("처리 시한은 2026년 7월 15일까지입니다")
    assert ("15", "일") not in nums
    assert "2026-07-15" in dates
