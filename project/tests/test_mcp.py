"""MCP 도구 계약(contract) 테스트: 반환 스키마·에러 경로.

에이전트가 도구를 신뢰하려면 반환 형태가 안정적이어야 한다(FDE 관점의 인터페이스 보증).
완전 분리 구조에 맞춰 프로덕션과 같은 경로로 검사한다 — conftest 가 세션 1회
기동한 HTTP MCP 서버에 fastmcp Client 로 붙어, 도구 호출이 실제 streamable HTTP
wire 를 왕복한다(직렬화·에러 계약이 transport 를 건너서도 성립하는지까지 고정).
"""
from __future__ import annotations

from fastmcp import Client

EXPECTED_TOOLS = {
    "search_regulations", "get_ra_deadlines", "get_submission_checklist",
    "assess_adverse_event", "draft_ae_report", "list_regulation_documents",
}


async def _call(url: str, name: str, args: dict) -> dict:
    async with Client(url) as c:
        return (await c.call_tool(name, args)).data


async def test_server_exposes_expected_tools(mcp_server_url):
    """도구 6종 노출 계약 — preflight check_mcp_reachable 과 같은 기대 집합."""
    async with Client(mcp_server_url) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == EXPECTED_TOOLS


async def test_search_regulations_contract(mcp_server_url):
    out = await _call(mcp_server_url, "search_regulations", {"query": "품목허가 심사 기간", "top_n": 3})
    assert set(["query", "results"]).issubset(out)
    assert 1 <= len(out["results"]) <= 3
    for r in out["results"]:
        for key in ["text", "title", "source", "section", "version", "effective_date", "score"]:
            assert key in r


async def test_search_as_of_returns_then_active_version(mcp_server_url):
    """시점 조회 계약: as_of 시점에 시행 중이던 버전(폐지 여부 무관)을 반환한다."""
    out = await _call(
        mcp_server_url, "search_regulations",
        {"query": "중대한 이상사례 보고 기한", "top_n": 1, "as_of": "2025-01-01"},
    )
    assert out["results"], "시점 조회가 0건 — 당시 현행 버전이 걸러졌다"
    assert out["results"][0]["doc_id"] == "REG-013"
    assert out["results"][0]["status"] == "superseded"  # 출처에 폐지 상태가 그대로 노출


async def test_search_bad_as_of_is_explicit_error(mcp_server_url):
    """형식이 틀린 as_of 를 조용히 무시하고 현행 기준으로 답하면 '그 시점 규정'을
    받았다고 믿게 되는 자신 있는 오답 — 명시적 에러 계약으로 답한다(다른 도구의
    error+available 계약과 같은 원칙)."""
    bad = await _call(mcp_server_url, "search_regulations", {"query": "보고 기한", "as_of": "2025/01/01"})
    assert "error" in bad and "expected" in bad
    assert "results" not in bad


async def test_search_empty_query_is_explicit_error(mcp_server_url):
    """빈 질의는 무신호라 리트리버가 빈 결과를 반환하는데, 그것을 그대로
    흘리면 "results": [] 가 '관련 규정 없음'이라는 자신 있는 오답으로
    소비된다 — as_of 형식 오류와 동일한 {"error","expected"} 계약으로 답한다
    (조용한 빈 결과 금지). 에러 문구에 예시 질의를 넣지 않는 규율도 동일."""
    for q in ["", "   "]:
        bad = await _call(mcp_server_url, "search_regulations", {"query": q})
        assert "error" in bad and "expected" in bad
        assert "results" not in bad


async def test_get_ra_deadlines_contract(mcp_server_url):
    out = await _call(mcp_server_url, "get_ra_deadlines", {"within_days": 365})
    assert "today" in out and "deadlines" in out
    assert out["count"] == len(out["deadlines"])
    # 마감일 오름차순 정렬 보장
    dates = [d["due_date"] for d in out["deadlines"]]
    assert dates == sorted(dates)
    assert all("d_day" in d for d in out["deadlines"])


async def test_get_ra_deadlines_type_filter(mcp_server_url):
    out = await _call(mcp_server_url, "get_ra_deadlines", {"within_days": 365, "task_type": "안전관리"})
    assert all(d["type"] == "안전관리" for d in out["deadlines"])


async def test_deadlines_unknown_type_is_explicit_error(mcp_server_url):
    """오타 난 유형 필터에 '마감 없음'(자신 있는 오답)이 아니라 에러+가용 목록으로
    답한다 — 에이전트가 available 을 보고 스스로 정정 재시도할 수 있는 계약."""
    bad = await _call(mcp_server_url, "get_ra_deadlines", {"within_days": 365, "task_type": "존재하지않는유형"})
    assert "error" in bad and bad["available"]
    ok = await _call(mcp_server_url, "get_ra_deadlines", {"within_days": 365, "task_type": bad["available"][0]})
    assert "deadlines" in ok and "error" not in ok


async def test_checklist_known_and_unknown(mcp_server_url):
    ok = await _call(mcp_server_url, "get_submission_checklist", {"category": "품목허가"})
    assert ok["category"] == "품목허가" and ok["items"]
    bad = await _call(mcp_server_url, "get_submission_checklist", {"category": "존재하지않는유형"})
    assert "error" in bad and "available" in bad


async def test_pv_intake_prompt_encodes_sop(mcp_server_url):
    """MCP Prompt: 케이스 처리 SOP(도구 호출 순서)가 프롬프트에 배포된다."""
    async with Client(mcp_server_url) as c:
        res = await c.get_prompt("pv_case_intake", {"case_description": "환자가 복용 후 입원"})
    p = res.messages[0].content.text
    assert "환자가 복용 후 입원" in p
    for tool in ["assess_adverse_event", "search_regulations", "draft_ae_report"]:
        assert tool in p


async def test_free_text_args_are_masked_at_tool_layer(mcp_server_url):
    """도구 계층 마스킹의 면적 — 'MCP 도구 계층은 모든 자유 텍스트 인자를
    마스킹한다'는 주장이 PV 도구 2종에서만 참이던 범위 과확장의 봉합(v7).
    stdio 단독 사용 시 이 인자들은 에이전트 입구 마스킹을 거치지 않는다:
    query 는 voyage 임베더 구성에서 외부 API 로 송신되고, 필터 인자는 미매칭
    에러 문구에 그대로 에코되며, Prompt 는 원문을 LLM 프롬프트에 삽입한다."""
    # 검색 query — 결과 에코(query)에 원문 개인정보가 남으면 안 된다
    out = await _call(
        mcp_server_url, "search_regulations",
        {"query": "김철수님 010-1234-5678 케이스 신속보고 규정", "top_n": 1},
    )
    assert "010-1234-5678" not in out["query"] and "김철수" not in out["query"]
    # 필터 인자 — 미매칭 에러 에코에 원문 개인정보가 남으면 안 된다
    bad = await _call(mcp_server_url, "get_ra_deadlines", {"task_type": "담당자 010-1234-5678"})
    assert "error" in bad and "010-1234-5678" not in bad["error"]
    bad2 = await _call(mcp_server_url, "get_submission_checklist", {"category": "김철수님 요청 체크리스트"})
    assert "error" in bad2 and "김철수" not in bad2["error"]
    # Prompt — SOP 프롬프트에 삽입되기 전에 마스킹
    async with Client(mcp_server_url) as c:
        res = await c.get_prompt(
            "pv_case_intake", {"case_description": "환자 김철수님(010-1234-5678)이 복용 후 입원"}
        )
    p = res.messages[0].content.text
    assert "010-1234-5678" not in p and "김철수" not in p
    assert "[전화번호]" in p
    # 날짜 형식 인자의 에러 에코 — 형식이 틀린 as_of 는 임의 문자열일 수 있다
    bad_as_of = await _call(
        mcp_server_url, "search_regulations", {"query": "보고 기한", "as_of": "김철수님 010-1234-5678"}
    )
    assert "error" in bad_as_of and "010-1234-5678" not in bad_as_of["error"]
    assert "김철수" not in bad_as_of["error"]


def test_resource_error_echo_is_masked():
    """Resource 의 미매칭 doc_id 에러 에코 마스킹 — 서버 내부 단위 테스트로 유지.

    regulation://{doc_id} 는 URI 템플릿이라 PII·공백이 든 임의 문자열은 URI
    인코딩을 깨끗하게 왕복하지 못한다 — 마스킹 자체는 함수 계층의 계약이므로
    직접 호출로 고정한다(HTTP 왕복 계약은 위의 도구 테스트들이 커버).
    """
    from src.mcp_server.server import get_regulation_document

    missing = get_regulation_document("김철수님 010-1234-5678")
    assert "010-1234-5678" not in missing and "김철수" not in missing


async def test_list_documents_contract(mcp_server_url):
    out = await _call(mcp_server_url, "list_regulation_documents", {})
    assert out["count"] >= 12
    assert all("doc_id" in d and "title" in d for d in out["documents"])
