"""버전 인지 검색 테스트: 폐지본 제외 · 이력 조회 · 과거 시점(as_of)."""
from __future__ import annotations

from src.mcp_server.server import search_regulations


def test_superseded_excluded_by_default(pipeline):
    # 폐지 구판(REG-013)은 기본 검색에서 제외되어야 한다.
    res = pipeline.retriever.retrieve("이상사례 보고 기한", top_k=8, rerank_n=8)
    assert all(s.chunk.status != "superseded" for s in res)
    assert all(s.chunk.doc_id != "REG-013" for s in res)


def test_include_superseded_surfaces_old_version(pipeline):
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=12, rerank_n=12, include_superseded=True
    )
    assert any(s.chunk.doc_id == "REG-013" for s in res), "이력 조회 시 폐지본 노출"


def test_current_answer_prefers_effective_doc(pipeline):
    # 현행(REG-005, 15일)이 폐지본(REG-013, 30일)보다 우선.
    top = pipeline.retriever.retrieve("중대한 이상사례 보고 기한", top_k=8, rerank_n=1)[0]
    assert top.chunk.doc_id == "REG-005"
    assert top.chunk.status == "active"


def test_as_of_excludes_not_yet_effective(pipeline):
    # 2024-06-01 시점에는 2025 시행 문서들이 아직 유효하지 않다.
    res = pipeline.retriever.retrieve(
        "품목허가 심사 기간", top_k=20, rerank_n=20, as_of="2024-06-01"
    )
    for s in res:
        if s.chunk.effective_date:
            assert s.chunk.effective_date <= "2024-06-01"


def test_mcp_search_exposes_version_fields():
    out = search_regulations("신약 품목허가 심사 기간", top_n=3)
    r = out["results"][0]
    assert "version" in r and "effective_date" in r and "status" in r
