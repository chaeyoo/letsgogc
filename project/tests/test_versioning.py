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


def test_as_of_includes_then_active_superseded(pipeline):
    """as_of 시점 조회의 핵심 의미론: '그 시점의 현행'을 반환한다.

    REG-013(구판, 2021-01-01 시행)은 지금은 폐지본이지만 2025-01-01 시점에는
    현행이었다(후속본 REG-005는 2025-04-01 시행). '폐지본 기본 제외'를 as_of에도
    그대로 적용하면 신·구판이 모두 걸러져 시점 조회가 0건이 된다 — 개정 이력이
    있는(=시점 조회가 필요한) 규정에서만 무너지는 사각지대였다."""
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=8, rerank_n=3, as_of="2025-01-01"
    )
    assert res, "시점 조회 결과가 0건 — 당시 현행 버전이 필터에서 걸러졌다"
    assert any(s.chunk.doc_id == "REG-013" for s in res), "당시 현행이던 구판이 포함되어야 함"
    assert all(s.chunk.doc_id != "REG-005" for s in res), "그 시점에 시행 전인 현행본은 제외"


def test_as_of_after_revision_returns_current(pipeline):
    # 후속본 시행(2025-04-01) 이후 시점에는 구판이 다시 제외된다.
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=8, rerank_n=8, as_of="2025-06-01"
    )
    assert any(s.chunk.doc_id == "REG-005" for s in res)
    assert all(s.chunk.doc_id != "REG-013" for s in res)


def test_mcp_search_exposes_version_fields():
    out = search_regulations("신약 품목허가 심사 기간", top_n=3)
    r = out["results"][0]
    assert "version" in r and "effective_date" in r and "status" in r


def test_malformed_effective_date_fails_closed_under_as_of():
    """시행일을 해석할 수 없는 문서는 시점(as_of) 조회에서 제외된다(fail-closed)
    — 예외 전파로 검색 전체가 죽지도, 무필터로 통과해 오답에 섞이지도 않는다.
    (데이터 결함 자체는 preflight 코퍼스 무결성 검사가 파일명과 함께 보고한다)"""
    from src.rag.chunker import Chunk
    from src.rag.embedder import get_embedder
    from src.rag.retriever import HybridRetriever
    from src.rag.vectorstore import InMemoryVectorStore

    chunks = [
        Chunk(chunk_id="ok::s0::w0", doc_id="OK", source="ok.md", title="정상 규정",
              section="본문", text="[정상 규정 > 본문]\n심사 기간은 60일이다.",
              effective_date="2024-01-01"),
        Chunk(chunk_id="bad::s0::w0", doc_id="BAD", source="bad.md", title="결함 규정",
              section="본문", text="[결함 규정 > 본문]\n심사 기간은 999일이다.",
              effective_date="2025.08.01"),  # 형식 오류(점 표기)
    ]
    r = HybridRetriever(InMemoryVectorStore(get_embedder("tfidf")))
    r.index(chunks)
    res = r.retrieve("심사 기간", top_k=5, rerank_n=5, as_of="2026-01-01")  # 크래시 없어야 함
    ids = {s.chunk.doc_id for s in res}
    assert "OK" in ids and "BAD" not in ids
    # as_of 없는 기본 검색에서는 (버전 필터가 개입하지 않으므로) 여전히 검색된다
    res_all = r.retrieve("심사 기간", top_k=5, rerank_n=5)
    assert "BAD" in {s.chunk.doc_id for s in res_all}
