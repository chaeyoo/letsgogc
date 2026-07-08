"""리트리버 구성요소 단위 테스트: BM25 · 정규화 · 하이브리드 · 리랭킹 불변식."""
from __future__ import annotations

from src.rag.retriever import BM25Index, _minmax


def test_minmax_normalizes_to_unit_range():
    out = _minmax([1.0, 3.0, 5.0])
    assert out[0] == 0.0 and out[-1] == 1.0
    assert all(0.0 <= v <= 1.0 for v in out)
    # 모두 동일하면 0으로(0분모 방지)
    assert _minmax([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    assert _minmax([]) == []


def test_bm25_ranks_matching_doc_highest():
    idx = BM25Index()
    docs = [
        "품목허가 심사 기간은 신약 120 근무일이다",
        "임상시험 계획 승인 IND 처리기한",
        "화장품 전성분 표시 광고 기준",
    ]
    idx.index(docs)
    scores = idx.scores("신약 품목허가 심사 기간")
    assert scores[0] == max(scores), "질의와 가장 겹치는 문서가 최고점"
    assert scores[0] > 0.0


def test_bm25_idf_penalizes_common_terms():
    idx = BM25Index()
    # '공통' 은 모든 문서에, '희귀' 는 한 문서에만 → 희귀가 더 높은 IDF
    idx.index(["공통 희귀", "공통 단어", "공통 단어"])
    assert idx.idf["희귀"] > idx.idf["공통"]


def test_hybrid_and_rerank_return_within_bounds(pipeline):
    q = "GMP 데이터 완전성 ALCOA 원칙"
    hybrid = pipeline.retriever._hybrid(q, top_k=8)
    assert 0 < len(hybrid) <= 8
    # 하이브리드 점수는 내림차순 정렬
    assert all(hybrid[i].score >= hybrid[i + 1].score for i in range(len(hybrid) - 1))
    reranked = pipeline.retriever.retrieve(q, top_k=8, rerank_n=3)
    assert 0 < len(reranked) <= 3


def test_rerank_fixes_hard_negative(pipeline):
    """어휘가 겹치는 하드네거티브(GMP 실태조사 REG-011)를 물리치고
    ALCOA 데이터 완전성(REG-003)을 최상위로 올려야 한다."""
    q = "GMP 데이터 완전성 ALCOA 원칙은 무엇인가요?"
    top = pipeline.retriever.retrieve(q, top_k=8, rerank_n=1)[0]
    assert top.chunk.doc_id == "REG-003"
