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


def test_colloquial_query_resolved_by_expansion(pipeline):
    """구어 질의('부작용이 심각', '섞이는 것')는 원 질의 토큰이 문서에 거의
    없어도, 동의어 확장(심각→중대한, 섞이→교차오염)으로 정답 문서를 찾아야
    한다 — eval 오류 분석에서 사전을 보강한 회귀 가드."""
    top = pipeline.retriever.retrieve(
        "부작용이 심각하게 나타났을 때 당국에 얼마나 빨리 알려야 하나요?",
        top_k=8, rerank_n=1,
    )[0]
    assert top.chunk.doc_id == "REG-005"

    top = pipeline.retriever.retrieve(
        "약을 만드는 시설에서 서로 다른 제품이 섞이는 것을 막는 기본 원칙은?",
        top_k=8, rerank_n=1,
    )[0]
    assert top.chunk.doc_id == "REG-003"


def test_aux_terms_keep_reranker_robust_without_prior(pipeline):
    """확장 토큰 보조신호(aux)의 존재 이유: 1차 prior 를 끄면(rerank_weight=1.0)
    완전 어휘 불일치 질의에서 aux 없는 리랭커는 판별력을 잃는다."""
    r = pipeline.retriever
    q = "부작용이 심각하게 나타났을 때 당국에 얼마나 빨리 알려야 하나요?"
    saved = r.rerank_weight
    try:
        r.rerank_weight = 1.0
        with_aux = r.retrieve(q, top_k=8, rerank_n=1, aux_in_rerank=True)[0]
        assert with_aux.chunk.doc_id == "REG-005", "aux 켜면 prior 없이도 정답"
    finally:
        r.rerank_weight = saved


def test_rerank_title_field_signal(pipeline):
    """제목 정합(title) 신호: 같은 본문 매칭이라도 문서 제목이 질의와 겹치는
    청크가 더 높은 점수를 받아야 한다(BM25F 감각의 필드 분리)."""
    from src.rag.chunker import Chunk

    def mk(title: str) -> Chunk:
        return Chunk(
            chunk_id="T-1", doc_id="T", source="t.md", title=title,
            section="1. 개요", text="의약품 용기 기재사항에 대한 내용",
        )

    r = pipeline.retriever
    q = "의약품 용기 기재사항은?"
    s_match = r._rerank_score(q, mk("의약품 표시기재 기준"))
    s_mismatch = r._rerank_score(q, mk("화장품 표시광고 규정"))
    assert s_match > s_mismatch
