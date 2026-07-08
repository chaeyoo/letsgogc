"""임베더 provider 테스트: TF-IDF · 해싱 · 코사인."""
from __future__ import annotations

import math

from src.rag.embedder import HashingEmbedder, TfidfEmbedder, cosine, get_embedder


def test_tfidf_embeds_unit_norm():
    emb = TfidfEmbedder()
    emb.fit(["품목허가 심사 기간", "이상사례 보고 기한"])
    v = emb.embed("품목허가 심사")
    assert v, "임베딩이 비어있지 않아야 함"
    norm = math.sqrt(sum(w * w for w in v.values()))
    assert abs(norm - 1.0) < 1e-6, "L2 정규화되어야 함"


def test_tfidf_requires_fit():
    emb = TfidfEmbedder()
    try:
        emb.embed("x")
        assert False, "fit 전 embed 는 예외여야 함"
    except RuntimeError:
        pass


def test_hashing_is_deterministic_and_handles_oov():
    emb = HashingEmbedder(n_buckets=256)
    emb.fit(["품목허가 심사", "이상사례 보고"])
    a = emb.embed("완전히 처음 보는 미등록 단어들")   # OOV 여도 예외 없이 벡터 생성
    b = emb.embed("완전히 처음 보는 미등록 단어들")
    assert a == b, "동일 입력은 동일 벡터(재현성)"
    assert a, "OOV 입력도 해싱으로 벡터화 가능"


def test_cosine_bounds_and_self_similarity():
    emb = TfidfEmbedder()
    emb.fit(["품목허가 심사 기간 신약", "화장품 전성분 표시"])
    v = emb.embed("품목허가 심사 기간")
    assert abs(cosine(v, v) - 1.0) < 1e-6, "자기 자신과의 코사인은 1"
    assert cosine(v, {}) == 0.0, "빈 벡터와는 0"


def test_factory_returns_distinct_providers():
    assert isinstance(get_embedder("tfidf"), TfidfEmbedder)
    assert isinstance(get_embedder("hashing"), HashingEmbedder)
    try:
        get_embedder("nope")
        assert False
    except ValueError:
        pass
