"""임베더 (Embedder) — 텍스트를 벡터로.

핵심 설계: **provider 교체 가능(pluggable)** 구조.
- 오프라인 기본값: TF-IDF 희소 벡터(외부 의존성·API 키 불필요). 데모가 항상 돌아간다.
- 실무 확장 지점: `EmbeddingProvider` 를 구현하면 상용 임베딩 API
  (예: OpenAI/Cohere/Voyage 임베딩)로 그대로 교체 가능하다.

벡터는 {term: weight} 형태의 희소 dict 로 표현하고, 코사인 유사도로 비교한다.
(numpy 등 무거운 의존성 없이 순수 파이썬으로 구현)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from .textutil import tokenize

SparseVec = dict[str, float]


class EmbeddingProvider(Protocol):
    """임베딩 제공자 인터페이스. 실무에선 이 자리에 상용 임베딩 API를 끼운다."""

    def fit(self, corpus: list[str]) -> None: ...
    def embed(self, text: str) -> SparseVec: ...


class TfidfEmbedder:
    """오프라인 TF-IDF 임베더 (기본 provider)."""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._n_docs = 0
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """코퍼스로 IDF(역문서빈도)를 학습한다."""
        self._n_docs = len(corpus)
        df: Counter[str] = Counter()
        for text in corpus:
            for term in set(tokenize(text)):
                df[term] += 1
        # smooth IDF
        self._idf = {
            term: math.log((self._n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }
        self._fitted = True

    def embed(self, text: str) -> SparseVec:
        """텍스트를 L2 정규화된 TF-IDF 희소 벡터로 변환."""
        if not self._fitted:
            raise RuntimeError("embed() 전에 fit()을 먼저 호출해야 한다.")
        tf = Counter(tokenize(text))
        if not tf:
            return {}
        max_tf = max(tf.values())
        vec: SparseVec = {}
        for term, freq in tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue  # 학습 코퍼스에 없던 미지의 단어는 무시
            vec[term] = (0.5 + 0.5 * freq / max_tf) * idf  # 증강 TF × IDF
        # L2 정규화
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}


def cosine(a: SparseVec, b: SparseVec) -> float:
    """두 희소 벡터의 코사인 유사도. (이미 L2 정규화됐다면 내적과 동일)"""
    if not a or not b:
        return 0.0
    # 더 작은 쪽을 순회
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values())) or 1.0
    nb = math.sqrt(sum(w * w for w in b.values())) or 1.0
    return dot / (na * nb)


def get_embedder(kind: str = "tfidf") -> EmbeddingProvider:
    """임베더 팩토리. 향후 'openai' 등 추가 시 여기서 분기."""
    if kind == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"알 수 없는 임베더: {kind}")
