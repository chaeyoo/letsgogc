"""하이브리드 리트리버 + 리랭커 (RAG '최적화'의 핵심).

2단계 검색으로 정밀도를 끌어올린다.
  1단계 (많이 회수):  하이브리드 검색 = 벡터(TF-IDF 코사인) + 키워드(BM25)
                     → 의미 유사(동의어)와 정확 용어(고유명사/코드) 모두 커버.
  2단계 (정밀 재정렬): 리랭커가 질의-청크 관련도를 다시 점수화해 상위 N개만 남김.
                     → Bi-Encoder(빠름) 로 넓게, Cross-Encoder 감각(정밀) 로 좁게.

리랭커는 오프라인에서 (질의 토큰 커버리지 + 정확 구문 매칭 + 섹션제목 가중)
으로 근사한다. 실무에선 이 자리에 Cross-Encoder 리랭커나 LLM 리랭커를 끼운다.
"""
from __future__ import annotations

import math
from collections import Counter

from .chunker import Chunk
from .embedder import cosine
from .textutil import tokenize
from .vectorstore import InMemoryVectorStore, Scored


class BM25Index:
    """BM25 키워드 검색 인덱스 (Okapi BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.tf: list[Counter[str]] = []

    def index(self, texts: list[str]) -> None:
        self.docs_tokens = [tokenize(t) for t in texts]
        self.tf = [Counter(toks) for toks in self.docs_tokens]
        self.doc_len = [len(toks) for toks in self.docs_tokens]
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for toks in self.docs_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out: list[float] = []
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1)
            out.append(s)
        return out


def _minmax(values: list[float]) -> list[float]:
    """0~1 정규화 (하이브리드 결합 전 스케일 정렬)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    """벡터 + BM25 하이브리드 1차 검색 후 리랭킹."""

    def __init__(self, store: InMemoryVectorStore, alpha: float = 0.5) -> None:
        self.store = store
        self.alpha = alpha              # 벡터 가중(1-alpha 는 BM25 가중)
        self.bm25 = BM25Index()

    def index(self, chunks: list[Chunk]) -> None:
        self.store.index(chunks)
        self.bm25.index([c.text for c in chunks])

    # ---- 1단계: 하이브리드 검색 ----
    def _hybrid(self, query: str, top_k: int) -> list[Scored]:
        qv = self.store.embedder.embed(query)
        vec_scores = [cosine(qv, v) for v in self.store.vectors]
        bm_scores = self.bm25.scores(query)
        vn, bn = _minmax(vec_scores), _minmax(bm_scores)
        combined = [
            Scored(chunk=c, score=self.alpha * vn[i] + (1 - self.alpha) * bn[i])
            for i, c in enumerate(self.store.chunks)
        ]
        combined.sort(key=lambda s: s.score, reverse=True)
        return combined[:top_k]

    # ---- 2단계: 리랭킹 ----
    def _rerank_score(self, query: str, chunk: Chunk) -> float:
        """질의-청크 관련도 재점수 (Cross-Encoder 근사)."""
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        q_set = set(q_terms)
        c_terms = tokenize(chunk.text)
        c_set = set(c_terms)

        # (1) 질의 토큰 커버리지: 질의 단어 중 몇 %가 청크에 등장하나
        coverage = len(q_set & c_set) / len(q_set)

        # (2) 정확 구문 매칭: 질의 원문이 청크에 통째로 등장하면 강한 신호
        phrase = 1.0 if query.strip() and query.strip().lower() in chunk.text.lower() else 0.0

        # (3) 섹션/제목 매칭 가중: 질의어가 섹션 제목에 있으면 가중
        sec_terms = set(tokenize(chunk.section + " " + chunk.title))
        section_hit = len(q_set & sec_terms) / len(q_set)

        return 0.6 * coverage + 0.25 * phrase + 0.15 * section_hit

    def retrieve(self, query: str, top_k: int, rerank_n: int) -> list[Scored]:
        """최종 검색: 하이브리드 top_k → 리랭킹 → 상위 rerank_n 반환."""
        first = self._hybrid(query, top_k)
        reranked = [
            Scored(chunk=s.chunk, score=self._rerank_score(query, s.chunk))
            for s in first
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        # 리랭킹 점수가 모두 0이면(질의어 미스) 하이브리드 순서를 폴백 유지
        if reranked and reranked[0].score == 0.0:
            return first[:rerank_n]
        return reranked[:rerank_n]
