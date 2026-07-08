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

import datetime as _dt
import math
from collections import Counter

from .chunker import Chunk
from .embedder import cosine
from .synonyms import expand_query
from .textutil import tokenize
from .vectorstore import InMemoryVectorStore, Scored


def _is_active(chunk: Chunk, as_of: str, include_superseded: bool) -> bool:
    """버전 인지 필터: 이 청크를 현재 검색 대상으로 삼을지 판정한다.

    - 폐지(superseded)된 문서는 기본 제외(이력 조회 시 include_superseded=True로 포함).
    - as_of(기준일)가 주어지면 그 시점에 아직 시행되지 않은 문서는 제외
      ("2024년 시점 기준 유효 규정" 같은 과거 시점 질의 지원).
    """
    if not include_superseded and chunk.status == "superseded":
        return False
    if as_of and chunk.effective_date:
        try:
            if _dt.date.fromisoformat(chunk.effective_date) > _dt.date.fromisoformat(as_of):
                return False
        except ValueError:
            pass
    return True


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

    def __init__(
        self, store: InMemoryVectorStore, alpha: float = 0.5, rerank_weight: float = 0.9
    ) -> None:
        self.store = store
        self.alpha = alpha              # 벡터 가중(1-alpha 는 BM25 가중)
        self.rerank_weight = rerank_weight  # 리랭커 신호 vs 1차 점수 prior 결합 비율
        self.bm25 = BM25Index()

    def index(self, chunks: list[Chunk]) -> None:
        self.store.index(chunks)
        self.bm25.index([c.text for c in chunks])

    def _candidate_indices(self, as_of: str, include_superseded: bool) -> list[int]:
        """버전 인지 필터를 통과한 청크 인덱스만 반환(세 검색 모드 공통 후보군)."""
        return [
            i
            for i, c in enumerate(self.store.chunks)
            if _is_active(c, as_of, include_superseded)
        ]

    # ---- 벡터 단독 검색(버전 필터 공유; eval 비교·폴백용) ----
    def vector_search(
        self, query: str, top_k: int, as_of: str = "", include_superseded: bool = False
    ) -> list[Scored]:
        qv = self.store.embedder.embed(query)
        idxs = self._candidate_indices(as_of, include_superseded)
        scored = [
            Scored(chunk=self.store.chunks[i], score=cosine(qv, self.store.vectors[i]))
            for i in idxs
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    # ---- 1단계: 하이브리드 검색 ----
    def _hybrid(
        self, query: str, top_k: int, as_of: str = "", include_superseded: bool = False
    ) -> list[Scored]:
        qv = self.store.embedder.embed(query)
        bm_scores_all = self.bm25.scores(query)
        idxs = self._candidate_indices(as_of, include_superseded)
        # 정규화는 '후보군 안에서' 수행해야 스케일이 왜곡되지 않는다.
        vec_scores = [cosine(qv, self.store.vectors[i]) for i in idxs]
        bm_scores = [bm_scores_all[i] for i in idxs]
        vn, bn = _minmax(vec_scores), _minmax(bm_scores)
        combined = [
            Scored(chunk=self.store.chunks[i], score=self.alpha * vn[j] + (1 - self.alpha) * bn[j])
            for j, i in enumerate(idxs)
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

    def retrieve(
        self,
        query: str,
        top_k: int,
        rerank_n: int,
        as_of: str = "",
        include_superseded: bool = False,
        expand: bool = True,
    ) -> list[Scored]:
        """최종 검색: (질의 확장) → 하이브리드 top_k → 리랭킹 → 상위 rerank_n 반환.

        as_of / include_superseded 로 버전 인지 검색을 제어한다.

        질의 확장은 '1단계 회수'에만 적용한다("부작용"→"이상사례" 같은
        어휘 불일치를 메워 recall 확보). 2단계 리랭킹은 원 질의로 재점수해
        확장어가 정밀도 신호를 희석하지 않게 한다(회수/정밀 역할 분리).
        """
        q1 = expand_query(query) if expand else query
        first = self._hybrid(q1, top_k, as_of, include_superseded)
        if not first:
            return []
        # 1차 하이브리드 점수를 prior 로 블렌딩(순수 재정렬은 쉬운 질의를 오히려 떨어뜨림).
        # 실무 Cross-Encoder 리랭커도 first-stage 점수와 결합해 안정화하는 관행을 반영.
        first_scores = _minmax([s.score for s in first])
        reranked = [
            Scored(
                chunk=s.chunk,
                score=self.rerank_weight * self._rerank_score(query, s.chunk)
                + (1 - self.rerank_weight) * first_scores[i],
            )
            for i, s in enumerate(first)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[:rerank_n]
