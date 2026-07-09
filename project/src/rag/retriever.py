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
        self,
        store: InMemoryVectorStore,
        alpha: float = 0.5,
        rerank_weight: float = 0.9,
        idf_power: float = 0.5,
    ) -> None:
        self.store = store
        self.alpha = alpha              # 벡터 가중(1-alpha 는 BM25 가중)
        self.rerank_weight = rerank_weight  # 리랭커 신호 vs 1차 점수 prior 결합 비율
        self.idf_power = idf_power      # 리랭커 토큰 가중 = idf^p (0=균등, 1=IDF 그대로)
        self.bm25 = BM25Index()
        # 리랭커용 사전 계산 캐시 (질의마다 청크를 재토크나이징하면 지연이 4배로 는다)
        self._chunk_tf: list[Counter[str]] = []
        self._sec_tokens: list[set[str]] = []
        self._title_w: list[dict[str, float]] = []   # 제목 토큰별 가중(질의 무관 → 사전 계산)
        self._title_total: list[float] = []
        self._chunk_ix: dict[int, int] = {}
        self._max_idf = 1.0

    def index(self, chunks: list[Chunk]) -> None:
        self.store.index(chunks)
        self.bm25.index([c.text for c in chunks])
        self._chunk_tf = list(self.bm25.tf)  # BM25 인덱스와 동일 토크나이즈 재사용
        self._max_idf = max(self.bm25.idf.values()) if self.bm25.idf else 1.0
        # 섹션 신호는 '섹션 제목만' 쓴다 — 문서 제목은 title 신호가 따로 담당
        # (섞으면 같은 문서의 모든 청크가 제목 토큰을 공유해 섹션 간 변별이 죽는다)
        self._sec_tokens = [set(tokenize(c.section)) for c in chunks]
        self._title_w = [
            {t: self._token_weight(t) for t in set(tokenize(c.title))} for c in chunks
        ]
        self._title_total = [sum(w.values()) for w in self._title_w]
        self._chunk_ix = {id(c): i for i, c in enumerate(chunks)}

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
    # 리랭커 컴포넌트 가중치. eval 스윕에서 이 데이터셋 지표는 title 0~0.15,
    # idf_power 0~1에 둔감했다(하드네거티브 실패 1건은 어떤 조합으로도 안 뒤집힘).
    # 그래서 '수치가 좋아서'가 아니라 신호의 역할 분리(본문/구문/섹션/제목)를
    # 기준으로 가중을 고정했다 — 코퍼스가 커져 신호가 분화되면 sweep 으로 재보정.
    RERANK_COMPONENT_WEIGHTS: dict[str, float] = {
        "coverage": 0.55,   # 본문: 질의 토큰 커버리지(idf^p 가중)
        "phrase": 0.20,     # 구문: 질의 원문 정확 매칭
        "section": 0.15,    # 섹션: 정답 '섹션' 선택 신호
        "title": 0.10,      # 제목: 문서 정합(BM25F 감각의 필드 신호)
    }

    def _token_weight(self, term: str) -> float:
        """리랭커용 토큰 가중 = idf^p (BM25 인덱스의 IDF 재사용).

        p=0 이면 균등(v1과 동일), p=1 이면 IDF 그대로. 매칭 밀도(tf 포화)
        신호와 결합한 초기 실험에서 p=1은 '의약품'처럼 흔하지만 문서 구분에
        필수인 토큰을 지나치게 죽여 하드네거티브를 악화시켰다(0.846→0.769).
        밀도 신호 제거 후에는 p에 둔감해졌지만, 큰 코퍼스에서 판별 토큰
        우대가 필요해질 것에 대비해 완만한 중간값 0.5를 기본으로 남겼다.
        코퍼스에 없는 토큰(OOV)은 가장 희귀한 수준으로 취급한다.
        """
        if not self.bm25.idf:
            return 1.0
        idf = self.bm25.idf.get(term) or self._max_idf
        return idf ** self.idf_power

    def _rerank_score(self, query: str, chunk: Chunk, aux_terms: set[str] | None = None) -> float:
        """질의-청크 관련도 재점수 (Cross-Encoder 근사).

        네 가지 신호의 가중합:
          (1) coverage — 본문이 질의 토큰을 얼마나 덮는가(idf^p 가중)
          (2) phrase   — 질의 원문이 본문에 통째로 등장하는가
          (3) section  — 질의 토큰이 '섹션 제목'에 있는가(같은 문서 안에서
              정답 섹션을 고르는 신호 — 문서 제목은 섞지 않는다)
          (4) title    — 문서 제목 정합: 제목 토큰 중 질의에 등장하는 비율.
              하드네거티브 문서(예: 화장품 규정 속 '의약품 표시기재와의 차이'
              섹션)는 본문이 질의 어휘를 많이 공유해 coverage 로는 못 거르므로,
              문서 제목('화장품 표시·광고')이 질의와 겹치지 않는 것을
              반증 신호로 쓴다 — BM25F 처럼 필드(제목)를 분리한 설계.

        aux_terms: 질의확장으로 '추가된' 토큰 집합(선택). 원 질의 토큰의
        절반 가중으로만 반영한다. 완전 어휘 불일치 질의("부작용이 심각…"은
        원 질의 토큰이 규정 문서에 하나도 없다)에서 리랭커가 판별력을 잃는
        것을 막는 안전망 — rerank_weight=1.0(1차 prior 없음) ablation 에서
        aux 유무가 Hit@1 0.967 vs 0.867 차이를 만든다.
        """
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        q_set = set(q_terms)

        ix = self._chunk_ix.get(id(chunk))
        if ix is None:  # 인덱스 밖 청크(단위 테스트 등) — 캐시 없이 즉석 계산
            c_set = set(tokenize(chunk.text))
            sec_terms = set(tokenize(chunk.section))
            title_w = {t: self._token_weight(t) for t in set(tokenize(chunk.title))}
            title_total = sum(title_w.values())
        else:
            c_set = set(self._chunk_tf[ix])
            sec_terms = self._sec_tokens[ix]
            title_w = self._title_w[ix]
            title_total = self._title_total[ix]

        # 토큰 가중치: 원 질의는 idf^p, 확장 토큰은 그 절반(보조 신호)
        weights = {t: self._token_weight(t) for t in q_set}
        for t in (aux_terms or set()) - q_set:
            weights[t] = 0.5 * self._token_weight(t)
        total_w = sum(weights.values()) or 1.0

        coverage = sum(w for t, w in weights.items() if t in c_set) / total_w
        phrase = 1.0 if query.strip() and query.strip().lower() in chunk.text.lower() else 0.0
        section_hit = sum(w for t, w in weights.items() if t in sec_terms) / total_w
        # 제목 정합은 '제목 쪽 기준' 비율(제목 토큰이 질의로 얼마나 설명되는가)
        title_match = (
            sum(w for t, w in title_w.items() if t in weights) / title_total
            if title_total
            else 0.0
        )

        w = self.RERANK_COMPONENT_WEIGHTS
        return (
            w["coverage"] * coverage
            + w["phrase"] * phrase
            + w["section"] * section_hit
            + w["title"] * title_match
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
        rerank_n: int,
        as_of: str = "",
        include_superseded: bool = False,
        expand: bool = True,
        aux_in_rerank: bool = True,
    ) -> list[Scored]:
        """최종 검색: (질의 확장) → 하이브리드 top_k → 리랭킹 → 상위 rerank_n 반환.

        as_of / include_superseded 로 버전 인지 검색을 제어한다.

        질의 확장은 '1단계 회수'에 전 가중으로 적용하고("부작용"→"이상사례"
        같은 어휘 불일치를 메워 recall 확보), 2단계 리랭킹은 원 질의 토큰을
        기준으로 재점수하되 확장 토큰을 절반 가중의 보조 신호로만 쓴다.
        원 질의만으로 재점수하면 완전 어휘 불일치 질의에서 리랭커가 판별력을
        잃는다(rerank_weight=1.0 ablation: aux 유무 = Hit@1 0.967 vs 0.867).
        """
        q1 = expand_query(query) if expand else query
        first = self._hybrid(q1, top_k, as_of, include_superseded)
        if not first:
            return []
        # 확장으로 '추가된' 토큰만 보조 신호로 리랭커에 전달
        # (aux_in_rerank=False 는 eval/sweep.py 의 ablation 용 스위치)
        aux = (
            (set(tokenize(q1)) - set(tokenize(query)))
            if (q1 != query and aux_in_rerank)
            else None
        )
        # 1차 하이브리드 점수를 prior 로 블렌딩(순수 재정렬은 쉬운 질의를 오히려 떨어뜨림).
        # 실무 Cross-Encoder 리랭커도 first-stage 점수와 결합해 안정화하는 관행을 반영.
        first_scores = _minmax([s.score for s in first])
        reranked = [
            Scored(
                chunk=s.chunk,
                score=self.rerank_weight * self._rerank_score(query, s.chunk, aux_terms=aux)
                + (1 - self.rerank_weight) * first_scores[i],
            )
            for i, s in enumerate(first)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[:rerank_n]
