"""인메모리 벡터 스토어 (Vector Store).

청크와 그 임베딩을 보관하고 코사인 유사도로 top-k 검색을 제공한다.
실무의 Vector DB(pgvector/Qdrant/Chroma) 자리에 해당하는 경량 대체물.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embedder import EmbeddingProvider, SparseVec, cosine


@dataclass
class Scored:
    """검색 결과 1건 (청크 + 점수)."""
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[SparseVec] = []

    def index(self, chunks: list[Chunk]) -> None:
        """청크 코퍼스로 임베더를 학습시키고 전체를 벡터화해 저장."""
        self.chunks = chunks
        self.embedder.fit([c.text for c in chunks])
        self.vectors = [self.embedder.embed(c.text) for c in chunks]

    def search(self, query: str, top_k: int) -> list[Scored]:
        """의미 검색(코사인 유사도) 상위 top_k."""
        qv = self.embedder.embed(query)
        scored = [
            Scored(chunk=c, score=cosine(qv, v))
            for c, v in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
