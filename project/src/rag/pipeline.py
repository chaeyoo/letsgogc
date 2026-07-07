"""RAG 파이프라인 오케스트레이터.

로딩 → 청킹 → 임베딩/인덱싱 을 부팅 시 1회 수행하고,
질의 시 검색(하이브리드+리랭킹) → 근거(context) 조립을 담당한다.
생성(Generation)은 agent 계층에서 이 검색 결과를 받아 수행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config
from .chunker import Chunk, chunk_documents
from .embedder import get_embedder
from .loader import load_documents
from .retriever import HybridRetriever
from .vectorstore import InMemoryVectorStore, Scored


@dataclass
class RetrievedContext:
    """검색 결과를 생성단계로 넘기기 위한 묶음."""
    query: str
    chunks: list[Scored]

    def to_prompt_block(self) -> str:
        """LLM 프롬프트에 넣을 근거 블록(출처 번호 포함)."""
        lines = []
        for i, s in enumerate(self.chunks, 1):
            lines.append(
                f"[근거 {i}] (출처: {s.chunk.title} · {s.chunk.source} · {s.chunk.section})\n"
                f"{s.chunk.text.strip()}"
            )
        return "\n\n".join(lines)

    def citations(self) -> list[dict]:
        """UI/응답용 출처 메타데이터."""
        return [
            {
                "n": i,
                "doc_id": s.chunk.doc_id,
                "title": s.chunk.title,
                "source": s.chunk.source,
                "section": s.chunk.section,
                "score": round(s.score, 4),
            }
            for i, s in enumerate(self.chunks, 1)
        ]


class RagPipeline:
    def __init__(self, reg_dir: Path | None = None) -> None:
        self.reg_dir = reg_dir or config.REG_DIR
        store = InMemoryVectorStore(get_embedder("tfidf"))
        self.retriever = HybridRetriever(store, alpha=config.HYBRID_ALPHA)
        self.n_docs = 0
        self.n_chunks = 0

    def build(self) -> "RagPipeline":
        """인덱스 구축 (부팅 시 1회)."""
        docs = load_documents(self.reg_dir)
        chunks: list[Chunk] = chunk_documents(
            docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        self.retriever.index(chunks)
        self.n_docs, self.n_chunks = len(docs), len(chunks)
        return self

    def retrieve(
        self, query: str, top_k: int | None = None, rerank_n: int | None = None
    ) -> RetrievedContext:
        top_k = top_k or config.RETRIEVE_TOP_K
        rerank_n = rerank_n or config.RERANK_TOP_N
        results = self.retriever.retrieve(query, top_k=top_k, rerank_n=rerank_n)
        return RetrievedContext(query=query, chunks=results)
