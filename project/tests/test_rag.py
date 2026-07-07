"""파이프라인·MCP·에이전트 스모크 테스트 (외부 의존성 없이 실행).

실행:  python -m tests.test_rag
성공 시 종료코드 0, 실패 시 AssertionError.
"""
from __future__ import annotations

import asyncio

from src.agent.agent import RaAgent
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.pipeline import RagPipeline
from src import config


def test_loader_and_chunker():
    docs = load_documents(config.REG_DIR)
    assert len(docs) >= 6, "규제문서 6건 이상 로드되어야 함"
    assert all(d.title and d.text for d in docs), "모든 문서에 제목·본문 존재"
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    assert len(chunks) > len(docs), "청킹으로 청크 수가 문서 수보다 많아야 함"
    print(f"  ✓ loader/chunker: docs={len(docs)}, chunks={len(chunks)}")


def test_retrieval_relevance():
    pipe = RagPipeline().build()
    cases = [
        ("신약 품목허가 심사 기간", "REG-001"),
        ("중대한 이상사례 보고 기한", "REG-005"),
        ("GMP 데이터 완전성 ALCOA", "REG-003"),
        ("변경허가 처리기한", "REG-002"),
    ]
    for query, gold in cases:
        ctx = pipe.retrieve(query)
        top_doc = ctx.chunks[0].chunk.doc_id
        assert top_doc == gold, f"'{query}' → 기대 {gold}, 실제 {top_doc}"
        assert ctx.citations(), "출처가 비어있지 않아야 함"
    print(f"  ✓ retrieval: {len(cases)}개 질의 모두 정답 문서 1순위")


def test_agent_offline_routing():
    agent = RaAgent()

    async def run():
        r1 = await agent.chat("품목허가 심사 기간은?")
        assert r1.tool_calls[0].name == "search_regulations"
        r2 = await agent.chat("이번 주 마감 임박한 업무 알려줘")
        assert r2.tool_calls[0].name == "get_ra_deadlines"
        r3 = await agent.chat("변경허가 체크리스트 알려줘")
        assert r3.tool_calls[0].name == "get_submission_checklist"
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(run())
    assert all(r.answer for r in (r1, r2, r3)), "모든 응답에 답변 존재"
    print("  ✓ agent: 3개 인텐트(검색/마감일/체크리스트) 라우팅 정상")


def main():
    print("RAG/MCP/Agent 스모크 테스트")
    print("-" * 40)
    test_loader_and_chunker()
    test_retrieval_relevance()
    test_agent_offline_routing()
    print("-" * 40)
    print("모든 테스트 통과 ✅")


if __name__ == "__main__":
    main()
