"""로더·청커·검색 파이프라인 기본 동작(pytest).

실행:  pytest            (권장)
       python -m tests.test_rag   (독립 실행도 유지)
"""
from __future__ import annotations

from src import config
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.pipeline import RagPipeline


def test_loader_and_chunker():
    docs = load_documents(config.REG_DIR)
    assert len(docs) >= 12, "확장된 규제문서(하드네거티브 포함) 12건 이상 로드"
    assert all(d.title and d.text for d in docs), "모든 문서에 제목·본문 존재"
    # frontmatter 메타(버전/시행일)가 파싱되는지
    assert any(d.metadata.get("version") for d in docs), "버전 메타 파싱"
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    assert len(chunks) > len(docs), "청킹으로 청크 수가 문서 수보다 많아야 함"
    # 버전 필드가 청크로 전파되는지
    assert all(c.title and c.text for c in chunks)
    assert any(c.status == "superseded" for c in chunks), "폐지본 청크가 존재(버전 테스트용)"


def test_retrieval_relevance(pipeline: RagPipeline):
    # 자연어 질문형 — 도메인 정답 문서가 1순위로 나와야 한다.
    cases = [
        ("신약 품목허가 심사는 며칠 이내에 처리되나요?", "REG-001"),
        ("중대한 이상사례는 며칠 이내에 보고해야 하나요?", "REG-005"),
        ("GMP 데이터 완전성 ALCOA 원칙은 무엇인가요?", "REG-003"),
        ("변경신고는 며칠 이내에 수리 통보되나요?", "REG-002"),
        ("의료기기 4등급 신개발 제품의 품목허가 심사는 며칠 이내인가요?", "REG-007"),
    ]
    for query, gold in cases:
        ctx = pipeline.retrieve(query)
        assert ctx.chunks, f"'{query}' 결과 없음"
        assert ctx.chunks[0].chunk.doc_id == gold, (
            f"'{query}' → 기대 {gold}, 실제 {ctx.chunks[0].chunk.doc_id}"
        )
        assert ctx.citations(), "출처가 비어있지 않아야 함"


def test_citations_carry_version(pipeline: RagPipeline):
    ctx = pipeline.retrieve("신약 품목허가 심사는 며칠 이내에 처리되나요?")
    cite = ctx.citations()[0]
    assert cite["doc_id"] and cite["title"]
    assert "version" in cite and "effective_date" in cite


if __name__ == "__main__":
    # 독립 실행(하위호환): fixture 없이 파이프라인 직접 구축
    pipe = RagPipeline().build()
    test_loader_and_chunker()
    test_retrieval_relevance(pipe)
    test_citations_carry_version(pipe)
    print("스모크 테스트 통과 ✅ (자세한 검증은 pytest 로 실행)")
