"""RA(규제업무) 어시스턴트를 위한 MCP 서버.

GC 'Hey.GC 2.0'가 MCP 구조로 사내 시스템을 통합하는 것과 동일한 패턴으로,
제약 RA 담당자의 업무 시스템(규제문서·마감일·체크리스트)을 MCP 도구로 노출한다.

노출하는 MCP primitive:
  - Tools:     search_regulations / get_ra_deadlines / get_submission_checklist
  - Resources: regulation://{doc_id}  (규제문서 원문 조회)

이 서버는 두 가지로 사용된다.
  (1) 독립 실행:  python -m src.mcp_server.server   (stdio transport, Claude Desktop/Cursor 연결)
  (2) 인메모리:   에이전트가 fastmcp.Client(mcp) 로 직접 연결 (본 데모 기본값)

도구의 '이름·설명(docstring)·타입힌트'가 곧 LLM에게 주는 사용설명서다.
FDE의 실력은 여기서 드러난다 — 에이전트가 언제 어떤 도구를 쓸지 이 설명으로 판단한다.
"""
from __future__ import annotations

import datetime as _dt
import json

from fastmcp import FastMCP

from .. import config
from ..rag.pipeline import RagPipeline

mcp = FastMCP("RA-Assistant")

# 규제문서 RAG 인덱스는 서버 로드시 1회 구축 (무거운 초기화를 캐시)
_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline().build()
    return _pipeline


def _load_ra_tasks() -> dict:
    return json.loads(config.RA_TASKS_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
def search_regulations(
    query: str, top_n: int = 3, as_of: str = "", include_superseded: bool = False
) -> dict:
    """제약 규제문서(품목허가·변경허가·GMP·라벨링·약물감시·임상)를 의미 기반으로 검색한다.

    허가 절차, 심사 기간, 보고 기한, 제출자료, GMP 요건 등 '규정이 어떻게 되어 있는지'를
    묻는 질문에 사용한다. 하이브리드 검색 + 리랭킹으로 가장 관련 높은 근거 문단을 반환하며,
    각 결과에는 출처(문서명·섹션·버전·시행일)가 포함되어 답변의 근거 추적이 가능하다.

    버전 인지: 폐지(superseded)된 구판은 기본 제외한다(현행 규정만 답하도록).
    과거 시점의 규정을 조회하려면 as_of(YYYY-MM-DD)를, 폐지본 이력까지 보려면
    include_superseded=True 를 사용한다.

    Args:
        query: 검색할 질문/키워드 (예: "중대한 이상사례 보고 기한").
        top_n: 반환할 근거 문단 수 (기본 3).
        as_of: 기준일(YYYY-MM-DD). 지정 시 그 시점에 시행 중이던 규정만 검색.
        include_superseded: True면 폐지된 구판도 포함(이력 조회용).

    Returns:
        {"results": [{"text","title","source","section","version","effective_date","score"}...]} 형태.
    """
    ctx = _get_pipeline().retrieve(
        query,
        rerank_n=max(1, min(top_n, 5)),
        as_of=as_of,
        include_superseded=include_superseded,
    )
    return {
        "query": query,
        "as_of": as_of or None,
        "results": [
            {
                "text": s.chunk.text.strip(),
                "doc_id": s.chunk.doc_id,
                "title": s.chunk.title,
                "source": s.chunk.source,
                "section": s.chunk.section,
                "version": s.chunk.version,
                "effective_date": s.chunk.effective_date,
                "status": s.chunk.status,
                "score": round(s.score, 4),
            }
            for s in ctx.chunks
        ],
    }


@mcp.tool
def get_ra_deadlines(within_days: int = 30, task_type: str = "") -> dict:
    """RA 담당자의 규제 업무 마감일/기한을 조회한다.

    "이번 주 마감", "곧 처리해야 할 규제 업무", "지연 위험 항목" 등
    일정·기한 관련 질문에 사용한다. 오늘 기준 within_days 이내 항목을 마감일 순으로 반환한다.

    Args:
        within_days: 오늘부터 며칠 이내의 마감을 조회할지 (기본 30).
        task_type: 특정 유형만 필터 (예: "안전관리", "품목허가", "변경관리", "라벨링", "임상"). 빈 문자열이면 전체.

    Returns:
        {"today", "deadlines": [{"item","type","due_date","d_day","owner","status"}...]}.
    """
    data = _load_ra_tasks()
    today = _dt.date.today()
    out = []
    for d in data["deadlines"]:
        due = _dt.date.fromisoformat(d["due_date"])
        d_day = (due - today).days
        if d_day > within_days:
            continue
        if task_type and d["type"] != task_type:
            continue
        out.append({**d, "d_day": d_day})
    out.sort(key=lambda x: x["due_date"])
    return {"today": today.isoformat(), "count": len(out), "deadlines": out}


@mcp.tool
def get_submission_checklist(category: str) -> dict:
    """규제 제출 유형별 준비 체크리스트를 반환한다.

    "품목허가 준비할 때 뭐가 필요하냐", "변경허가 체크리스트" 같은 질문에 사용한다.

    Args:
        category: 체크리스트 유형. 지원값: "품목허가", "변경허가", "안전성보고".

    Returns:
        {"category", "items": [...]} 또는 미지원 시 {"error", "available": [...]}.
    """
    data = _load_ra_tasks()
    checklists = data["checklists"]
    if category not in checklists:
        return {"error": f"'{category}' 체크리스트 없음", "available": list(checklists)}
    return {"category": category, "items": checklists[category]}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp.resource("regulation://{doc_id}")
def get_regulation_document(doc_id: str) -> str:
    """규제문서 원문 전체를 doc_id(예: REG-003)로 조회한다."""
    from ..rag.loader import load_documents

    for doc in load_documents(config.REG_DIR):
        if doc.doc_id.lower() == doc_id.lower():
            return f"# {doc.title}\n\n{doc.text}"
    return f"문서를 찾을 수 없음: {doc_id}"


@mcp.tool
def list_regulation_documents() -> dict:
    """검색 가능한 규제문서 목록(제목·카테고리·doc_id)을 반환한다."""
    from ..rag.loader import load_documents

    docs = load_documents(config.REG_DIR)
    return {
        "count": len(docs),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "category": d.metadata.get("category", ""),
                "version": d.metadata.get("version", ""),
            }
            for d in docs
        ],
    }


if __name__ == "__main__":
    # 독립 실행: stdio 트랜스포트로 MCP 서버 구동 (Claude Desktop/Cursor 등에서 연결 가능)
    _get_pipeline()  # 인덱스 미리 구축
    mcp.run()
