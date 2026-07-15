"""FastAPI 백엔드 — RA·PV 어시스턴트 웹 서비스.

엔드포인트:
  GET  /            → 웹 챗 UI (single page)
  GET  /health      → 실행 모드/인덱스 상태 + 검증 게이트 경고율 계기판
  POST /chat        → 사용자 메시지 → 에이전트 응답(+도구호출·출처)
  GET  /api/deadlines → 대시보드용 마감일 (부가)

FDE 관점: 에이전트(백엔드)를 API로 서빙하고 프론트(챗 UI)와 연결하는
전형적인 풀스택 구조. 실제 배포 시 인증·로깅·관측성을 이 계층에 추가한다.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import config
from ..agent.agent import RaAgent
from ..mcp_server.server import _get_pipeline
from ..observability import gate_stats
from ..ra.tasks import load_ra_tasks

agent = RaAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 부팅 시 RAG 인덱스 구축(무거운 초기화 1회)
    p = _get_pipeline()
    app.state.pipeline_info = {"docs": p.n_docs, "chunks": p.n_chunks}
    print(config.mode_banner())
    print(f"[RAG] 문서 {p.n_docs}건 · 청크 {p.n_chunks}개 인덱싱 완료")
    yield


app = FastAPI(title="RAPV-Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    history: list[dict] = Field(default_factory=list, description="이전 대화(LLM 모드)")


class ChatResponse(BaseModel):
    answer: str
    mode: str
    tool_calls: list[dict]
    citations: list[dict]
    grounded: bool = True
    trace: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0
    redactions: list[dict] = Field(default_factory=list)  # PII 마스킹 내역(유형·건수만)
    verification: dict = Field(default_factory=dict)  # 답변 사후 검증(수치 대조·버전 점검)


@app.get("/health")
async def health() -> JSONResponse:
    p = _get_pipeline()
    return JSONResponse(
        {
            "status": "ok",
            "mode": "llm" if config.LLM_AVAILABLE else "offline",
            "banner": config.mode_banner(),
            "rag": {"docs": p.n_docs, "chunks": p.n_chunks},
            "rag_params": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "retrieve_top_k": config.RETRIEVE_TOP_K,
                "rerank_top_n": config.RERANK_TOP_N,
                "hybrid_alpha": config.HYBRID_ALPHA,
            },
            # 검증 게이트 운영 계기판 — 경고율이 오르면 (a) 답변 품질 회귀 또는
            # (b) 검증기 오탐 증가(alert fatigue 위험)의 조기 신호다.
            "verification_gate": gate_stats.snapshot(),
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = await agent.chat(req.message, req.history)
    return ChatResponse(
        answer=result.answer,
        mode=result.mode,
        tool_calls=[
            {"name": t.name, "args": t.args, "summary": t.result_summary}
            for t in result.tool_calls
        ],
        citations=result.citations,
        grounded=result.grounded,
        trace=result.trace,
        latency_ms=result.latency_ms,
        redactions=result.redactions,
        verification=result.verification,
    )


@app.get("/api/deadlines")
async def deadlines() -> JSONResponse:
    return JSONResponse(load_ra_tasks()["deadlines"])


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (config.WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
