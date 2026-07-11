"""RA(규제업무)·PV(약물감시) 어시스턴트를 위한 MCP 서버.

GC 'Hey.GC 2.0'가 MCP 구조로 사내 시스템을 통합하는 것과 동일한 패턴으로,
제약 RA·PV 담당자의 업무 시스템(규제문서·마감일·체크리스트·이상사례 처리)을 MCP 도구로 노출한다.

노출하는 MCP primitive:
  - Tools:     search_regulations / get_ra_deadlines / get_submission_checklist
               / assess_adverse_event (PV 트리아지+인과성+코딩)
               / draft_ae_report (ICSR 초안+최소보고요건 검증)
               / list_regulation_documents
  - Resources: regulation://{doc_id}  (규제문서 원문 조회)
  - Prompts:   pv_case_intake  (케이스 처리 SOP — 클라이언트 무관 동일 절차)

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
        task_type 이 데이터에 없는 유형이면 {"error", "available": [...]} — 매칭 0건과
        구분되는 명시적 에러다.
    """
    data = _load_ra_tasks()
    # 존재하지 않는 유형은 빈 목록이 아니라 에러로 답한다 — 오타 난 필터에
    # "임박한 마감이 없습니다"라고 답하는 것은 '자신 있는 오답'이다(조용한 빈
    # 결과 금지). available 을 함께 주므로 LLM 에이전트가 스스로 정정 재시도한다
    # (get_submission_checklist 와 동일한 에러 계약).
    known_types = sorted({d["type"] for d in data["deadlines"]})
    if task_type and task_type not in known_types:
        return {"error": f"'{task_type}' 유형의 업무가 없음", "available": known_types}
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


@mcp.tool
def assess_adverse_event(case_description: str, awareness_date: str = "") -> dict:
    """이상사례(AE) 케이스를 트리아지한다: 중대성(Serious) 판정 + 보고 경로/기한 계산.

    "환자가 복용 후 입원했는데 언제까지 보고해야 하나" 같은 '구체적 케이스'가
    주어졌을 때 사용한다. (규정 자체가 궁금한 질문은 search_regulations 를 사용.)

    규칙 기반(결정론적)으로 판정한다 — 보고기한 계산은 컴플라이언스라 LLM 추론이
    아닌 감사 가능한 규칙으로 수행하고, 판정 근거 규정(REG-005) 문단을 함께 반환한다.
    입력 속 개인정보(주민번호·연락처·이름 등)는 반환 전에 마스킹된다.
    부가로 인과성(WHO-UMC) 제안과 이상사례 표준 용어 코딩(MedDRA 방식)도 포함한다.

    Args:
        case_description: 이상사례 케이스 서술 (예: "환자가 복용 3일 후 아나필락시스로 입원").
        awareness_date: 회사 인지일(YYYY-MM-DD). 생략 시 오늘 기준으로 기한을 계산.

    Returns:
        {"case", "is_serious", "criteria_met", "expectedness", "route",
         "awareness_date", "deadline_date", "rationale", "caveats",
         "causality": {"suggested","rationale","signals","missing_info"},
         "coded_terms": [{"verbatim","pt","pt_en","soc"}...] (확정 코딩),
         "candidate_terms": [...] (LLT 참조 매칭 후보 — 사람 승인/기각 필요),
         "uncoded_expressions": [...] (감지만 된 미코딩 증상 표현),
         "basis": {"results": [...]}}  — basis 는 판정 근거 규정 문단(출처 포함).
    """
    from ..pv.causality import assess_causality
    from ..pv.coding import code_terms, flag_uncoded_expressions, suggest_candidates
    from ..pv.redactor import redact
    from ..pv.triage import assess_case

    masked = redact(case_description)  # 심층방어: 도구 단독 사용(stdio) 시에도 PII 비노출
    t = assess_case(masked.text, awareness_date)
    c = assess_causality(masked.text)
    coded = code_terms(masked.text)
    candidates = suggest_candidates(masked.text, coded)
    uncoded = flag_uncoded_expressions(masked.text, coded, candidates)
    # 판정 근거 규정 문단을 RAG로 회수해 부착(추적성) — 보고기한의 출처는 REG-005
    basis = search_regulations("중대한 이상사례 보고 기한 신속보고", top_n=2)
    return {
        "case": masked.text,
        "pii_masked": masked.summary(),
        "is_serious": t.is_serious,
        "criteria_met": t.criteria_met,
        "expectedness": t.expectedness,
        "route": t.route,
        "awareness_date": t.awareness_date,
        "deadline_days": t.deadline_days,
        "deadline_date": t.deadline_date,
        "rationale": t.rationale,
        "caveats": t.caveats,
        "causality": {
            "suggested": c.suggested,
            "rationale": c.rationale,
            "signals": c.signals,
            "missing_info": c.missing_info,
        },
        "coded_terms": [ct.as_dict() for ct in coded],
        "candidate_terms": [ct.as_dict() for ct in candidates],
        "uncoded_expressions": uncoded,
        "basis": basis,
    }


@mcp.tool
def draft_ae_report(
    case_description: str,
    suspected_drug: str = "",
    reporter: str = "",
    patient_info: str = "",
    awareness_date: str = "",
) -> dict:
    """이상사례 개별사례보고(ICSR) 초안을 작성한다 — KAERS 제출용.

    "이 케이스 보고서 초안 만들어줘", "KAERS 보고서 작성해줘" 처럼
    '보고서 작성'을 요청받았을 때 사용한다. (중대성/기한만 궁금하면
    assess_adverse_event, 규정 자체가 궁금하면 search_regulations 를 사용.)

    수행 내용(전부 규칙 기반·결정론적):
      1) 최소보고요건(ICH E2D 4요소: 환자·보고자·의심약·이상사례) 충족 검증
         — 미충족이면 reportable=False 와 보완 항목을 반환한다.
      2) 중대성 판정 + 보고기한 계산 (assess_adverse_event 와 동일 규칙)
      3) 인과성(WHO-UMC) 등급 제안 + 부족 정보 follow-up 질문 생성
      4) 이상사례 표준 용어 코딩(MedDRA 방식 PT/SOC) — 확정/후보/미코딩 3계층
         (사전 미수록 표현은 LLT 참조 후보 또는 '미코딩 감지'로 표시하고
         사람 확정을 follow-up 으로 요청)
      5) 위를 종합한 마크다운 초안(draft_markdown) 조립
    개인정보는 초안 생성 전에 마스킹된다 — case_description 뿐 아니라 reporter·
    patient_info 등 모든 자유 텍스트 인자가 대상이다(초안에는 비식별 서술만 남음).

    Args:
        case_description: 이상사례 케이스 서술(자유 텍스트).
        suspected_drug: 의심 의약품명(알면 지정 — 최소요건 ③).
        reporter: 보고자 정보(예: "의사", "약사 홍OO" — 최소요건 ②).
        patient_info: 환자 비식별 정보(예: "45세 남성" — 최소요건 ①).
        awareness_date: 회사 인지일(YYYY-MM-DD). 생략 시 오늘.

    Returns:
        {"reportable", "missing", "followups", "draft_markdown",
         "is_serious", "deadline_date", "causality", "coded_terms",
         "candidate_terms", "uncoded_expressions",
         "pii_masked", "basis": {"results": [...]}}.
    """
    from ..pv.redactor import merged_summary, redact
    from ..pv.report import build_report

    # 자유 텍스트 인자는 '전부' 마스킹한다 — case_description 만 막으면 나머지
    # 인자(reporter="약사 홍길동님 010-…")가 PII 우회로가 된다. 에이전트 경유
    # 시에는 인자가 이미 마스킹된 메시지에서 파생되지만, stdio 단독 사용
    # (Claude Desktop 등)에서는 인자가 외부에서 직접 들어온다(심층방어).
    masked = redact(case_description)
    masked_drug = redact(suspected_drug)
    masked_reporter = redact(reporter)
    masked_patient = redact(patient_info)
    r = build_report(
        masked.text,
        suspected_drug=masked_drug.text,
        reporter=masked_reporter.text,
        patient_info=masked_patient.text,
        awareness_date=awareness_date,
    )
    basis = search_regulations("중대한 이상사례 보고 기한 신속보고 인과성 평가", top_n=2)
    return {
        "reportable": r.reportable,
        "missing": r.missing,
        "followups": r.followups,
        "draft_markdown": r.draft_markdown,
        "is_serious": r.triage.is_serious,
        "deadline_date": r.triage.deadline_date,
        "causality": {
            "suggested": r.causality.suggested,
            "rationale": r.causality.rationale,
            "missing_info": r.causality.missing_info,
        },
        "coded_terms": [ct.as_dict() for ct in r.coded_terms],
        "candidate_terms": [ct.as_dict() for ct in r.candidate_terms],
        "uncoded_expressions": r.uncoded_expressions,
        "pii_masked": merged_summary(masked, masked_drug, masked_reporter, masked_patient),
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# Prompts — MCP 의 세 번째 primitive (Tools·Resources·Prompts 3종 완성).
# 서버가 '도구를 올바른 순서로 쓰는 워크플로'까지 배포한다: 어느 클라이언트
# (Claude Desktop·Cursor·사내 에이전트)가 붙어도 같은 SOP 로 케이스를 처리한다.
# ---------------------------------------------------------------------------
@mcp.prompt
def pv_case_intake(case_description: str) -> str:
    """PV 이상사례 케이스 접수 SOP 프롬프트 — 접수부터 보고서 초안까지의 표준 절차."""
    return (
        "다음 이상사례 케이스를 PV 접수 SOP에 따라 처리하라.\n\n"
        f"케이스: {case_description}\n\n"
        "절차:\n"
        "1. assess_adverse_event 로 중대성·보고기한·인과성 제안·용어 코딩을 확인한다.\n"
        "2. 보고 기한의 근거 규정이 필요하면 search_regulations 로 원문을 확인한다.\n"
        "3. draft_ae_report 로 ICSR 초안을 만들고, 최소보고요건 누락(missing)과\n"
        "   follow-up 질문(followups)을 사용자에게 명확히 안내한다.\n"
        "4. 모든 판정에 '최종 확정은 PV 담당자' 임을 밝히고, 출처(문서·섹션)를 제시한다."
    )


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
