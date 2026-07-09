"""이상사례 개별사례보고(ICSR) 초안 — 최소보고요건 검증 + KAERS 초안 조립.

트리아지(triage)·인과성 제안(causality)·용어 코딩(coding)을 한 번에 묶어
PV 담당자가 KAERS(한국 의약품 이상사례 보고시스템)에 올릴 보고서의
'초안'을 만든다. 사람이 빈칸을 채워 확정하는 것이 전제다.

핵심은 초안 생성보다 **최소보고요건(minimum reporting criteria) 검증**이다.
ICH E2D가 정의하는 유효한 케이스의 4요소:
  ① 식별 가능한 환자  ② 식별 가능한 보고자  ③ 의심 의약품  ④ 이상사례
하나라도 없으면 '보고 가능한 케이스'가 아니라 '정보 보완 대상'이다.
→ 이 도구는 4요소 충족 여부를 판정하고, 빠진 요소를 follow-up 항목으로 안내한다.

PII 마스킹과의 관계(설계상 긴장 지점):
  ①의 요건은 '환자를 특정할 수 있는 정보가 존재하는가'이지 원문 값이 아니다.
  마스킹은 값을 지우지만 존재 신호([이름]님, 환자번호 [번호], "45세 남성")는
  남기므로, 마스킹된 텍스트 위에서도 요건 판정이 가능하다.
  즉 '외부 API 경계에서는 값을 감추고, 요건 판정은 신호로 한다'로 양립시켰다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .causality import CausalityResult, assess_causality
from .coding import CodedTerm, code_terms
from .triage import TriageResult, assess_case

# 최소보고요건 감지 신호 (마스킹된 텍스트에서도 남는 '존재 신호' 위주)
_PATIENT_MARKERS = ["환자", "[이름]", "환아", "수진자"]
_PATIENT_RE = re.compile(r"\d{1,3}\s*세|[남여]성|남아|여아")
_REPORTER_MARKERS = ["보고자", "의사", "약사", "간호사", "보호자", "본인이 직접", "의료진", "병원에서"]
# 의심약 감지: "OO정/캡슐/주사 + 복용/투여" 처럼 노출(exposure) 맥락이 따라올 때만
# 매칭한다 — '규정·판정·일정' 같은 '-정'으로 끝나는 일반 명사의 오탐을 막기 위해.
_DRUG_RE = re.compile(
    r"[가-힣A-Za-z0-9]+\s*(?:정|캡슐|주사제|주사|시럽|연고|패치)"
    r"(?=\s*(?:을|를)?\s*(?:복용|투여|접종|먹|맞))"
)


@dataclass
class ReportDraft:
    reportable: bool                     # 최소보고요건 4요소 충족 여부
    missing: list[str]                   # 빠진 요소(보완 안내)
    triage: TriageResult
    causality: CausalityResult
    coded_terms: list[CodedTerm]
    draft_markdown: str = ""
    followups: list[str] = field(default_factory=list)   # 보고자에게 되물을 질문


def _check_minimum_criteria(
    case_text: str, suspected_drug: str, reporter: str, patient_info: str,
    coded: list[CodedTerm], triage: TriageResult,
) -> tuple[list[str], dict[str, str]]:
    """ICH E2D 최소보고요건 4요소를 점검한다. (빠진 요소 목록, 채워진 값) 반환."""
    fields: dict[str, str] = {}
    missing: list[str] = []

    if patient_info.strip():
        fields["환자"] = patient_info.strip()
    elif any(m in case_text for m in _PATIENT_MARKERS) or _PATIENT_RE.search(case_text):
        fields["환자"] = "케이스 서술에서 확인(비식별 처리됨) — 나이/성별/이니셜 보완 권장"
    else:
        missing.append("① 식별 가능한 환자 (나이·성별·이니셜 등 최소 식별 정보)")

    if reporter.strip():
        fields["보고자"] = reporter.strip()
    elif any(m in case_text for m in _REPORTER_MARKERS):
        fields["보고자"] = "케이스 서술에서 확인 — 보고자 자격(의사/약사/소비자 등) 명시 권장"
    else:
        missing.append("② 식별 가능한 보고자 (보고자 자격과 연락 경로)")

    drug_match = _DRUG_RE.search(case_text)
    if suspected_drug.strip():
        fields["의심 의약품"] = suspected_drug.strip()
    elif drug_match:
        fields["의심 의약품"] = f"서술에서 감지: {drug_match.group(0).strip()} — 제품명·성분 확정 필요"
    else:
        missing.append("③ 의심 의약품 (제품명 또는 성분명)")

    if coded or triage.criteria_met:
        fields["이상사례"] = ", ".join(t.pt for t in coded) if coded else "서술에서 확인(코딩 필요)"
    else:
        missing.append("④ 이상사례 (구체적 증상/사건)")

    return missing, fields


def _render_markdown(
    case_text: str, fields: dict[str, str], missing: list[str],
    triage: TriageResult, causality: CausalityResult, coded: list[CodedTerm],
) -> str:
    """KAERS 개별사례안전성보고(ICSR) 항목 구조를 따르는 사람이 읽는 초안."""
    lines = ["# 이상사례 개별사례보고(ICSR) 초안 — KAERS 제출용", ""]
    status = "✅ 최소보고요건 충족(초안 검토 후 제출 가능)" if not missing else "⛔ 정보 보완 필요(최소보고요건 미충족)"
    lines += [f"**상태: {status}**", ""]

    lines.append("## 1. 최소보고요건 (ICH E2D 4요소)")
    for name in ["환자", "보고자", "의심 의약품", "이상사례"]:
        lines.append(f"- {name}: {fields.get(name, '**(미확인 — 보완 필요)**')}")
    lines.append("")

    if coded:
        lines.append("## 2. 이상사례 표준 용어 코딩 (MedDRA 방식)")
        lines.append("| 서술 표현 | PT(표준 용어) | SOC(기관계) |")
        lines.append("|---|---|---|")
        for t in coded:
            lines.append(f"| {t.verbatim} | {t.pt} ({t.pt_en}) | {t.soc} |")
        lines.append("")

    lines.append("## 3. 중대성 및 보고 기한 (규칙 기반 판정)")
    if triage.is_serious:
        lines.append(f"- 중대성: **중대(Serious)** — 기준: {', '.join(triage.criteria_met)}")
    else:
        lines.append("- 중대성: 비중대 (중대성 기준 미감지)")
    lines.append(f"- 보고 경로: {triage.route}")
    if triage.deadline_date:
        lines.append(f"- 보고 기한: **{triage.deadline_date}** (인지일 {triage.awareness_date} 기준)")
    lines.append("")

    lines.append("## 4. 인과성 평가 (WHO-UMC · 제안)")
    lines.append(f"- 제안 등급: **{causality.suggested}**")
    lines.append(f"- 사유: {causality.rationale}")
    detected = [k for k, v in causality.signals.items() if v]
    if detected:
        lines.append(f"- 감지된 판단 요소: {', '.join(detected)}")
    lines.append("")

    lines.append("## 5. 경과 서술 (개인정보 비식별 처리본)")
    lines.append(f"> {case_text.strip()}")
    lines.append("")

    lines.append("---")
    lines.append("⚠ 본 초안은 규칙 기반 자동 생성입니다. 중대성·인과성·코딩의 최종 확정과 제출 책임은 PV 담당자에게 있습니다.")
    return "\n".join(lines)


def build_report(
    case_text: str,
    suspected_drug: str = "",
    reporter: str = "",
    patient_info: str = "",
    awareness_date: str = "",
) -> ReportDraft:
    """케이스 서술(+선택 필드)로 ICSR 초안을 조립한다.

    입력 텍스트는 이미 PII 마스킹된 상태를 전제한다(MCP 도구 계층에서 수행).
    """
    triage = assess_case(case_text, awareness_date)
    causality = assess_causality(case_text)
    coded = code_terms(case_text)
    missing, fields = _check_minimum_criteria(
        case_text, suspected_drug, reporter, patient_info, coded, triage
    )

    # follow-up: 최소요건 누락 + 인과성 판단에 부족한 정보를 하나의 질문 목록으로
    followups = [f"최소보고요건 보완: {m}" for m in missing]
    followups += [f"인과성 평가 보완: {q}" for q in causality.missing_info]

    draft = _render_markdown(case_text, fields, missing, triage, causality, coded)
    return ReportDraft(
        reportable=not missing,
        missing=missing,
        triage=triage,
        causality=causality,
        coded_terms=coded,
        draft_markdown=draft,
        followups=followups,
    )
