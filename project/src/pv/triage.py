"""이상사례(Adverse Event) 트리아지 — 중대성 판정 + 보고 경로/기한 계산.

PV 담당자의 첫 업무는 접수된 케이스를 '어떤 경로로 언제까지 보고해야 하는가'로
분류(triage)하는 것이다. 이 판정을 LLM에게 맡기지 않고 규칙 기반으로 구현한 이유:

  - 보고기한 계산은 컴플라이언스 그 자체다. 하루라도 틀리면 사고인데,
    LLM의 날짜 연산·기준 적용은 확률적이라 감사(audit) 대상이 될 수 없다.
  - 중대성(Serious) 기준은 규정(REG-005)에 닫힌 목록으로 정의되어 있어
    결정론적 규칙으로 충분히 표현된다 — 판단이 아니라 '대조'다.
  - 에이전트 설계 원칙: LLM은 도구 선택·설명(orchestration)을 맡고,
    규정이 정한 계산은 결정론적 도구가 맡는다. 결과가 항상 재현·검증된다.

판정 기준·기한은 사내 규제문서 REG-005(시판 후 안전관리 및 약물감시 기준)를
코드로 옮긴 것이며, MCP 도구 계층에서 해당 근거 문단을 함께 반환한다.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

# 중대성(Serious) 기준 — REG-005 §2 의 닫힌 목록을 (기준명, 감지 키워드) 로 코드화.
# 키워드는 케이스 서술(자유 텍스트)에서 기준 충족을 감지하는 데 쓴다.
SERIOUSNESS_CRITERIA: dict[str, list[str]] = {
    "사망": ["사망", "숨졌", "죽었", "사망함"],
    "생명 위협": ["생명", "생명위협", "아나필락시스", "쇼크", "심정지", "소생술", "중환자실"],
    "입원 또는 입원기간 연장": ["입원", "입원기간 연장", "재입원"],
    "지속적/중대한 장애": ["장애", "실명", "청력 상실", "영구적"],
    "선천적 기형": ["기형", "선천성", "선천적"],
    "기타 의학적으로 중요한 사건": ["의학적으로 중요"],
}

# 즉시(지체 없이) 신속보고 대상 기준 — REG-005 §2 표의 '사망·생명위협' 행
_IMMEDIATE = {"사망", "생명 위협"}

# 예상 여부(expectedness) 감지 — 허가사항(첨부문서)에 이미 기재된 반응인가
_EXPECTED_MARKERS = ["허가사항에 기재", "첨부문서에 기재", "알려진 부작용", "예상된"]
_UNEXPECTED_MARKERS = ["예상치 못한", "예상하지 못한", "허가사항에 없", "알려지지 않은"]


@dataclass
class TriageResult:
    is_serious: bool
    criteria_met: list[str]            # 충족한 중대성 기준
    expectedness: str                  # expected | unexpected | unknown
    route: str                         # 보고 경로 요약
    deadline_days: int | None          # 인지일로부터 며칠(즉시=0, 정기보고=None)
    awareness_date: str                # 인지일(YYYY-MM-DD)
    deadline_date: str | None          # 계산된 보고 마감일
    rationale: str                     # 판정 사유(사람이 읽는 설명)
    caveats: list[str] = field(default_factory=list)


def _detect_criteria(case_text: str) -> list[str]:
    low = case_text.lower()
    return [
        name
        for name, keywords in SERIOUSNESS_CRITERIA.items()
        if any(k in low for k in keywords)
    ]


def _detect_expectedness(case_text: str) -> str:
    if any(m in case_text for m in _UNEXPECTED_MARKERS):
        return "unexpected"
    if any(m in case_text for m in _EXPECTED_MARKERS):
        return "expected"
    return "unknown"


def assess_case(case_text: str, awareness_date: str = "") -> TriageResult:
    """케이스 서술을 트리아지한다.

    Args:
        case_text: 이상사례 케이스 서술(자유 텍스트).
        awareness_date: 회사가 케이스를 인지한 날(YYYY-MM-DD). 없으면 오늘.
            형식이 잘못되면 오늘로 폴백하되 caveat 로 명시한다(조용한 폴백 금지).

    판정 규칙(REG-005 §2):
      - 사망·생명위협           → 지체 없이(신속보고, D+0)
      - 그 외 중대 기준 충족     → 인지일로부터 15일 이내 신속보고
      - 비중대                  → 정기보고(PSUR)에 포함
    예상 여부를 판별할 수 없으면 '예상치 못한 사례'로 보수적으로 취급한다
    (기한을 놓치는 쪽보다 이르게 잡는 쪽이 안전한 실패).
    """
    # 인지일 형식 오류는 '조용히' 오늘로 대체하지 않는다 — 보고기한이 잘못된
    # 기준일로 계산되는 것은 컴플라이언스 리스크인데, 폴백이 조용하면 사용자는
    # 기한이 틀렸다는 신호를 받을 길이 없다(시끄러운 실패 원칙).
    caveats: list[str] = []
    try:
        aware = _dt.date.fromisoformat(awareness_date) if awareness_date else _dt.date.today()
    except ValueError:
        aware = _dt.date.today()
        caveats.append(
            f"인지일 '{awareness_date}' 이(가) YYYY-MM-DD 형식이 아니어서 "
            f"오늘({aware.isoformat()}) 기준으로 기한을 계산했습니다. "
            "실제 인지일로 기한 재계산이 필요합니다."
        )
    # 형식은 유효해도 '오늘보다 미래'인 인지일은 오타(연도·월 뒤바뀜)일 가능성이
    # 높다 — 형식 검사만 하고 타당성(plausibility)은 안 보면, 2027 로 잘못 친
    # 인지일이 기한을 1년 뒤로 조용히 밀어도 아무 신호가 없다(형식 오류에는
    # 시끄러운 caveat 를 달아 놓고 값 오류는 조용히 통과시키는 비대칭).
    # 계산은 입력값대로 수행한다 — 자동 정정은 또 다른 조용한 폴백이다.
    if aware > _dt.date.today():
        caveats.append(
            f"인지일 {aware.isoformat()} 이(가) 오늘 이후의 미래 날짜입니다 — "
            "입력 오류(연도·월 오타)일 수 있으니 실제 인지일을 확인하세요. "
            "기한은 입력된 값 기준으로 계산되어 있습니다."
        )

    criteria = _detect_criteria(case_text)
    expectedness = _detect_expectedness(case_text)

    if not criteria:
        return TriageResult(
            is_serious=False,
            criteria_met=[],
            expectedness=expectedness,
            route="비중대 이상사례 → 정기보고(PSUR)에 포함",
            deadline_days=None,
            awareness_date=aware.isoformat(),
            deadline_date=None,
            rationale="케이스 서술에서 중대성(Serious) 기준(사망·생명위협·입원·장애·기형 등)이 감지되지 않았다.",
            caveats=[*caveats, "규칙 기반 1차 분류입니다. 최종 중대성 판단은 PV 담당자가 확정해야 합니다."],
        )

    if _IMMEDIATE & set(criteria):
        deadline_days = 0
        route = "신속보고 — 지체 없이(사망·생명위협)"
        rationale = f"사망·생명위협 기준 충족({', '.join(sorted(_IMMEDIATE & set(criteria)))}) → 지체 없이 보고."
    else:
        deadline_days = 15
        route = "신속보고 — 인지일로부터 15일 이내"
        rationale = f"중대성 기준 충족({', '.join(criteria)}) → 15일 이내 신속보고."
        if expectedness == "expected":
            caveats.append(
                "허가사항에 기재된(예상된) 반응으로 보이나, 보수적으로 15일 트래킹을 권고합니다. "
                "정기보고 전환 여부는 규정 원문과 PV 담당자 확인이 필요합니다."
            )
        elif expectedness == "unknown":
            caveats.append("예상 여부(expectedness)를 판별하지 못해 '예상치 못한 사례'로 보수 적용했습니다.")

    caveats.append("규칙 기반 1차 분류입니다. 최종 중대성·인과성 판단은 PV 담당자가 확정해야 합니다.")
    return TriageResult(
        is_serious=True,
        criteria_met=criteria,
        expectedness=expectedness,
        route=route,
        deadline_days=deadline_days,
        awareness_date=aware.isoformat(),
        deadline_date=(aware + _dt.timedelta(days=deadline_days)).isoformat(),
        rationale=rationale,
        caveats=caveats,
    )
