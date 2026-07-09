"""인과성 평가 (Causality Assessment) — WHO-UMC 척도의 규칙 기반 '제안'.

PV 트리아지 다음 단계는 "이 이상사례가 정말 약 때문인가"를 평가하는 것이다.
REG-005 §2가 명시하듯 실무는 WHO-UMC 또는 Naranjo 척도를 쓴다.

트리아지(중대성 판정)와 달리 인과성은 규칙만으로 '확정'할 수 없다:
중대성 기준은 닫힌 목록(대조)이지만, 인과성은 임상 정보(중단/재투여 경과,
병용약물, 기저질환)를 종합하는 '판단'이라 정보가 부족하면 답이 없다.

그래서 이 모듈의 출력은 판정이 아니라 **제안(suggested) + 근거 신호 + 부족 정보 질문**이다:
  - 케이스 서술에서 WHO-UMC 판단 요소 4가지 신호를 감지한다
    (시간적 선후관계 · dechallenge · rechallenge · 대체 원인).
  - 감지된 신호 조합으로 도달 '가능한 최대' 등급을 제안하되, 정보가 없는
    요소는 충족으로 간주하지 않는다(보수적 — 등급을 올려주지 않는다).
  - 판단에 필요한데 서술에 없는 정보를 `missing_info` 질문 목록으로 돌려줘,
    PV 담당자가 보고자에게 무엇을 되물어야 하는지(follow-up query) 안내한다.

이 경계 설계가 핵심이다: 규칙은 '신호 감지와 질문 생성'까지, 확정은 사람.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# WHO-UMC 판단 요소별 감지 키워드 (케이스 자유 서술 대상)
_TEMPORAL_MARKERS = ["복용 후", "투여 후", "접종 후", "복용 직후", "투여 직후", "맞은 후", "먹은 후", "복용하고", "투여하고"]
_DECHALLENGE_POSITIVE = ["중단 후 호전", "중단 후 회복", "중단하니", "끊은 후 호전", "끊으니", "중지 후 회복", "중지 후 호전", "중단 후 소실"]
_RECHALLENGE_POSITIVE = ["재투여 후 재발", "재투여하니", "다시 복용하니", "다시 투여하니", "재복용 후"]
_ALTERNATIVE_CAUSE = ["병용", "함께 복용", "다른 약", "기저질환", "원질환", "원래 앓", "기존 질환", "지병"]

# WHO-UMC 등급 (영문 원어 병기 — KAERS/해외 보고 시 그대로 쓰는 용어)
CERTAIN = "확실함(Certain)"
PROBABLE = "상당히 확실함(Probable/Likely)"
POSSIBLE = "가능함(Possible)"
UNLIKELY = "가능성 적음(Unlikely)"
UNASSESSABLE = "평가곤란(Unassessable)"


@dataclass
class CausalityResult:
    suggested: str                       # 제안 등급(WHO-UMC)
    signals: dict[str, bool]             # 감지된 판단 요소 신호
    rationale: str                       # 등급 제안 사유
    missing_info: list[str] = field(default_factory=list)  # 보고자에게 되물을 질문


def _detect(case_text: str, markers: list[str]) -> bool:
    return any(m in case_text for m in markers)


def assess_causality(case_text: str) -> CausalityResult:
    """케이스 서술에서 WHO-UMC 신호를 감지해 인과성 등급을 '제안'한다.

    등급 규칙(WHO-UMC 기준을 감지 가능한 신호로 근사):
      - 시간관계 + 중단 후 호전 + 재투여 후 재발 + 대체원인 없음 → Certain
      - 시간관계 + 중단 후 호전 + 대체원인 없음               → Probable
      - 시간관계 있음(대체원인 유무 무관)                      → Possible
      - 시간관계 없음 + 대체원인 있음                          → Unlikely
      - 신호 자체가 없음                                       → Unassessable
    정보가 서술에 없으면 '충족'으로 치지 않는다 — 등급은 아래로만 보수 적용된다.
    """
    signals = {
        "시간적 선후관계": _detect(case_text, _TEMPORAL_MARKERS),
        "중단 후 호전(dechallenge)": _detect(case_text, _DECHALLENGE_POSITIVE),
        "재투여 후 재발(rechallenge)": _detect(case_text, _RECHALLENGE_POSITIVE),
        "대체 원인 가능성(병용약·기저질환)": _detect(case_text, _ALTERNATIVE_CAUSE),
    }
    temporal = signals["시간적 선후관계"]
    dechallenge = signals["중단 후 호전(dechallenge)"]
    rechallenge = signals["재투여 후 재발(rechallenge)"]
    alternative = signals["대체 원인 가능성(병용약·기저질환)"]

    # 부족 정보 → PV 담당자가 보고자에게 되물을 follow-up 질문 생성
    missing: list[str] = []
    if not temporal:
        missing.append("의심약 투여와 증상 발생의 시간적 선후관계(투여 후 언제 발생했는가)")
    if not dechallenge:
        missing.append("의심약 중단(dechallenge) 후 증상 경과(호전/지속)")
    if not rechallenge:
        missing.append("재투여(rechallenge) 여부와 재발 여부")
    if not alternative:
        missing.append("병용약물·기저질환 등 대체 원인 유무")

    if temporal and dechallenge and rechallenge and not alternative:
        suggested = CERTAIN
        rationale = "시간관계·중단 후 호전·재투여 후 재발이 모두 확인되고 대체 원인 언급이 없다."
    elif temporal and dechallenge and not alternative:
        suggested = PROBABLE
        rationale = "시간관계와 중단 후 호전이 확인되고 대체 원인 언급이 없다(재투여 정보는 없음)."
    elif temporal:
        suggested = POSSIBLE
        rationale = (
            "시간적 선후관계는 확인되나 "
            + ("병용약·기저질환 등 대체 원인 가능성이 있다." if alternative else "중단/재투여 경과 정보가 없어 그 이상 올릴 수 없다.")
        )
    elif alternative:
        suggested = UNLIKELY
        rationale = "시간적 선후관계가 서술에 없고 대체 원인(병용약·기저질환) 가능성이 있다."
    else:
        suggested = UNASSESSABLE
        rationale = "인과성 판단 요소(시간관계·경과·대체원인)가 서술에서 감지되지 않는다."

    return CausalityResult(
        suggested=suggested,
        signals=signals,
        rationale=rationale,
        missing_info=missing,
    )
