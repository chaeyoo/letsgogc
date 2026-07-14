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

# 경과(outcome) 창 기반 감지 (v8) — "중단하니"·"다시 복용하니" 같은 결과
# 미포함 마커는 뒤따르는 경과가 호전이든 악화든 무조건 양성이 됐다.
# negative dechallenge("중단하니 오히려 악화")와 negative rechallenge
# ("다시 복용하니 아무 증상이 없었다")는 인과성의 '반증'인데, 마커가 결과를
# 안 보니 신호가 반대로 최고 등급(Certain)을 지지하고 rationale 문장까지
# 사실과 어긋났다. 마커는 '맥락의 시작'만 표시하고, 경과 어휘는 그 마커
# 이후의 창(window) 안에서 따로 판정한다 — 창의 끝은 다음 맥락 마커
# (중단 서술 뒤에 재투여 서술이 이어지는 케이스에서 경과가 섞이지 않도록).
_DECHALLENGE_CONTEXT = ["중단", "끊은", "끊으니", "끊었", "중지"]
_RECHALLENGE_CONTEXT = ["재투여", "다시 복용", "다시 투여", "재복용"]
_IMPROVE_WORDS = ["호전", "회복", "소실", "사라", "좋아", "가라앉"]
_NOT_IMPROVE_WORDS = ["악화", "에도 지속", "그대로", "계속되", "여전히"]
_RECUR_WORDS = ["재발", "다시 나타", "다시 발생", "또 나타", "같은 증상", "동일 증상", "재현"]
_NO_RECUR_WORDS = ["재발하지 않", "재발은 없", "재발이 없", "증상이 없", "증상은 없", "아무 증상", "나타나지 않", "발생하지 않"]

# 대체 원인은 3상태(v8): 존재(present) / 명시 배제(excluded) / 미언급(unknown).
# '미언급'을 '없음 충족'으로 치면 docstring 의 원칙("정보가 없는 요소는 등급을
# 올려주지 않는다")과 정면 모순 — 대체원인만은 not alternative 로 위로 열려
# 있었다. 또 "병용약물은 없었다" 같은 명시 부정문이 '병용' 부분 매칭으로
# 대체원인 '있음'이 되어 등급이 반대 방향으로 밀리고 사유 문장도 틀렸다.
_ALTERNATIVE_CAUSE = ["병용", "함께 복용", "다른 약", "기저질환", "원질환", "원래 앓", "기존 질환", "지병"]
_ALTERNATIVE_NEGATED = [
    "병용약물은 없", "병용약은 없", "병용약 없", "병용 약물은 없", "병용약물이나 기저질환은 없",
    "기저질환은 없", "기저질환 없", "기저질환이나 병용약물은 없", "다른 약은 없", "다른 약물은 없",
    "지병은 없", "특이 병력 없", "특이 기저질환 없", "대체 원인 없",
]

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


def _window(text: str, start_markers: list[str], stop_markers: list[str]) -> str | None:
    """첫 start 마커부터 (있다면) 다음 stop 마커 직전까지의 경과 서술 창."""
    starts = [text.find(m) for m in start_markers if m in text]
    if not starts:
        return None
    start = min(starts)
    stops = [i for m in stop_markers if (i := text.find(m, start + 1)) > start]
    return text[start : min(stops) if stops else len(text)]


def _dechallenge_state(case_text: str) -> str:
    """중단(dechallenge) 경과 — improved | not_improved | unknown.

    창의 끝을 재투여 맥락에서 끊는다 — "중단 후 호전, 재투여하니 악화"에서
    악화(재투여 경과)가 중단 경과를 오염시키지 않도록.
    """
    win = _window(case_text, _DECHALLENGE_CONTEXT, _RECHALLENGE_CONTEXT)
    if win is None:
        return "unknown"
    if any(w in win for w in _NOT_IMPROVE_WORDS):
        return "not_improved"
    if any(w in win for w in _IMPROVE_WORDS):
        return "improved"
    return "unknown"


def _rechallenge_state(case_text: str) -> str:
    """재투여(rechallenge) 경과 — recurred | no_recurrence | unknown."""
    win = _window(case_text, _RECHALLENGE_CONTEXT, [])
    if win is None:
        return "unknown"
    if any(w in win for w in _NO_RECUR_WORDS):
        return "no_recurrence"
    if any(w in win for w in _RECUR_WORDS):
        return "recurred"
    return "unknown"


def _alternative_state(case_text: str) -> str:
    """대체 원인 — present | excluded | unknown. 명시 부정문을 먼저 본다."""
    if any(m in case_text for m in _ALTERNATIVE_NEGATED):
        return "excluded"
    if _detect(case_text, _ALTERNATIVE_CAUSE):
        return "present"
    return "unknown"


def assess_causality(case_text: str) -> CausalityResult:
    """케이스 서술에서 WHO-UMC 신호를 감지해 인과성 등급을 '제안'한다.

    등급 규칙(WHO-UMC 기준을 감지 가능한 신호로 근사, v8):
      - 시간관계 + 중단 후 호전 + 재투여 후 재발 + 대체원인 '명시 배제' → Certain
      - 위와 같으나 대체원인 미언급                                   → Probable
        (미언급은 배제 확인이 아니다 — Certain 은 대체원인 배제가 '확인'될 때만)
      - 재투여 후 '재발하지 않음'(반증 신호)                           → Possible 상한
      - 시간관계 + 중단 후 호전 (대체원인 미확인/배제)                 → Probable
      - 시간관계 있음(대체원인 존재 포함)                              → Possible
      - 시간관계 없음 + 대체원인 존재                                  → Unlikely
      - 신호 자체가 없음                                               → Unassessable
    정보가 서술에 없으면 '충족'으로 치지 않는다 — 등급은 아래로만 보수 적용된다.
    """
    temporal = _detect(case_text, _TEMPORAL_MARKERS)
    de_state = _dechallenge_state(case_text)
    re_state = _rechallenge_state(case_text)
    alt_state = _alternative_state(case_text)

    dechallenge = de_state == "improved"
    rechallenge = re_state == "recurred"
    alternative = alt_state == "present"

    # 표시용 신호는 '양성 확인'만 True — 반증/미언급은 rationale·질문이 나른다.
    signals = {
        "시간적 선후관계": temporal,
        "중단 후 호전(dechallenge)": dechallenge,
        "재투여 후 재발(rechallenge)": rechallenge,
        "대체 원인 가능성(병용약·기저질환)": alternative,
    }

    # 부족 정보 → follow-up 질문. '미언급(unknown)'일 때만 묻는다 —
    # 반증(악화·미재발)이나 명시 배제는 이미 '정보가 있는' 상태다.
    missing: list[str] = []
    if not temporal:
        missing.append("의심약 투여와 증상 발생의 시간적 선후관계(투여 후 언제 발생했는가)")
    if de_state == "unknown":
        missing.append("의심약 중단(dechallenge) 후 증상 경과(호전/지속)")
    if re_state == "unknown":
        missing.append("재투여(rechallenge) 여부와 재발 여부")
    if alt_state == "unknown":
        missing.append("병용약물·기저질환 등 대체 원인 유무")

    if temporal and dechallenge and rechallenge and alt_state == "excluded":
        suggested = CERTAIN
        rationale = "시간관계·중단 후 호전·재투여 후 재발이 모두 확인되고 대체 원인이 명시적으로 배제되었다."
    elif temporal and dechallenge and rechallenge:
        suggested = PROBABLE
        rationale = (
            "시간관계·중단 후 호전·재투여 후 재발은 확인되나, 대체 원인의 배제가 "
            "서술로 확인되지 않아 Certain 은 보류한다(미언급은 배제 확인이 아니다)."
        )
    elif temporal and re_state == "no_recurrence":
        suggested = POSSIBLE
        rationale = "재투여 후 재발하지 않아(반증 신호) 시간관계만으로 Possible 이상 올리지 않는다."
    elif temporal and dechallenge and alt_state != "present":
        suggested = PROBABLE
        rationale = (
            "시간관계와 중단 후 호전이 확인되고 대체 원인이 명시적으로 배제되었다(재투여 정보는 없음)."
            if alt_state == "excluded"
            else "시간관계와 중단 후 호전이 확인된다(대체 원인 미언급 — 재투여 정보 없이 이 이상 올리지 않는다)."
        )
    elif temporal:
        suggested = POSSIBLE
        rationale = (
            "시간적 선후관계는 확인되나 "
            + (
                "병용약·기저질환 등 대체 원인 가능성이 있다."
                if alternative
                else (
                    "중단 후 호전이 확인되지 않아(악화/지속) 그 이상 올릴 수 없다."
                    if de_state == "not_improved"
                    else "중단/재투여 경과 정보가 없어 그 이상 올릴 수 없다."
                )
            )
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
