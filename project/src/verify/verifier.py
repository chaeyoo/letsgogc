"""답변 사후 검증(post-generation verification) — 생성 경계의 마지막 관문.

검색 평가(evaluate)·신뢰성 평가(faithfulness)는 **오프라인 품질 측정**이다.
운영에서 LLM 모드의 답변은 매 요청마다 새로 생성되는데, 그 답변 속 수치가
근거와 일치하는지 평가셋은 보증해 주지 않는다. 규제 도메인에서 답변 속
"15일"이 "30일"로 바뀌는 오류는 문장이 아무리 유창해도 컴플라이언스 사고다.

그래서 **모든 응답에 대해 런타임으로 실행되는 결정론적 검증 계층**을 둔다:

  1. 수치 클레임 검증 — 답변에서 '숫자+단위'(15일, 6개월, 120일, 90% …)와
     날짜(YYYY-MM-DD)를 추출해, 각각이 **신뢰 소스(trusted sources)** 안에
     실제로 존재하는지 대조한다.
  2. 인용 버전 검증 — 답변의 출처(citation)에 폐지(superseded)된 문서가
     섞였는지 확인한다(이력 조회를 명시하지 않았다면 그 자체가 결함).

신뢰 소스의 정의가 이 모듈의 핵심 설계다:
  신뢰 소스 = 검색된 근거 문단 ∪ 결정론적 도구의 출력
  근거 문단만 보면 도구가 '계산해 낸' 값(예: 인지일+15일=마감일)이 근거
  원문에 없다는 이유로 오탐된다. 규칙 기반 도구의 출력은 테스트로 검증된
  결정론이므로 근거와 같은 신뢰 등급으로 취급한다.

실패 방향의 설계:
  검증 실패 시 답변을 차단하지 않고 **경고를 부착**한다. 검증기 자신도
  오탐 가능성이 있고(패러프레이즈 등), 이 도구의 원칙은 '사람의 최종
  확정을 빠르게'이지 자동 차단이 아니다. 단, 경고는 조용히 숨기지 않고
  답변 본문·API 필드·UI 배지에 모두 노출한다(시끄러운 실패).

왜 LLM 재검증(self-critique)이 아니라 규칙인가:
  검증자가 확률적이면 '검증의 검증'이 다시 필요해지는 순환이 생긴다.
  결정론적 검증은 같은 답변에 항상 같은 판정을 내려 감사 가능하고,
  그 자체를 평가셋(eval/verify_eval.py)으로 측정할 수 있다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 수치 클레임: 숫자 + 규제 맥락에서 의미가 무거운 단위.
# '주(週)'를 포함하는 이유: LLM이 "15일"을 "약 2주"로 패러프레이즈하면
# 근거에 없는 환산값이 생기고, 마감일 환산 오차는 그 자체가 리스크다 —
# 근거에 없는 단위 환산은 '지원되지 않는 클레임'으로 취급한다.
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(일|개월|년|주(?![가-힣])|시간|회|세|%)")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass
class ClaimCheck:
    claim: str        # 정규화된 클레임 표기 (예: "15일", "2026-07-25")
    kind: str         # "numeric" | "date"
    supported: bool   # 신뢰 소스에서 확인됐는가

    def as_dict(self) -> dict:
        return {"claim": self.claim, "kind": self.kind, "supported": self.supported}


@dataclass
class VerificationResult:
    checks: list[ClaimCheck] = field(default_factory=list)
    superseded_cited: list[str] = field(default_factory=list)  # 폐지본 인용 doc_id

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.checks if not c.supported]

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.superseded_cited

    def summary(self) -> dict:
        """API/UI 노출용 요약."""
        return {
            "ok": self.ok,
            "checked": len(self.checks),
            "unsupported": self.unsupported,
            "superseded_cited": self.superseded_cited,
        }


def extract_claims(text: str) -> tuple[set[tuple[str, str]], set[str]]:
    """텍스트에서 (수치, 단위) 클레임 집합과 날짜 집합을 추출한다.

    수치는 (값, 단위) 튜플로 정규화한다 — "15일 이내"와 "15 일"이 같은
    클레임으로 대조되도록. 값은 소수점 표기 차이를 흡수하기 위해 float 문자열이
    아닌 원문 숫자 문자열을 그대로 쓰되 불필요한 선행 0만 제거한다.
    """
    nums = {
        (m.group(1).lstrip("0") or "0", m.group(2))
        for m in _NUM_UNIT_RE.finditer(text)
    }
    dates = set(_DATE_RE.findall(text))
    return nums, dates


def verify_answer(
    answer: str,
    trusted_texts: list[str],
    citations: list[dict] | None = None,
    allow_superseded: bool = False,
) -> VerificationResult:
    """답변의 수치·날짜 클레임을 신뢰 소스와 대조하고 인용 버전을 점검한다.

    Args:
        answer: 사용자에게 나가는 최종 답변 텍스트.
        trusted_texts: 신뢰 소스 — 검색 근거 문단 + 결정론적 도구 출력의
            직렬화 문자열. 이 밖에서 온 수치는 전부 '미확인'으로 판정한다.
        citations: 답변에 부착된 출처 메타(status 포함 시 버전 점검).
        allow_superseded: 사용자가 명시적으로 이력(폐지본) 조회를 요청한
            경우 True — 폐지본 인용이 결함이 아니라 목적이 된다.
    """
    result = VerificationResult()

    ans_nums, ans_dates = extract_claims(answer)
    trusted = "\n".join(trusted_texts)
    src_nums, src_dates = extract_claims(trusted)

    for num, unit in sorted(ans_nums):
        result.checks.append(
            ClaimCheck(claim=f"{num}{unit}", kind="numeric", supported=(num, unit) in src_nums)
        )
    for d in sorted(ans_dates):
        result.checks.append(ClaimCheck(claim=d, kind="date", supported=d in src_dates))

    if citations and not allow_superseded:
        result.superseded_cited = [
            c.get("doc_id") or c.get("source") or "?"
            for c in citations
            if c.get("status") == "superseded"
        ]
    return result


def warning_text(v: VerificationResult) -> str:
    """검증 실패 시 답변에 부착할 경고문(시끄러운 실패)."""
    parts: list[str] = []
    if v.unsupported:
        parts.append(
            "⚠ 자동 검증 경고: 답변 속 수치 "
            + ", ".join(f"'{c}'" for c in v.unsupported)
            + " 이(가) 검색 근거·도구 결과에서 확인되지 않았습니다. "
            "제출·회신 전 규정 원문 대조가 필요합니다."
        )
    if v.superseded_cited:
        parts.append(
            "⚠ 버전 경고: 폐지(superseded)된 규정 "
            + ", ".join(v.superseded_cited)
            + " 이(가) 출처에 포함되어 있습니다. 현행 규정 기준으로 재확인하세요."
        )
    return "\n".join(parts)
