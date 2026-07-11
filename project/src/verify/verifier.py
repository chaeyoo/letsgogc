"""답변 사후 검증(post-generation verification) — 생성 경계의 마지막 관문.

검색 평가(evaluate)·신뢰성 평가(faithfulness)는 **오프라인 품질 측정**이다.
운영에서 LLM 모드의 답변은 매 요청마다 새로 생성되는데, 그 답변 속 수치가
근거와 일치하는지 평가셋은 보증해 주지 않는다. 규제 도메인에서 답변 속
"15일"이 "30일"로 바뀌는 오류는 문장이 아무리 유창해도 컴플라이언스 사고다.

그래서 **모든 응답에 대해 런타임으로 실행되는 결정론적 검증 계층**을 둔다:

  1. 수치 클레임 검증 — 답변에서 '숫자+단위'(15일, 6개월, 120 근무일, 90% …),
     고유어 수량 표현(보름, 이틀, 한 달 …), 날짜(YYYY-MM-DD)를 추출해,
     각각이 **신뢰 소스(trusted sources)** 안에 실제로 존재하는지 대조한다.
  2. 방향 한정어 검증 — 수치가 근거에 있어도 **한정어의 방향이 뒤집히면**
     ("15일 이내" → "15일 이후") 별도 경고를 낸다. 방향 한정어는
     닫힌 어휘 집합(이내·이하·미만·까지 / 이상·이후·초과)이라 기계 검증이
     가능하다 — '관계 왜곡은 전부 LLM judge 몫'이라는 초기 경계 설정을
     재심사해 결정론으로 끌어온 부분이다.
  3. 날짜 역할 검증 — 도구 출력에 날짜가 여러 개면(인지일·마감일·오늘)
     답변이 두 날짜의 **역할을 맞바꿔도**("보고 기한은 <인지일>입니다") 각
     날짜가 신뢰 소스에 존재하므로 존재 대조(1)는 통과한다. 결정론적 도구는
     날짜를 역할 키(deadline_date·due_date·awareness_date)로 라벨링해
     출력하므로, 답변에서 역할 키워드(기한/마감·인지일)에 직접 붙은 날짜를
     그 역할의 라벨 집합과 대조한다 — 방향 한정어와 같은 원리로, 닫힌
     키워드·라벨 기반이라 기계 검증이 가능한 축이다.
  4. 인용 버전 검증 — 답변의 출처(citation)에 폐지(superseded)된 문서가
     섞였는지 확인한다(이력 조회를 명시하지 않았다면 그 자체가 결함).

신뢰 소스의 정의가 이 모듈의 핵심 설계다:
  신뢰 소스 = 검색된 근거 문단 ∪ 결정론적 도구의 출력 − 질문 에코
  근거 문단만 보면 도구가 '계산해 낸' 값(예: 인지일+15일=마감일)이 근거
  원문에 없다는 이유로 오탐된다. 규칙 기반 도구의 출력은 테스트로 검증된
  결정론이므로 근거와 같은 신뢰 등급으로 취급한다.
  반대로 도구 출력에 에코된 **사용자 질의(query)는 신뢰 소스에서 제외**한다
  — 포함하면 사용자가 틀린 수치를 전제로 물었을 때 모델이 맞장구쳐도
  통과하는 구멍이 생긴다(전제의 승격). 대신 질문에 있던 수치가 미확인으로
  판정되면 `from_question` 라벨로 구분해, 경고 문구가 '환각'이 아니라
  '전제 확인 필요'를 가리키게 한다(부정·정정 맥락의 오탐 완화).

단위의 엄격성:
  '근무일'과 '일'은 **다른 단위**다 — "120 근무일"을 "120일"로 옮기면 실제
  달력 기한이 달라진다. '주(週)' 환산("15일"→"약 2주")과 마찬가지로, 근거에
  없는 단위 환산은 '지원되지 않는 클레임'으로 취급한다. 고유어 수사는 값이
  정확히 같은 표기 변형(보름=15일)만 사전으로 동치 처리한다 — 환산이 아니라
  표기 정규화이므로 엄격성과 충돌하지 않는다.

실패 방향의 설계:
  검증 실패 시 답변을 차단하지 않고 **경고를 부착**한다. 검증기 자신도
  오탐 가능성이 있고(패러프레이즈 등), 이 도구의 원칙은 '사람의 최종
  확정을 빠르게'이지 자동 차단이 아니다. 단, 경고는 조용히 숨기지 않고
  답변 본문·API 필드·UI 배지에 모두 노출한다(시끄러운 실패). 지원된
  클레임에는 근거 위치 스니펫(evidence)을 붙여 사람이 즉시 대조하게 한다.

왜 LLM 재검증(self-critique)이 아니라 규칙인가:
  검증자가 확률적이면 '검증의 검증'이 다시 필요해지는 순환이 생긴다.
  결정론적 검증은 같은 답변에 항상 같은 판정을 내려 감사 가능하고,
  그 자체를 평가셋(eval/verify_eval.py)으로 측정할 수 있다.

이 계층이 잡지 **않는** 것(경계의 명시):
  부정문("30일이 아니라 15일"), 주어 바꿔치기, 수치 없는 의미 왜곡은
  결정론 규칙의 범위 밖이다 — LLM judge 2차 검증의 자리로 남긴다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 단위 목록: 긴 단위가 먼저 와야 한다 ('근무일'이 '일'보다 먼저 매칭되도록).
# '근무일'을 별도 단위로 두는 이유: 코퍼스의 핵심 처리기한(120·75·60 근무일…)이
# '일'만 인식하는 정규식에는 아예 안 잡혀 검증 사각지대였다 — 그리고
# 근무일→역일 환산은 실제 달력 기한을 바꾸는 오류라 단위를 구분해 대조해야 한다.
# '주(週)'의 lookahead: LLM이 "15일"을 "약 2주"로 패러프레이즈하면 근거에 없는
# 환산값이 생기고, 마감일 환산 오차는 그 자체가 리스크다.
_UNIT_ALT = r"근무일|영업일|개월|주일|시간|일|년|주(?![가-힣])|회|세|%"
_NUM_UNIT_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# 범위 표기("15~30일")의 하한 — 주 정규식은 상한(30일)만 잡아 하한이 검증을
# 벗어난다. 구분자에 '-'를 넣지 않는 이유: 날짜(2026-07-25)와 충돌한다.
_RANGE_RE = re.compile(rf"(?<![\d.\-])(\d+(?:\.\d+)?)\s*[~∼〜–—]\s*\d+(?:\.\d+)?\s*({_UNIT_ALT})")

# 고유어 수량 표현 → 허용 canonical (값, 단위) 형태들.
# '환산'이 아니라 값이 정확히 같은 '표기 변형'만 담는다(보름=15일).
# 사전이 보수적인 이유: 오탐 없는 어휘만 넣는다 — '하루'는 "하루빨리" 같은
# 관용구 오탐 위험이 있어 제외(코딩 사전과 같은 철학: 확신 없는 항목은 안 넣는다).
_NATIVE_NUMERALS: dict[str, tuple[tuple[str, str], ...]] = {
    "이틀": (("2", "일"),),
    "사흘": (("3", "일"),),
    "나흘": (("4", "일"),),
    "닷새": (("5", "일"),),
    "열흘": (("10", "일"),),
    "보름": (("15", "일"),),
    "일주일": (("7", "일"), ("1", "주")),
    "한 달": (("1", "개월"),),
    "한달": (("1", "개월"),),
    "두 달": (("2", "개월"),),
    "두달": (("2", "개월"),),
    "석 달": (("3", "개월"),),
    "세 달": (("3", "개월"),),
    "반년": (("6", "개월"),),
    "반 년": (("6", "개월"),),
}
_NATIVE_RE = re.compile("|".join(sorted((re.escape(w) for w in _NATIVE_NUMERALS), key=len, reverse=True)))

# 방향 한정어 — 닫힌 어휘 집합이라 결정론적 대조가 가능하다.
# 수치+단위 바로 뒤(마크다운 강조 등 브리지 문자 허용)에 붙은 경우만 본다.
_UPPER_WORDS = ("이내", "안에", "이하", "미만", "까지")
_LOWER_WORDS = ("이상", "이후", "초과", "경과")
_QUAL_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})[\s*_)\]】'\"”]*({'|'.join(_UPPER_WORDS + _LOWER_WORDS)})"
)
_OPPOSITE = {"상한": "하한", "하한": "상한"}

# 날짜 역할 대조 — 결정론적 도구의 역할 라벨(직렬화된 JSON 키) ↔ 답변의 역할 키워드.
# 답변 쪽은 키워드가 날짜에 '직접' 붙은 경우만 본다("기한: 2026-07-25",
# "보고 기한은 2026-07-25", "(인지일 2026-07-10") — 사이에 다른 단어가 끼면
# ("기한 규정은 2025-04-01 시행…") 역할 주장으로 보지 않는다. 보수성:
# 신뢰 소스에 해당 역할 라벨이 없으면 판단 근거가 없으므로 플래그하지 않고,
# 존재 대조를 통과한 날짜만 본다(미확인 날짜는 존재 대조 축이 먼저 잡는다).
_ROLE_BRIDGE = r"[은는이가]?\s*[:：]?[\s*_\"'(（]*"
_ROLE_LABEL_RE: dict[str, re.Pattern[str]] = {
    "기한": re.compile(r'"(?:deadline_date|due_date)"\s*:\s*"(\d{4}-\d{2}-\d{2})"'),
    "인지일": re.compile(r'"awareness_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"'),
}
_ROLE_ANSWER_RE: dict[str, re.Pattern[str]] = {
    "기한": re.compile(rf"(?:기한|마감)일?{_ROLE_BRIDGE}(\d{{4}}-\d{{2}}-\d{{2}})"),
    "인지일": re.compile(rf"인지일{_ROLE_BRIDGE}(\d{{4}}-\d{{2}}-\d{{2}})"),
}


def _qual_class(word: str) -> str:
    return "상한" if word in _UPPER_WORDS else "하한"


def _normalize(text: str) -> str:
    """추출 전 정규화 — 천단위 콤마 제거("1,000회"가 "000회"로 오추출되는 것 방지)."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


@dataclass
class ClaimCheck:
    claim: str        # 정규화된 클레임 표기 (예: "15일", "보름", "2026-07-25", "15일 이후")
    kind: str         # "numeric" | "date" | "direction" | "role"
    supported: bool   # 신뢰 소스에서 확인됐는가 (direction/role 은 항상 False=충돌)
    evidence: str = ""        # 지원 시 신뢰 소스의 해당 위치 스니펫(사람 대조용)
    from_question: bool = False  # 미확인 수치가 사용자 질문에 있던 값인가(전제 에코)

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "kind": self.kind,
            "supported": self.supported,
            "evidence": self.evidence,
            "from_question": self.from_question,
        }


@dataclass
class VerificationResult:
    checks: list[ClaimCheck] = field(default_factory=list)
    superseded_cited: list[str] = field(default_factory=list)  # 폐지본 인용 doc_id

    @property
    def unsupported(self) -> list[str]:
        """근거에서 확인되지 않은 수치·날짜 클레임(방향·역할 충돌은 별도 축)."""
        return [c.claim for c in self.checks if not c.supported and c.kind in ("numeric", "date")]

    @property
    def direction_conflicts(self) -> list[str]:
        """수치는 근거에 있으나 한정어 방향이 뒤집힌 클레임."""
        return [c.claim for c in self.checks if c.kind == "direction"]

    @property
    def role_conflicts(self) -> list[str]:
        """날짜는 근거에 있으나 역할(기한↔인지일)이 도구 라벨과 어긋난 클레임."""
        return [c.claim for c in self.checks if c.kind == "role"]

    @property
    def question_origin(self) -> list[str]:
        """미확인 클레임 중 사용자 질문에 있던 값(전제 에코 — 경고 문구를 달리 한다)."""
        return [c.claim for c in self.checks if not c.supported and c.from_question]

    @property
    def ok(self) -> bool:
        return (
            not self.unsupported
            and not self.direction_conflicts
            and not self.role_conflicts
            and not self.superseded_cited
        )

    def summary(self) -> dict:
        """API/UI 노출용 요약."""
        return {
            "ok": self.ok,
            "checked": len([c for c in self.checks if c.kind in ("numeric", "date")]),
            "unsupported": self.unsupported,
            "direction_conflicts": self.direction_conflicts,
            "role_conflicts": self.role_conflicts,
            "question_origin": self.question_origin,
            "superseded_cited": self.superseded_cited,
            "checks": [c.as_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class _Occurrence:
    display: str                          # 답변 속 원문 표기
    forms: tuple[tuple[str, str], ...]    # 허용 canonical (값, 단위) — 고유어는 복수 가능
    kind: str                             # "numeric" | "date"


def _numeric_forms(text: str) -> set[tuple[str, str]]:
    """텍스트의 모든 수치 클레임을 canonical (값, 단위) 집합으로.

    수치는 (값, 단위) 튜플로 정규화한다 — "15일 이내"와 "15 일"이 같은
    클레임으로 대조되도록. 값은 불필요한 선행 0만 제거한다("03년"="3년").
    '주일' 단위는 '주'로 정규화한다("2주일"="2주").
    고유어 수사(보름 등)는 canonical 형태 전부를 집합에 더한다 — 추출이
    답변·신뢰 소스 양쪽에 대칭으로 적용되므로, 근거가 "보름"이라 쓰고
    답변이 "15일"이라 써도(또는 그 반대) 동치로 대조된다.
    """
    text = _normalize(text)
    forms = {
        (m.group(1).lstrip("0") or "0", "주" if m.group(2) == "주일" else m.group(2))
        for m in _NUM_UNIT_RE.finditer(text)
    }
    for m in _RANGE_RE.finditer(text):  # 범위 하한
        unit = "주" if m.group(2) == "주일" else m.group(2)
        forms.add((m.group(1).lstrip("0") or "0", unit))
    for m in _NATIVE_RE.finditer(text):
        forms |= set(_NATIVE_NUMERALS[m.group(0)])
    return forms


def _qualifier_map(text: str) -> dict[tuple[str, str], set[str]]:
    """(값, 단위) → 그 수치에 붙어 등장한 방향 한정어 클래스 집합."""
    text = _normalize(text)
    out: dict[tuple[str, str], set[str]] = {}
    for m in _QUAL_RE.finditer(text):
        unit = "주" if m.group(2) == "주일" else m.group(2)
        key = (m.group(1).lstrip("0") or "0", unit)
        out.setdefault(key, set()).add(_qual_class(m.group(3)))
    return out


def extract_claims(text: str) -> tuple[set[tuple[str, str]], set[str]]:
    """텍스트에서 (수치, 단위) 클레임 집합과 날짜 집합을 추출한다."""
    return _numeric_forms(text), set(_DATE_RE.findall(_normalize(text)))


def _occurrences(answer: str) -> list[_Occurrence]:
    """답변 쪽 추출 — 표기(display) 단위로 클레임을 나열한다(경고 문구용)."""
    answer = _normalize(answer)
    occs: list[_Occurrence] = []
    seen: set[str] = set()

    def _add(display: str, forms: tuple[tuple[str, str], ...], kind: str) -> None:
        if display not in seen:
            seen.add(display)
            occs.append(_Occurrence(display, forms, kind))

    for m in _NUM_UNIT_RE.finditer(answer):
        num = m.group(1).lstrip("0") or "0"
        unit = "주" if m.group(2) == "주일" else m.group(2)
        _add(f"{num}{unit}", ((num, unit),), "numeric")
    for m in _RANGE_RE.finditer(answer):
        num = m.group(1).lstrip("0") or "0"
        unit = "주" if m.group(2) == "주일" else m.group(2)
        _add(f"{num}{unit}", ((num, unit),), "numeric")
    for m in _NATIVE_RE.finditer(answer):
        _add(m.group(0), _NATIVE_NUMERALS[m.group(0)], "numeric")
    for d in _DATE_RE.findall(answer):
        _add(d, ((d, ""),), "date")
    return occs


def _snippet(trusted: str, form: tuple[str, str], kind: str, width: int = 34) -> str:
    """신뢰 소스에서 클레임이 등장한 위치의 스니펫 — 사람의 대조를 빠르게."""
    if kind == "date":
        pat = re.escape(form[0])
    else:
        num, unit = form
        unit_pat = "주일?" if unit == "주" else re.escape(unit)
        pat = rf"(?<!\d)0*{re.escape(num)}\s*{unit_pat}"
    m = re.search(pat, trusted)
    if not m:  # 고유어 표기로 지원된 경우
        for word, canon in _NATIVE_NUMERALS.items():
            if form in canon:
                m = re.search(re.escape(word), trusted)
                if m:
                    break
    if not m:
        return ""
    lo, hi = max(0, m.start() - width), min(len(trusted), m.end() + width)
    return " ".join(trusted[lo:hi].split())


def verify_answer(
    answer: str,
    trusted_texts: list[str],
    citations: list[dict] | None = None,
    allow_superseded: bool = False,
    question: str = "",
) -> VerificationResult:
    """답변의 수치·날짜 클레임과 방향 한정어를 신뢰 소스와 대조하고 인용 버전을 점검한다.

    Args:
        answer: 사용자에게 나가는 최종 답변 텍스트.
        trusted_texts: 신뢰 소스 — 검색 근거 문단 + 결정론적 도구 출력의
            직렬화 문자열(질문 에코 필드는 호출자가 제거). 이 밖에서 온
            수치는 전부 '미확인'으로 판정한다.
        citations: 답변에 부착된 출처 메타(status 포함 시 버전 점검).
        allow_superseded: 사용자가 명시적으로 이력(폐지본) 조회를 요청한
            경우 True — 폐지본 인용이 결함이 아니라 목적이 된다.
        question: 사용자 질문 원문. 미확인 수치가 질문에 있던 값이면
            from_question 으로 라벨링해 경고 문구를 '전제 확인'으로 조정한다
            (부정·정정 답변의 오탐 완화 — 신뢰하지도, 조용히 넘기지도 않는다).
    """
    result = VerificationResult()

    trusted = _normalize("\n".join(trusted_texts))
    src_nums, src_dates = extract_claims(trusted)
    src_quals = _qualifier_map(trusted)
    q_nums, q_dates = extract_claims(question) if question else (set(), set())

    for occ in _occurrences(answer):
        if occ.kind == "date":
            supported = occ.display in src_dates
            from_q = (not supported) and occ.display in q_dates
            evidence = _snippet(trusted, (occ.display, ""), "date") if supported else ""
        else:
            hit = next((f for f in occ.forms if f in src_nums), None)
            supported = hit is not None
            from_q = (not supported) and any(f in q_nums for f in occ.forms)
            evidence = _snippet(trusted, hit, "numeric") if hit else ""
        result.checks.append(
            ClaimCheck(occ.display, occ.kind, supported, evidence=evidence, from_question=from_q)
        )

    # 방향 한정어 대조 — 수치가 지원된 클레임에 한해, 답변의 한정어 방향이
    # 신뢰 소스와 뒤집혔는지 본다. 신뢰 소스에 한정어 없이 값만 있으면
    # 판단 근거가 없으므로 플래그하지 않는다(보수적 — 오탐 방지).
    seen_dir: set[str] = set()
    for m in _QUAL_RE.finditer(_normalize(answer)):
        unit = "주" if m.group(2) == "주일" else m.group(2)
        key = (m.group(1).lstrip("0") or "0", unit)
        if key not in src_nums:
            continue  # 값 자체가 미확인 — 위에서 이미 unsupported 로 잡혔다
        cls = _qual_class(m.group(3))
        trusted_cls = src_quals.get(key, set())
        if cls not in trusted_cls and _OPPOSITE[cls] in trusted_cls:
            display = f"{key[0]}{key[1]} {m.group(3)}"
            if display in seen_dir:
                continue
            seen_dir.add(display)
            result.checks.append(
                ClaimCheck(display, "direction", False, evidence=_snippet(trusted, key, "numeric"))
            )

    # 날짜 역할 대조 — 존재 대조를 통과한 날짜에 한해, 답변이 그 날짜에 부여한
    # 역할(기한/인지일)이 결정론적 도구의 역할 라벨과 일치하는지 본다.
    # 신뢰 소스에 해당 역할 라벨이 없으면(검색 근거만 있는 경우 등) 판단
    # 근거가 없으므로 플래그하지 않는다(보수적 — 오탐 방지).
    norm_answer = _normalize(answer)
    seen_role: set[str] = set()
    for role, answer_re in _ROLE_ANSWER_RE.items():
        labels = set(_ROLE_LABEL_RE[role].findall(trusted))
        if not labels:
            continue
        for m in answer_re.finditer(norm_answer):
            d = m.group(1)
            if d in labels or d not in src_dates:
                continue  # 역할 일치, 또는 미확인 날짜(존재 대조 축이 이미 잡았다)
            display = f"{role} {d}"
            if display in seen_role:
                continue
            seen_role.add(display)
            result.checks.append(
                ClaimCheck(display, "role", False, evidence=", ".join(sorted(labels)))
            )

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
    q_origin = set(v.question_origin)
    hallucinated = [c for c in v.unsupported if c not in q_origin]
    if hallucinated:
        parts.append(
            "⚠ 자동 검증 경고: 답변 속 수치 "
            + ", ".join(f"'{c}'" for c in hallucinated)
            + " 이(가) 검색 근거·도구 결과에서 확인되지 않았습니다. "
            "제출·회신 전 규정 원문 대조가 필요합니다."
        )
    if q_origin:
        parts.append(
            "⚠ 전제 확인 필요: 수치 "
            + ", ".join(f"'{c}'" for c in sorted(q_origin))
            + " 은(는) 질문에 포함되어 있던 값으로, 검색 근거에서는 확인되지 않았습니다. "
            "답변이 이를 정정하는 맥락인지 포함해 질문의 전제 자체를 규정 원문과 대조하세요."
        )
    for c in (x for x in v.checks if x.kind == "direction"):
        parts.append(
            f"⚠ 방향 한정어 경고: 답변의 '{c.claim}' 은(는) 근거와 방향이 반대입니다"
            + (f" (근거: \"…{c.evidence}…\")" if c.evidence else "")
            + ". 기한·범위의 방향이 뒤집히면 수치가 맞아도 컴플라이언스 오류입니다."
        )
    for c in (x for x in v.checks if x.kind == "role"):
        role, date = c.claim.split(" ", 1)
        parts.append(
            f"⚠ 날짜 역할 경고: 답변이 '{role}'(으)로 제시한 {date} 은(는) 근거에 존재하지만, "
            f"도구가 해당 역할로 계산한 날짜({c.evidence})와 다릅니다. "
            "날짜들의 역할(인지일↔마감일)이 서로 뒤바뀌지 않았는지 대조하세요."
        )
    if v.superseded_cited:
        parts.append(
            "⚠ 버전 경고: 폐지(superseded)된 규정 "
            + ", ".join(v.superseded_cited)
            + " 이(가) 출처에 포함되어 있습니다. 현행 규정 기준으로 재확인하세요."
        )
    return "\n".join(parts)
