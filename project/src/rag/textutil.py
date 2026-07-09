"""텍스트 토크나이징 유틸.

한국어는 조사·띄어쓰기 변형이 많아 단어 토큰만으로는 검색 매칭이 약하다.
그래서 (1) 단어/영숫자 토큰 + (2) CJK 문자 bi-gram 을 함께 사용해
'감기약' ↔ '감기' 같은 부분 매칭을 보강한다. (데모용 경량 형태소 근사)
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(r"[가-힣一-鿿]+")

# 검색에 노이즈가 되는 흔한 한국어 조사/불용어(경량)
_STOP = {
    "및", "등", "의", "를", "을", "은", "는", "이", "가", "에", "에서", "으로",
    "로", "와", "과", "그", "것", "수", "때", "및는", "하는", "한다", "된다",
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "is", "are",
}


def tokenize(text: str) -> list[str]:
    """텍스트를 검색용 토큰 리스트로 변환."""
    text = text.lower()
    tokens: list[str] = []

    # 영문/숫자 단어
    for w in _WORD_RE.findall(text):
        if w not in _STOP and len(w) > 1:
            tokens.append(w)

    # CJK: 어절 + 문자 bi-gram
    for run in _CJK_RE.findall(text):
        if run not in _STOP and len(run) >= 2:
            tokens.append(run)                       # 어절 통째
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])              # 문자 bi-gram

    return tokens
