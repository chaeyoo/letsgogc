"""이상사례 용어 표준화 코딩 (MedDRA 방식의 경량 사전).

PV 케이스는 "숨쉬기 힘들다", "온몸에 두드러기" 같은 구어 서술(verbatim)로
들어오지만, 규제 보고(KAERS/E2B)는 표준 용어(MedDRA PT: Preferred Term)로
코딩해야 한다. 코딩이 흔들리면 같은 이상사례가 다른 용어로 흩어져
시그널 탐지(집계)가 무너진다 — PV 데이터 품질의 출발점이 코딩이다.

구현 원칙:
  - 실제 MedDRA는 유료 라이선스 용어집(수만 개 PT)이라 데모에 담을 수 없다.
    여기서는 '구어 → PT(+SOC 기관계 분류)' 매핑 구조와 코딩 파이프라인의
    자리를 소사전으로 증명하고, 실무에선 사전만 MedDRA 본체로 교체한다.
  - 질의확장 사전(rag/synonyms.py)과 같은 철학: 결정론적(감사 가능)·저지연.
    사전이 못 잡는 롱테일 표현은 LLM 보조 코딩 후보 제시 → 사람 확정으로 확장.
  - 같은 PT가 여러 표현으로 감지되면 1건으로 dedupe한다(집계 왜곡 방지).
"""
from __future__ import annotations

from dataclasses import dataclass

# (PT 한글, PT 영문, SOC 기관계, 감지 키워드) — MedDRA 스타일 소사전.
# 키워드는 케이스 자유 서술에서 해당 PT를 감지하는 데 쓴다(부분 일치).
_TERM_DICT: list[tuple[str, str, str, list[str]]] = [
    ("아나필락시스 반응", "Anaphylactic reaction", "면역계 장애", ["아나필락시스"]),
    ("두드러기", "Urticaria", "피부 및 피하조직 장애", ["두드러기", "담마진"]),
    ("발진", "Rash", "피부 및 피하조직 장애", ["발진"]),
    ("소양증", "Pruritus", "피부 및 피하조직 장애", ["가려움", "소양감", "가려워"]),
    ("호흡곤란", "Dyspnoea", "호흡기, 흉곽 및 종격 장애", ["호흡곤란", "숨쉬기 힘", "숨이 차", "숨을 쉬기 어려"]),
    ("두통", "Headache", "신경계 장애", ["두통", "머리가 아프", "머리가 아파"]),
    ("어지러움", "Dizziness", "신경계 장애", ["어지러움", "어지럼", "현기증", "어지러워"]),
    ("실신", "Syncope", "신경계 장애", ["실신", "의식을 잃", "의식 소실"]),
    ("경련", "Seizure", "신경계 장애", ["경련", "발작"]),
    ("오심", "Nausea", "위장관 장애", ["오심", "메스꺼움", "메스꺼워", "메스껍", "구역감"]),
    ("구토", "Vomiting", "위장관 장애", ["구토", "토했", "토를 하"]),
    ("설사", "Diarrhoea", "위장관 장애", ["설사"]),
    ("발열", "Pyrexia", "전신 장애 및 투여부위 병태", ["발열", "고열", "열이 나", "열이 났"]),
    ("안면부종", "Face oedema", "전신 장애 및 투여부위 병태", ["얼굴이 붓", "안면부종", "얼굴 부종", "입술이 붓"]),
    ("저혈압", "Hypotension", "혈관 장애", ["저혈압", "혈압이 떨어", "혈압 저하"]),
    ("심정지", "Cardiac arrest", "심장 장애", ["심정지"]),
    ("간효소 상승", "Hepatic enzyme increased", "임상검사", ["간수치 상승", "간수치가 올", "ast 상승", "alt 상승", "간효소"]),
    ("간손상", "Liver injury", "간담도 장애", ["간손상", "간독성", "황달"]),
]


@dataclass
class CodedTerm:
    verbatim: str      # 서술에서 감지된 원 표현(첫 매칭 키워드)
    pt: str            # 표준 용어(PT) 한글
    pt_en: str         # PT 영문 (KAERS/E2B 국제 보고용)
    soc: str           # 기관계 대분류(SOC)

    def as_dict(self) -> dict:
        return {"verbatim": self.verbatim, "pt": self.pt, "pt_en": self.pt_en, "soc": self.soc}


def code_terms(case_text: str) -> list[CodedTerm]:
    """케이스 서술에서 이상사례 표현을 감지해 표준 용어(PT)로 코딩한다.

    같은 PT는 1건으로 dedupe하고, 서술에 등장한 순서를 유지한다
    (보고서의 '주요 이상사례'가 서술 흐름과 일치하게).
    """
    low = case_text.lower()
    hits: list[tuple[int, CodedTerm]] = []
    seen_pt: set[str] = set()
    for pt, pt_en, soc, keywords in _TERM_DICT:
        if pt in seen_pt:
            continue
        positions = [(low.find(k.lower()), k) for k in keywords if k.lower() in low]
        if not positions:
            continue
        pos, keyword = min(positions)
        seen_pt.add(pt)
        hits.append((pos, CodedTerm(verbatim=keyword, pt=pt, pt_en=pt_en, soc=soc)))
    hits.sort(key=lambda h: h[0])
    return [t for _, t in hits]
