"""프로젝트 전역 설정.

환경변수로 동작 모드를 제어한다.
- ANTHROPIC_API_KEY 가 있으면 실제 Claude(Enterprise LLM API)로 에이전트가 동작한다.
- 없으면 오프라인 모드로 폴백한다(검색 근거 기반 추출형 답변). 데모는 키 없이도 항상 실행된다.
"""
from __future__ import annotations

import os
from pathlib import Path

# 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REG_DIR = DATA_DIR / "regulations"
RA_TASKS_FILE = DATA_DIR / "ra_tasks.json"
WEB_DIR = BASE_DIR / "web"

# LLM (Enterprise LLM API) 설정 — 있으면 사용, 없으면 오프라인 폴백
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-4-8")
LLM_AVAILABLE = bool(ANTHROPIC_API_KEY)

# RAG 하이퍼파라미터 (RAG '최적화'의 손잡이들)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))       # 청크 크기(문자)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))  # 겹침(경계 손실 방지)
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "8"))  # 1차 회수 개수
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "3"))     # 리랭킹 후 최종 개수
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))  # 벡터(TF-IDF) vs 키워드(BM25) 가중
RERANK_WEIGHT = float(os.environ.get("RERANK_WEIGHT", "0.9"))  # 리랭커 신호 vs 1차점수 prior 결합
RERANK_IDF_POWER = float(os.environ.get("RERANK_IDF_POWER", "0.5"))  # 리랭커 토큰 가중 idf^p (0=균등)
# 섹션 타입 prior (리랭커 v3): 질의 의도로 게이트되는 구조 신호.
#  - contrast: "X와의 차이/구분(주의)" 대조 섹션 페널티(질의가 비교를 묻지 않을 때만)
#  - preamble: "목적/개요/총칙" 서두 섹션 감쇠(질의가 정의/취지를 묻지 않을 때만)
# 크기 근거는 eval/sweep.py 의 스윕 참고(contrast 는 0.25 이상에서 플랫,
# preamble 은 0.04~0.07 유효 밴드의 중앙).
RERANK_CONTRAST_PENALTY = float(os.environ.get("RERANK_CONTRAST_PENALTY", "0.3"))
RERANK_PREAMBLE_PENALTY = float(os.environ.get("RERANK_PREAMBLE_PENALTY", "0.055"))
EMBEDDER_KIND = os.environ.get("EMBEDDER_KIND", "tfidf")     # tfidf | hashing | voyage
QUERY_EXPANSION = os.environ.get("QUERY_EXPANSION", "1") not in ("0", "false", "off")
# 도메인 동의어 질의 확장(부작용→이상사례 등). 1단계 회수에만 적용.


def mode_banner() -> str:
    """현재 실행 모드를 한 줄로 반환(로그/헬스체크용)."""
    if LLM_AVAILABLE:
        return f"[LLM 모드] Enterprise LLM API 사용 · model={LLM_MODEL}"
    return "[오프라인 모드] API 키 없음 → 검색 근거 기반 추출형 답변으로 폴백"
