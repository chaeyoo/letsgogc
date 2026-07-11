"""경량 관측성(observability) — 스텝별 지연·에러를 구조화 로그로 남긴다.

엔터프라이즈 에이전트에서 '무슨 도구를 몇 ms에, 성공/실패로 호출했는가'는
디버깅·SLA·비용 관리의 기본이다. 실무에선 이 자리에 LangSmith/OpenTelemetry를
끼우지만, 여기서는 의존성 없이 동일한 개념(span·trace)을 순수 파이썬으로 구현한다.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("ra_assistant")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


@dataclass
class Span:
    """단일 작업 구간(도구 호출·LLM 스텝·검색 등)."""
    name: str
    kind: str            # tool | llm | pipeline | agent
    duration_ms: float
    ok: bool = True
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "ms": round(self.duration_ms, 2),
            "ok": self.ok,
            **self.detail,
        }


@dataclass
class Trace:
    """한 요청의 span 모음(에이전트 실행 트레이스)."""
    spans: list[Span] = field(default_factory=list)

    def add(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def total_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 2)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]


@dataclass
class GateStats:
    """검증 게이트 운영 계기판 — 프로세스 수명 동안의 경고율을 상시 집계한다.

    검증 계층의 운영 리스크는 오탐 그 자체보다 **경고율의 추이**다: 경고가
    잦아지면 담당자가 경고를 무시하기 시작하고(alert fatigue) 그 순간 계층
    전체가 죽는다. 배치 후 이 신호를 볼 수단이 없으면 죽는 순간을 모른 채
    지나간다 — 그래서 게이트 통과/경고를 축별로 세어 /health 에 노출한다
    (FDE가 배치 다음 날 아침에 확인하는 계기판). 집계는 응답 단위다:
    한 응답에 미확인 수치가 3개여도 unsupported 축 +1 — 계기판이 재는 것은
    '경고가 붙은 응답의 비율'이지 클레임 개수가 아니다.

    검증 결과는 응답 단위 감사 로그(JSON)로도 남는다 — 규제 도메인에서
    "그 답변이 그때 검증을 통과했는가"는 사후 감사의 질문이기 때문이다.
    답변 원문이 아니라 판정 요약만 남긴다(로그에 클레임 수치·판정만, PII 없음).
    """
    responses: int = 0
    warned: int = 0
    by_axis: dict[str, int] = field(default_factory=dict)

    _AXES = ("unsupported", "direction_conflicts", "role_conflicts", "question_origin", "superseded_cited")

    def record(self, summary: dict) -> None:
        self.responses += 1
        if not summary.get("ok", True):
            self.warned += 1
        for axis in self._AXES:
            if summary.get(axis):
                self.by_axis[axis] = self.by_axis.get(axis, 0) + 1

    def snapshot(self) -> dict:
        return {
            "responses": self.responses,
            "warned": self.warned,
            "warn_rate": round(self.warned / self.responses, 4) if self.responses else 0.0,
            "by_axis": dict(self.by_axis),
        }

    def reset(self) -> None:
        self.responses = 0
        self.warned = 0
        self.by_axis = {}


gate_stats = GateStats()


def record_verification(summary: dict) -> None:
    """검증 게이트 결과 1건을 계기판에 집계하고 감사 로그를 남긴다."""
    gate_stats.record(summary)
    logger.info(
        "verification "
        + json.dumps(
            {
                "ok": summary.get("ok"),
                "checked": summary.get("checked"),
                "unsupported": summary.get("unsupported"),
                "direction_conflicts": summary.get("direction_conflicts"),
                "role_conflicts": summary.get("role_conflicts"),
                "question_origin": summary.get("question_origin"),
                "superseded_cited": summary.get("superseded_cited"),
            },
            ensure_ascii=False,
        )
    )


@contextmanager
def timed(trace: Trace, name: str, kind: str, detail: dict | None = None):
    """with 블록의 실행시간을 측정해 trace에 span으로 기록하고 구조화 로그를 남긴다.

    블록에서 예외가 나도 span(ok=False)을 남긴 뒤 예외를 다시 던진다.
    """
    t0 = time.perf_counter()
    ok = True
    err_name = ""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - 관측 목적상 모든 예외 기록 후 재전파
        ok = False
        err_name = type(e).__name__
        raise
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        d = dict(detail or {})
        if err_name:
            d["error"] = err_name
        trace.add(Span(name=name, kind=kind, duration_ms=dt, ok=ok, detail=d))
        logger.info(
            "span " + json.dumps({"name": name, "kind": kind, "ms": round(dt, 2), "ok": ok, **d}, ensure_ascii=False)
        )
