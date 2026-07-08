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
