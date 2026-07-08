"""pytest 공용 fixture. 무거운 인덱스 구축은 세션 1회로 공유."""
from __future__ import annotations

import pytest

from src.rag.pipeline import RagPipeline


@pytest.fixture(scope="session")
def pipeline() -> RagPipeline:
    return RagPipeline().build()
