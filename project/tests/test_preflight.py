"""배포 전 점검(src/preflight) 테스트 — 게이트가 실제 결함을 잡는지.

preflight 는 '부팅은 통과하지만 운영 중에 오답으로 나타나는' 데이터·설정
결함을 기동 전에 차단하는 FDE 게이트다. 여기서는 두 방향을 모두 고정한다:
(1) 리포의 실제 데이터·기본 설정은 통과해야 하고(오탐 감시 — run.sh/CI가
이 게이트 뒤에 있으므로 오탐은 곧 배포 불능), (2) 대표 결함(메타 누락·
doc_id 충돌·폐지 체인 단절·날짜 오타·모순 설정)은 잡혀야 한다.
"""
from __future__ import annotations

from pathlib import Path

from src import config, preflight


def test_repo_data_passes_preflight():
    """리포 상태 그대로는 전 그룹 통과 — 게이트 오탐은 곧 배포 불능이다."""
    report = preflight.run_preflight()
    assert all(not problems for problems in report.values()), report


def _write_doc(dirpath: Path, name: str, meta: dict, body: str = "본문 내용") -> None:
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items())
    (dirpath / name).write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")


def test_corpus_check_catches_meta_defects(tmp_path):
    _write_doc(tmp_path, "a.md", {"doc_id": "R-1", "title": "t"})  # version·effective_date 누락
    _write_doc(tmp_path, "b.md", {"doc_id": "R-2", "title": "t", "version": "1.0",
                                  "effective_date": "2025-13-01"})  # 존재하지 않는 달
    _write_doc(tmp_path, "c.md", {"doc_id": "R-2", "title": "t", "version": "1.1",
                                  "effective_date": "2025-01-01"})  # doc_id 중복
    text = "\n".join(preflight.check_corpus(tmp_path))
    assert "필수 필드 누락" in text
    assert "YYYY-MM-DD" in text
    assert "중복" in text


def test_corpus_check_catches_broken_supersede_chain(tmp_path):
    # 폐지본이 가리키는 후속 문서가 코퍼스에 없다 — 버전 인지 검색의 전제 붕괴
    _write_doc(tmp_path, "old.md", {"doc_id": "R-OLD", "title": "t", "version": "0.9",
                                    "effective_date": "2020-01-01", "status": "superseded",
                                    "superseded_by": "R-GONE"})
    problems = preflight.check_corpus(tmp_path)
    assert any("R-GONE" in p for p in problems)
    # superseded_by 자체가 없는 경우도 잡는다
    _write_doc(tmp_path, "old2.md", {"doc_id": "R-OLD2", "title": "t", "version": "0.9",
                                     "effective_date": "2020-01-01", "status": "superseded"})
    problems = preflight.check_corpus(tmp_path)
    assert any("superseded_by 가 없다" in p for p in problems)


def test_tasks_check_catches_bad_schema(tmp_path):
    f = tmp_path / "ra_tasks.json"
    f.write_text(
        '{"deadlines": [{"item": "x", "due_date": "다음주", "type": "t", "owner": "o", "status": "s"},'
        ' {"item": "y"}], "checklists": {"a": []}}',
        encoding="utf-8",
    )
    text = "\n".join(preflight.check_tasks(f))
    assert "YYYY-MM-DD" in text          # 날짜 형식
    assert "필수 필드 누락" in text        # deadlines[1]
    assert "비어 있다" in text             # 빈 체크리스트


def test_config_check_catches_contradictions(monkeypatch):
    """각 값은 유효해도 조합이 모순인 경우 — 조용한 품질 붕괴의 형태."""
    monkeypatch.setattr(config, "RERANK_TOP_N", 99)  # top_k(8)보다 큼
    problems = preflight.check_config()
    assert any("RERANK_TOP_N" in p for p in problems)
    monkeypatch.setattr(config, "RERANK_TOP_N", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 700)  # chunk_size(500)보다 큼
    problems = preflight.check_config()
    assert any("CHUNK_OVERLAP" in p for p in problems)
