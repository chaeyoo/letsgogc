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


def test_corpus_check_catches_chain_date_inversion(tmp_path):
    """폐지 체인의 시행일 역전 — as_of 시점 조회의 구간 판정([구판 시행일,
    후속본 시행일))이 빈 구간이 되어 시점 조회를 조용히 망가뜨리는 결함."""
    _write_doc(tmp_path, "old.md", {"doc_id": "R-OLD", "title": "t", "version": "0.9",
                                    "effective_date": "2025-01-01", "status": "superseded",
                                    "superseded_by": "R-NEW"})
    _write_doc(tmp_path, "new.md", {"doc_id": "R-NEW", "title": "t", "version": "1.0",
                                    "effective_date": "2024-01-01"})  # 구판보다 이른 시행일
    problems = preflight.check_corpus(tmp_path)
    assert any("늦지 않다" in p for p in problems)


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


def test_tasks_check_catches_stale_deadlines(tmp_path):
    """시한부 샘플 데이터의 부패 — 마감일이 전부 과거가 되면 '임박한 마감' 데모가
    전항목 연체 목록으로 조용히 죽는다(스키마는 유효하므로 형식 검사만으로는
    영원히 통과). 전건 과거일 때만 결함으로 센다 — 과거 1건은 '지남/긴급' 연출용
    의도된 데이터라 오탐이 되면 안 된다(리포 데이터 통과는 별도 테스트가 고정)."""
    import datetime as dt
    import json as _json

    past = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    future = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    row = {"item": "x", "type": "t", "owner": "o", "status": "s"}
    f = tmp_path / "ra_tasks.json"

    f.write_text(_json.dumps({
        "deadlines": [{**row, "due_date": past}, {**row, "due_date": past}],
        "checklists": {"a": ["1"]},
    }), encoding="utf-8")
    assert any("모두 과거" in p for p in preflight.check_tasks(f))

    f.write_text(_json.dumps({
        "deadlines": [{**row, "due_date": past}, {**row, "due_date": future}],
        "checklists": {"a": ["1"]},
    }), encoding="utf-8")
    assert not any("모두 과거" in p for p in preflight.check_tasks(f))


def test_gate_self_tests_cover_every_warning_axis():
    """자가 테스트의 축 대칭성 — 런타임 게이트의 모든 경고 축(GateStats._AXES)에
    대해 '심은 오류' 케이스가 존재해야 한다. 처음에는 수치 존재 축만 자가
    테스트했는데, 그러면 방향·역할·버전 축은 고장난 채 배포될 수 있다 —
    새 축을 추가하면 이 테스트가 자가 테스트 추가를 강제한다."""
    from src.observability import GateStats

    planted_axes = {t["axis"] for t in preflight._GATE_SELF_TESTS if not t["expect_ok"]}
    assert planted_axes == set(GateStats._AXES), (
        f"자가 테스트 미커버 축: {set(GateStats._AXES) - planted_axes}"
    )
    # 정상(clean) 케이스도 최소 1건 이상 — 오탐 방향의 자가 테스트
    assert any(t["expect_ok"] for t in preflight._GATE_SELF_TESTS)


def test_smoke_catches_broken_gate(monkeypatch):
    """게이트가 모든 답변을 통과시키는 고장(항상 ok) 상태로는 배포되지 않는다."""
    from src.verify import verifier

    monkeypatch.setattr(
        verifier, "verify_answer", lambda *a, **k: verifier.VerificationResult()
    )
    problems = preflight.smoke_checks()
    assert any("검증 게이트 자가 테스트 실패" in p for p in problems)


def test_smoke_catches_broken_redactor(monkeypatch):
    """안전장치 자가 테스트의 대칭성: 검증 게이트뿐 아니라 PII 마스킹도 고장난
    채로는 배포되지 않는다 — 심은 개인정보가 살아남으면 smoke 가 잡아야 한다."""
    from src.pv import redactor

    monkeypatch.setattr(
        redactor, "redact",
        lambda text: redactor.RedactionReport(text=text, counts={}),  # 아무것도 안 지우는 고장
    )
    problems = preflight.smoke_checks()
    assert any("PII 마스킹 자가 테스트 실패" in p for p in problems)


def test_smoke_catches_broken_history_block_masking(monkeypatch):
    """이력 마스킹의 표기 변형 자가 테스트 — 블록 리스트 표기(content=[{type:
    text}])의 마스킹이 죽은 채로는 배포되지 않는다. 문자열 표기 자가 테스트만
    있으면 블록 경로의 고장은 신호 없이 통과한다(표기 변형 커버리지의 대칭)."""
    from src.agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "_redact_history", lambda history: history)  # 원문 통과 고장
    problems = preflight.smoke_checks()
    assert any("블록 표기" in p for p in problems)


def test_config_check_catches_contradictions(monkeypatch):
    """각 값은 유효해도 조합이 모순인 경우 — 조용한 품질 붕괴의 형태."""
    monkeypatch.setattr(config, "RERANK_TOP_N", 99)  # top_k(8)보다 큼
    problems = preflight.check_config()
    assert any("RERANK_TOP_N" in p for p in problems)
    monkeypatch.setattr(config, "RERANK_TOP_N", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 700)  # chunk_size(500)보다 큼
    problems = preflight.check_config()
    assert any("CHUNK_OVERLAP" in p for p in problems)


def test_corpus_check_accepts_valid_two_tier_chain(tmp_path):
    """정당한 2단 폐지 체인(구판→중간판(폐지)→현행)은 결함이 아니다. (v8)

    superseded_by 규약은 '직전 후속본'이다 — 리트리버의 as_of 구간 판정
    [시행일, 후속본 시행일)이 이를 전제하는데, 종전 preflight 는 '후속본이
    폐지본이면 결함'이라 정반대 규약을 요구했다(다단 이력을 넣는 순간 어느
    쪽으로 써도 한쪽이 깨지는 잠복 상충)."""
    _write_doc(tmp_path, "v1.md", {"doc_id": "R-V1", "title": "t", "version": "1.0",
                                   "effective_date": "2020-01-01", "status": "superseded",
                                   "superseded_by": "R-V2"})
    _write_doc(tmp_path, "v2.md", {"doc_id": "R-V2", "title": "t", "version": "2.0",
                                   "effective_date": "2022-01-01", "status": "superseded",
                                   "superseded_by": "R-V3"})
    _write_doc(tmp_path, "v3.md", {"doc_id": "R-V3", "title": "t", "version": "3.0",
                                   "effective_date": "2024-01-01"})
    assert preflight.check_corpus(tmp_path) == []


def test_corpus_check_catches_chain_cycle_and_dangling_tail(tmp_path):
    """순환 체인과, 중간에서 끊기는 체인(폐지 종점)은 결함으로 잡는다. (v8)"""
    _write_doc(tmp_path, "a.md", {"doc_id": "R-A", "title": "t", "version": "1",
                                  "effective_date": "2020-01-01", "status": "superseded",
                                  "superseded_by": "R-B"})
    _write_doc(tmp_path, "b.md", {"doc_id": "R-B", "title": "t", "version": "2",
                                  "effective_date": "2021-01-01", "status": "superseded",
                                  "superseded_by": "R-A"})  # 순환
    problems = preflight.check_corpus(tmp_path)
    assert any("순환" in p for p in problems)


def test_corpus_check_catches_mid_chain_date_inversion(tmp_path):
    """다단 체인 중간 링크의 시행일 역전도 잡는다. (v8)"""
    _write_doc(tmp_path, "v1.md", {"doc_id": "R-V1", "title": "t", "version": "1",
                                   "effective_date": "2020-01-01", "status": "superseded",
                                   "superseded_by": "R-V2"})
    _write_doc(tmp_path, "v2.md", {"doc_id": "R-V2", "title": "t", "version": "2",
                                   "effective_date": "2023-01-01", "status": "superseded",
                                   "superseded_by": "R-V3"})
    _write_doc(tmp_path, "v3.md", {"doc_id": "R-V3", "title": "t", "version": "3",
                                   "effective_date": "2022-01-01"})  # 중간 링크 역전
    problems = preflight.check_corpus(tmp_path)
    assert any("늦지 않다" in p for p in problems)
