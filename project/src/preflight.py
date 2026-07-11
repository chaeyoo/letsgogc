"""배포 전 점검(preflight) — 서버를 띄우기 전에 통과해야 하는 Day-0 게이트.

FDE가 시스템을 고객사에 배치할 때 가장 먼저 깨지는 것은 코드가 아니라
**데이터와 설정**이다: 문서 메타데이터 누락, 폐지 체인 단절, 업무 데이터의
날짜 오타, 상충하는 하이퍼파라미터. 이런 결함은 부팅은 통과하고 **운영 중에
오답으로** 나타난다 — 규제 도메인에서 가장 나쁜 실패 방향이다(예: status
필드가 빠진 폐지 구판은 버전 필터를 그대로 통과해 현행 답변에 섞인다).

그래서 기동 전에 결정론적 점검을 강제한다. 네 그룹:

  1. 설정 불변식 — 하이퍼파라미터끼리의 관계(overlap < chunk_size,
     rerank_top_n ≤ retrieve_top_k, 0 ≤ α ≤ 1 …). 각각은 유효해도 조합이
     모순이면 조용히 품질이 깨진다.
  2. 코퍼스 무결성 — frontmatter 필수 필드(doc_id·title·version·
     effective_date), doc_id 유일성, status 어휘, **폐지 체인**(superseded
     문서는 superseded_by 가 실존하는 현행 문서를 가리켜야 한다 — 버전 인지
     검색의 전제가 데이터에서 성립하는지).
  3. 업무 데이터 스키마 — ra_tasks.json 의 마감일 필수 필드·날짜 형식,
     체크리스트 비어있지 않음.
  4. 스모크(canary) — 실제 파이프라인을 태워 본다: 대표 질의가 정답 문서를
     1위로 회수하는가, 대표 케이스가 중대+15일로 트리아지되는가, 그리고
     **안전장치들의 자가 테스트** — 검증 게이트(근거 밖 수치를 심은 답변이
     실제로 걸리는가 / 정상 답변이 통과하는가)와 PII 마스킹(심은 개인정보가
     실제로 지워지는가). 안전장치가 고장난 채 배포되는 것은 안전장치가 없는
     것보다 나쁘다 — 안전장치 자신도 기동 전에 검사받는다.

실패 방향: 문제가 하나라도 있으면 **exit 1** — run.sh 와 CI 가 이 게이트를
통과해야 서버를 띄운다(운영 중 경고 부착과 달리, 배포 시점은 차단이 옳다:
아직 사용자가 없으므로 시끄럽게 멈추는 비용이 0이다).

실행:  python -m src.preflight
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

from . import config

# 코퍼스 frontmatter 계약 — 출처 표시·버전 인지 검색이 의존하는 최소 필드
_REQUIRED_META = ("doc_id", "title", "version", "effective_date")
_VALID_STATUS = ("", "active", "superseded")  # 없음(현행 암시)·현행·폐지

# 스모크 canary — 코퍼스의 핵심 문서·핵심 판정이 살아 있는지 확인하는 대표 케이스.
# canary 는 '어려운' 케이스가 아니라 '무조건 되어야 하는' 케이스다: 이것이
# 깨졌다면 데이터 교체·설정 실수로 시스템의 전제 자체가 무너진 것이다.
_CANARY_QUERY = "중대한 이상사례는 며칠 안에 보고해야 하나?"
_CANARY_DOC = "REG-005"
# 입원(중대) but 생명위협 아님 → '인지일+15일' 경로: 기한 '일수'와 '날짜 연산'을 함께 검사
_CANARY_CASE = "환자가 A정 복용 후 심한 두드러기로 입원했습니다."
# 시점(as_of) 조회 canary — 현행(REG-005, 2025-04-01 시행) 이전 시점에는
# 폐지 구판(REG-013)이 '당시 현행'으로 반환되어야 한다. 이것이 깨지는 대표
# 형태가 '폐지본 기본 제외를 as_of에도 적용 → 개정된 규정은 과거 시점에서
# 아무 버전도 안 나옴'이라, 시점 조회의 의미론 자체를 기동 전에 검사한다.
_CANARY_AS_OF = "2025-01-01"
_CANARY_AS_OF_DOC = "REG-013"

# 검증 게이트 자가 테스트 케이스 — 런타임 게이트의 **모든 경고 축**을 기동 전에
# 1건씩 태워 본다. 처음에는 수치 존재 축 1개만 심어 봤는데, 그러면 방향·역할·
# 버전·전제 라벨 축은 고장난 채 배포될 수 있다 — '자가 테스트를 한 안전장치에만
# 적용하는 비대칭이 사각지대'라는 원칙(게이트 vs PII 마스킹)은 게이트 **내부의
# 축들 사이에도** 똑같이 적용되어야 한다. planted(심은 오류) 케이스는 기대한
# 바로 그 축에 걸려야 하고(다른 축에 우연히 걸린 통과는 통과가 아니다),
# clean(정상) 케이스는 오탐 없이 통과해야 한다. 축 목록과 테이블의 커버리지가
# 어긋나면 테스트(test_preflight)가 실패한다 — 새 축을 추가하면 자가 테스트도
# 함께 추가하도록 구조로 강제한 것.
_GATE_TOOL_LABELED = '{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25"}'
_GATE_SELF_TESTS: tuple[dict, ...] = (
    {"name": "수치 존재·심은 오류", "answer": "보고 기한은 30일 이내입니다",
     "trusted": ["보고 기한은 15일 이내"], "expect_ok": False, "axis": "unsupported"},
    {"name": "수치 존재·정상", "answer": "보고 기한은 15일 이내입니다",
     "trusted": ["보고 기한은 15일 이내"], "expect_ok": True},
    {"name": "날짜 존재·심은 오류", "answer": "마감일은 2026-07-30 입니다",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "unsupported"},
    {"name": "방향 한정어·심은 오류", "answer": "15일 이후에 보고하면 됩니다",
     "trusted": ["인지일로부터 15일 이내 신속보고"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "날짜 방향 한정어·심은 오류", "answer": "2026-07-25 이후에 제출하면 됩니다",
     "trusted": ["보완자료는 2026-07-25까지 제출한다"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "날짜 역할 스왑·심은 오류", "answer": "보고 기한: 2026-07-10 (인지일 2026-07-25 기준)",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "role_conflicts"},
    {"name": "날짜 역할·정상", "answer": "보고 기한: 2026-07-25 (인지일 2026-07-10 기준)",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": True},
    {"name": "폐지본 인용·심은 오류", "answer": "이상사례는 30일 이내 보고한다",
     "trusted": ["30일 이내"], "citations": [{"doc_id": "REG-013", "status": "superseded"}],
     "expect_ok": False, "axis": "superseded_cited"},
    {"name": "폐지본 인용·이력 모드 정상", "answer": "구판 기준은 30일이었다",
     "trusted": ["30일"], "citations": [{"doc_id": "REG-013", "status": "superseded"}],
     "allow_superseded": True, "expect_ok": True},
    {"name": "전제 에코 라벨", "answer": "30일이 아니라 15일 이내입니다",
     "trusted": ["인지일로부터 15일 이내"], "question": "보고 기한이 30일 맞나요?",
     "expect_ok": False, "axis": "question_origin"},
)


def check_config() -> list[str]:
    """하이퍼파라미터 간 불변식 — 값 각각이 아니라 '조합'의 모순을 잡는다."""
    problems: list[str] = []
    if not (0 < config.CHUNK_OVERLAP < config.CHUNK_SIZE):
        problems.append(
            f"CHUNK_OVERLAP({config.CHUNK_OVERLAP})은 0보다 크고 CHUNK_SIZE({config.CHUNK_SIZE})보다 작아야 한다"
        )
    if not (0.0 <= config.HYBRID_ALPHA <= 1.0):
        problems.append(f"HYBRID_ALPHA({config.HYBRID_ALPHA})는 [0,1] 범위여야 한다")
    if not (0.0 <= config.RERANK_WEIGHT <= 1.0):
        problems.append(f"RERANK_WEIGHT({config.RERANK_WEIGHT})는 [0,1] 범위여야 한다")
    if config.RERANK_TOP_N > config.RETRIEVE_TOP_K:
        problems.append(
            f"RERANK_TOP_N({config.RERANK_TOP_N})이 RETRIEVE_TOP_K({config.RETRIEVE_TOP_K})보다 크다"
            " — 리랭킹이 1차 회수보다 많이 반환할 수 없다"
        )
    if config.RERANK_TOP_N < 1 or config.RETRIEVE_TOP_K < 1:
        problems.append("RERANK_TOP_N/RETRIEVE_TOP_K 는 1 이상이어야 한다")
    if config.EMBEDDER_KIND not in ("tfidf", "hashing", "voyage"):
        problems.append(f"EMBEDDER_KIND('{config.EMBEDDER_KIND}')는 tfidf|hashing|voyage 중 하나여야 한다")
    if config.EMBEDDER_KIND == "voyage" and not os.environ.get("VOYAGE_API_KEY", ""):
        problems.append("EMBEDDER_KIND=voyage 인데 VOYAGE_API_KEY 가 없다")
    return problems


def check_corpus(reg_dir: Path | None = None) -> list[str]:
    """규제문서 frontmatter 계약 + 폐지 체인 무결성."""
    from .rag.loader import load_documents

    reg_dir = reg_dir or config.REG_DIR
    problems: list[str] = []
    docs = load_documents(reg_dir)
    if not docs:
        return [f"규제문서가 없다: {reg_dir}"]

    seen_ids: dict[str, str] = {}
    all_ids = {d.metadata.get("doc_id") for d in docs}
    for d in docs:
        meta = d.metadata
        missing = [k for k in _REQUIRED_META if not meta.get(k)]
        if missing:
            problems.append(f"{d.source}: frontmatter 필수 필드 누락 {missing}")
        doc_id = meta.get("doc_id")
        if doc_id:
            if doc_id in seen_ids:
                problems.append(f"{d.source}: doc_id '{doc_id}' 중복 ({seen_ids[doc_id]}와 충돌)")
            seen_ids[doc_id] = d.source
        status = str(meta.get("status", ""))
        if status not in _VALID_STATUS:
            problems.append(f"{d.source}: status '{status}' 는 허용 어휘 {_VALID_STATUS} 밖")
        eff = str(meta.get("effective_date", ""))
        if eff:
            try:
                _dt.date.fromisoformat(eff)
            except ValueError:
                problems.append(f"{d.source}: effective_date '{eff}' 가 YYYY-MM-DD 형식이 아님")
        if not d.text.strip():
            problems.append(f"{d.source}: 본문이 비어 있음")
        # 폐지 체인: 폐지본은 후속 문서를 가리켜야 하고, 그 문서는 실존하는 현행이어야 한다.
        # 단절된 체인은 '이력 조회'와 '현행 대체본 안내'를 조용히 망가뜨린다.
        if status == "superseded":
            succ = meta.get("superseded_by")
            if not succ:
                problems.append(f"{d.source}: superseded 인데 superseded_by 가 없다")
            elif succ not in all_ids:
                problems.append(f"{d.source}: superseded_by '{succ}' 문서가 코퍼스에 없다")
            else:
                succ_doc = next(x for x in docs if x.metadata.get("doc_id") == succ)
                if str(succ_doc.metadata.get("status", "")) == "superseded":
                    problems.append(f"{d.source}: superseded_by '{succ}' 도 폐지본 — 현행 종점이 없는 체인")
                # 체인의 시간 단조성: 후속본 시행일이 구판 시행일보다 늦어야 한다.
                # as_of 시점 조회는 [구판 시행일, 후속본 시행일) 구간으로 '당시
                # 현행'을 판정하므로, 시행일이 역전된 체인은 빈 구간(어느 시점에도
                # 유효하지 않은 버전)을 만들어 시점 조회를 조용히 망가뜨린다.
                succ_eff, pred_eff = str(succ_doc.metadata.get("effective_date", "")), eff
                try:
                    if (
                        succ_eff and pred_eff
                        and _dt.date.fromisoformat(succ_eff) <= _dt.date.fromisoformat(pred_eff)
                    ):
                        problems.append(
                            f"{d.source}: 후속본 '{succ}' 시행일({succ_eff})이 구판 시행일({pred_eff})보다"
                            " 늦지 않다 — 시점(as_of) 조회의 구간 판정이 깨진다"
                        )
                except ValueError:
                    pass  # 날짜 형식 오류는 위의 형식 검사가 별도로 보고한다
    return problems


def check_tasks(tasks_file: Path | None = None) -> list[str]:
    """RA 업무 데이터(ra_tasks.json) 스키마 — 도구가 읽다가 죽거나 오답을 내기 전에."""
    tasks_file = tasks_file or config.RA_TASKS_FILE
    problems: list[str] = []
    try:
        data = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{tasks_file.name}: 읽기/파싱 실패 — {e}"]

    for i, d in enumerate(data.get("deadlines", [])):
        missing = [k for k in ("item", "due_date", "type", "owner", "status") if not d.get(k)]
        if missing:
            problems.append(f"deadlines[{i}]: 필수 필드 누락 {missing}")
        due = str(d.get("due_date", ""))
        if due:
            try:
                _dt.date.fromisoformat(due)
            except ValueError:
                problems.append(f"deadlines[{i}]: due_date '{due}' 가 YYYY-MM-DD 형식이 아님")
    checklists = data.get("checklists", {})
    if not checklists:
        problems.append("checklists 가 비어 있다")
    for cat, items in checklists.items():
        if not items:
            problems.append(f"checklists['{cat}'] 가 비어 있다")
    return problems


def smoke_checks() -> list[str]:
    """실제 파이프라인 canary + 검증 게이트 자가 테스트."""
    from .mcp_server.server import assess_adverse_event, search_regulations
    from .verify.verifier import verify_answer

    problems: list[str] = []

    # (a) RAG canary: 핵심 규정 질의가 정답 문서를 1위로 회수하는가
    res = search_regulations(_CANARY_QUERY, top_n=1).get("results", [])
    if not res:
        problems.append(f"RAG canary 실패: '{_CANARY_QUERY}' 검색 결과 0건")
    elif res[0].get("doc_id") != _CANARY_DOC:
        problems.append(
            f"RAG canary 실패: '{_CANARY_QUERY}' 1위가 {res[0].get('doc_id')} (기대: {_CANARY_DOC})"
            " — 코퍼스/인덱스/하이퍼파라미터 변경이 핵심 검색을 깨뜨렸다"
        )

    # (a') 시점 조회 canary: 과거 시점(as_of)에는 당시 시행 중이던 구판이 나오는가
    hist = search_regulations(_CANARY_QUERY, top_n=1, as_of=_CANARY_AS_OF).get("results", [])
    if not hist:
        problems.append(
            f"시점 조회 canary 실패: as_of={_CANARY_AS_OF} 검색 결과 0건"
            " — 당시 시행 중이던 버전이 필터에서 전부 걸러졌다"
        )
    elif hist[0].get("doc_id") != _CANARY_AS_OF_DOC:
        problems.append(
            f"시점 조회 canary 실패: as_of={_CANARY_AS_OF} 1위가 {hist[0].get('doc_id')}"
            f" (기대: {_CANARY_AS_OF_DOC} — 그 시점의 현행 버전)"
        )

    # (b) PV canary: 대표 케이스(입원=중대)가 인지일+15일로 트리아지되는가
    tri = assess_adverse_event(_CANARY_CASE, awareness_date="2026-01-01")
    if not tri.get("is_serious"):
        problems.append("PV canary 실패: 입원 케이스가 비중대로 판정됨")
    elif tri.get("deadline_days") != 15:
        problems.append(f"PV canary 실패: 보고기한 {tri.get('deadline_days')}일 (기대: 15일)")
    elif tri.get("deadline_date") != "2026-01-16":
        problems.append(
            f"PV canary 실패: 마감일 {tri.get('deadline_date')} (기대: 2026-01-16 = 인지일+15일)"
        )

    # (c) 검증 게이트 자가 테스트: 게이트가 고장난 채 배포되면 안전장치가
    #     없는 것보다 나쁘다(있다고 믿게 만들므로). 런타임 게이트의 모든 경고
    #     축(존재·방향·역할·버전·전제 라벨)에 대해 '심은 오류는 그 축에 걸리고,
    #     정상 케이스는 통과한다'를 테이블로 태워 본다(_GATE_SELF_TESTS).
    for t in _GATE_SELF_TESTS:
        v = verify_answer(
            t["answer"],
            t["trusted"],
            t.get("citations"),
            t.get("allow_superseded", False),
            question=t.get("question", ""),
        )
        s = v.summary()
        if t["expect_ok"]:
            if not v.ok:
                problems.append(f"검증 게이트 자가 테스트 실패[{t['name']}]: 정상 케이스에 오탐 — {s}")
        elif v.ok:
            problems.append(f"검증 게이트 자가 테스트 실패[{t['name']}]: 심은 오류가 통과됨")
        elif not s.get(t["axis"]):
            problems.append(
                f"검증 게이트 자가 테스트 실패[{t['name']}]: 기대 축({t['axis']})이 아닌"
                f" 다른 축에 걸렸다 — 해당 축의 탐지가 죽었을 수 있음: {s}"
            )

    # (d) PII 마스킹 자가 테스트: 안전장치 자가 검사를 검증 게이트에만 하고
    #     입구 경계(마스킹)에는 안 하는 비대칭이 사각지대였다 — 마스킹이 고장난
    #     채 배포되면 첫 실사용 케이스의 개인정보가 외부 API·로그로 나간다.
    from .pv.redactor import redact

    probe = redact("환자 김철수님(연락처 010-1234-5678, 주민번호 900101-1234567) 케이스")
    for planted in ("김철수", "010-1234-5678", "900101-1234567"):
        if planted in probe.text:
            problems.append(f"PII 마스킹 자가 테스트 실패: 심은 개인정보 '{planted}' 가 마스킹되지 않았다")
    return problems


def run_preflight() -> dict[str, list[str]]:
    """전 그룹을 실행해 {그룹명: 문제 목록} 을 반환한다(빈 목록 = 통과)."""
    return {
        "설정 불변식": check_config(),
        "코퍼스 무결성": check_corpus(),
        "업무 데이터 스키마": check_tasks(),
        "스모크(canary)": smoke_checks(),
    }


def main() -> int:
    print("=" * 60)
    print("배포 전 점검 (preflight) — 데이터·설정·스모크·게이트 자가 테스트")
    print("=" * 60)
    report = run_preflight()
    n_problems = 0
    for group, problems in report.items():
        mark = "✓" if not problems else "✗"
        print(f"[{mark}] {group}")
        for p in problems:
            print(f"    - {p}")
        n_problems += len(problems)
    print("-" * 60)
    if n_problems:
        print(f"결과: 실패 — 문제 {n_problems}건. 서버를 띄우기 전에 해결해야 한다.")
        return 1
    print("결과: 통과 — 배포 가능.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
