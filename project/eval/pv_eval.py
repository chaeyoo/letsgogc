"""PV 워크플로 수치 평가 — 규칙 기반이라도 '측정 없이 신뢰 없음'.

트리아지→인과성→코딩→최소보고요건 전 단계를 라벨 데이터셋(pv_dataset.json,
20케이스)으로 평가한다. 검색(evaluate)·신뢰성(faithfulness)과 같은 철학:
PV 도구도 회귀가 수치로 잡혀야 사전·규칙을 안심하고 고칠 수 있다.

라벨은 '도메인 정답'이지 코드 출력의 복사가 아니다. 사전이 못 잡는 롱테일
구어(저혈당·울렁거림·난청)를 일부러 포함해 코딩 재현율이 정직하게 1.0 미만이
되게 했다 — 이 갭이 'MedDRA 본체 교체 + LLM 후보 제시' 확장 지점의 크기다.

측정 지표
  - SeriousnessAcc : 중대성(Serious) 판정 정확도 — 닫힌 목록 대조라 1.0이어야 정상
  - DeadlineAcc    : 보고기한 계산 정확도(일수 + 날짜 연산) — 컴플라이언스 그 자체
  - CausalityAcc   : WHO-UMC 제안 등급 일치율
  - Coding P/R/F1  : PT 코딩 micro 정밀도/재현율/F1 (재현율<1 = 사전 롱테일의 크기)
  - ReportableAcc  : 최소보고요건(ICH E2D 4요소) 판정 정확도
  - MissingCountAcc: 빠진 요소 개수까지 정확히 짚는가(라벨이 있는 문항만)

실행:  python -m eval.pv_eval
"""
from __future__ import annotations

import json

from src import config
from src.pv.report import build_report


def _load() -> list[dict]:
    path = config.BASE_DIR / "eval" / "pv_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def evaluate() -> dict:
    items = _load()
    n = len(items)
    serious_ok = deadline_ok = causality_ok = reportable_ok = 0
    missing_total = missing_ok = 0
    tp = fp = fn = 0
    failures: list[str] = []

    for item in items:
        exp = item["expect"]
        args = item.get("args", {})
        r = build_report(item["case"], **args)

        if r.triage.is_serious == exp["serious"]:
            serious_ok += 1
        else:
            failures.append(f"{item['id']} 중대성: got={r.triage.is_serious} want={exp['serious']}")

        dd_ok = r.triage.deadline_days == exp["deadline_days"]
        if "deadline_date" in exp:
            dd_ok = dd_ok and r.triage.deadline_date == exp["deadline_date"]
        if dd_ok:
            deadline_ok += 1
        else:
            failures.append(
                f"{item['id']} 기한: got={r.triage.deadline_days}({r.triage.deadline_date}) "
                f"want={exp['deadline_days']}({exp.get('deadline_date', '-')})"
            )

        # 라벨은 WHO-UMC 영문 키워드(Possible 등), 출력은 '가능함(Possible)' 형식
        if exp["causality"] in r.causality.suggested:
            causality_ok += 1
        else:
            failures.append(f"{item['id']} 인과성: got={r.causality.suggested} want={exp['causality']}")

        got_pts = {t.pt for t in r.coded_terms}
        want_pts = set(exp["pts"])
        tp += len(got_pts & want_pts)
        fp += len(got_pts - want_pts)
        fn += len(want_pts - got_pts)
        if want_pts - got_pts:
            failures.append(f"{item['id']} 코딩 미검출(사전 롱테일): {sorted(want_pts - got_pts)}")
        if got_pts - want_pts:
            failures.append(f"{item['id']} 코딩 오탐: {sorted(got_pts - want_pts)}")

        if r.reportable == exp["reportable"]:
            reportable_ok += 1
        else:
            failures.append(
                f"{item['id']} 보고요건: got={r.reportable} want={exp['reportable']} "
                f"(missing={r.missing})"
            )

        if "missing_n" in exp:
            missing_total += 1
            if len(r.missing) == exp["missing_n"]:
                missing_ok += 1
            else:
                failures.append(f"{item['id']} 누락개수: got={len(r.missing)} want={exp['missing_n']}")

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "n": n,
        "SeriousnessAcc": round(serious_ok / n, 3),
        "DeadlineAcc": round(deadline_ok / n, 3),
        "CausalityAcc": round(causality_ok / n, 3),
        "CodingPrecision": round(precision, 3),
        "CodingRecall": round(recall, 3),
        "CodingF1": round(f1, 3),
        "ReportableAcc": round(reportable_ok / n, 3),
        "MissingCountAcc": round(missing_ok / missing_total, 3) if missing_total else None,
        "failures": failures,
    }


def main() -> None:
    res = evaluate()
    print("=" * 62)
    print(f"PV 워크플로 평가 · 라벨 케이스 {res['n']}건 (트리아지→인과성→코딩→보고요건)")
    print("=" * 62)
    print(f"SeriousnessAcc  (중대성 판정)          : {res['SeriousnessAcc']:.3f}")
    print(f"DeadlineAcc     (보고기한 일수+날짜 연산): {res['DeadlineAcc']:.3f}")
    print(f"CausalityAcc    (WHO-UMC 제안 등급)     : {res['CausalityAcc']:.3f}")
    print(
        f"Coding P/R/F1   (PT 표준화 코딩)        : "
        f"{res['CodingPrecision']:.3f} / {res['CodingRecall']:.3f} / {res['CodingF1']:.3f}"
    )
    print(f"ReportableAcc   (최소보고요건 4요소)    : {res['ReportableAcc']:.3f}")
    if res["MissingCountAcc"] is not None:
        print(f"MissingCountAcc (빠진 요소 개수 정확도) : {res['MissingCountAcc']:.3f}")
    if res["failures"]:
        print("-" * 62)
        print("불일치 상세 (의도된 롱테일 감점 포함):")
        for f in res["failures"]:
            print(f"  - {f}")
    print("-" * 62)
    print("해석: 중대성·기한은 규정의 닫힌 목록 대조 + 날짜 연산이라 1.0이 정상이며,")
    print("      1.0이 깨지면 규칙 회귀다(CI가 잡는다). 코딩 재현율(<1.0)은 소사전")
    print("      롱테일의 크기 — 실무에선 MedDRA 본체 교체 + 'LLM 후보 제시→사람 확정'")
    print("      으로 메우는 확장 지점이다. 오탐(정밀도)은 1.0을 유지해야 한다 —")
    print("      잘못 붙은 코드는 집계를 오염시키지만, 못 붙인 코드는 보완 요청으로")
    print("      드러난다(시끄러운 실패가 올바른 실패 방향).")


if __name__ == "__main__":
    main()
