"""검증기의 검증(meta-evaluation) — "그 검증기는 누가 검증하나"에 대한 답.

답변 사후 검증(src/verify)은 운영에서 모든 응답이 통과하는 안전장치다.
안전장치는 두 방향 모두에서 측정돼야 한다:

  (1) 잡아야 할 것을 잡는가 — 근거에 없는 수치·뒤집힌 방향·틀린 날짜가
      섞인 답변을 탐지하는가.
  (2) 잡지 말아야 할 것을 흘려보내는가 — 정상(근거 발췌) 답변과 정당한
      표기 변형(보름=15일)에 오탐을 내지 않는가. 오탐이 잦으면 담당자가
      경고를 무시하기 시작하고(alert fatigue), 그 순간 검증 계층 전체가 죽는다.

정상 답변의 '오류 버전'을 사람이 라벨링하는 대신, 정상 답변에 **결정론적
변조(seeded corruption)**를 가해 오류 케이스를 합성한다(메타모픽 테스트).
변조는 실제 환각의 형태를 모사한다:

  - 교차문서 치환(cross-doc swap): 답변 속 수치를 '코퍼스의 다른 문서에는
    실제로 존재하는' 같은 단위의 값으로 바꾼다. LLM 환각의 전형인
    '그럴듯한 혼동'(의료기기 80 근무일 ↔ 의약품 120 근무일)이며, 검증이
    전역 코퍼스가 아니라 **이 질의의 근거**를 기준으로 하는지를 함께 확인한다.
  - 오프셋 변조(offset): 수치에 상수를 더한다(15일→22일). 날짜 연산
    오류·자릿수 실수 형태의 환각 모사.
  - 방향 뒤집기(direction flip): 수치는 그대로 두고 한정어만 뒤집는다
    ("15일 이내"→"15일 이후"). 수치 대조만으로는 통과하는, 가장 교묘한
    형태의 왜곡 — 검증기 v2의 방향 한정어 축이 잡아야 한다.
  - 고유어 치환(native numeral swap): 수치를 값이 **다른** 고유어 수사로
    바꾼다("15일"→"열흘"). v1에서는 아예 추출되지 않아 **조용히 통과**하던
    형태다(측정 자체가 불가능했던 사각지대의 가시화).
  - 날짜 시프트(date shift): 도구가 계산한 보고 마감일(YYYY-MM-DD)을
    며칠 밀어 바꾼다. 날짜 클레임 축은 운영 코드에 있었지만 v1 평가는
    수치만 변조해 이 축을 측정하지 않았다 — 측정 없는 축은 없는 축이다.

지표 해석의 규율 — '핀'과 '실측'을 구분해 읽는다:
  탐지율(Swap/Offset/Direction/Native/DateShift)은 '근거에 없는 값·방향'으로
  합성했으므로 1.0이 정상이며, 새로운 발견이 아니라 **검증기 회귀를 고정하는
  핀**이다 — 클레임 추출 정규식이나 단위 목록을 건드려 탐지가 새기 시작하면
  이 수치가 먼저 떨어진다. CleanPassRate 도 절반은 구성적이다: 발췌 답변은
  신뢰 소스의 부분 문자열이라 **통과하도록 만들어져** 있고, 이 지표가 잡는
  것은 추출·정규화의 비대칭(같은 텍스트를 답변에서 볼 때와 근거에서 볼 때
  다르게 읽는 회귀)이다. 구성이 개입하지 않는 진짜 실측은 E2EPassRate —
  포매터가 근거 밖 수치를 만들어내면 여기가 먼저 깨진다.

실행:  python -m eval.verify_eval
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re

from src import config
from src.agent.agent import _strip_query_echo
from src.mcp_server.server import assess_adverse_event, search_regulations
from src.rag.loader import load_documents
from src.verify.verifier import (
    _QUAL_RE,
    _qual_class,
    extract_claims,
    verify_answer,
)

_OFFSETS = (7, 11, 13, 17, 23)  # 오프셋 변조 시 근거와의 우연한 충돌을 피하는 결정론적 시퀀스

# 고유어 치환 후보: 값 → 고유어 수사('일' 단위만 — 사전과 동일 어휘)
_NATIVE_BY_VALUE = {"2": "이틀", "3": "사흘", "4": "나흘", "5": "닷새", "10": "열흘", "15": "보름"}
_AWARENESS_DATE = "2026-07-10"  # 날짜 시프트 축의 고정 인지일(결정론)


def _load_items(name: str) -> list[dict]:
    return json.loads((config.BASE_DIR / "eval" / name).read_text(encoding="utf-8"))["items"]


def _corpus_claim_pool() -> set[tuple[str, str]]:
    """전 규제문서에서 (수치, 단위) 풀을 수집 — 교차문서 치환 후보."""
    pool: set[tuple[str, str]] = set()
    for doc in load_documents(config.REG_DIR):
        nums, _ = extract_claims(doc.text)
        pool |= nums
    return pool


def _trusted_of(search_data: dict) -> list[str]:
    """운영(agent._finalize)과 동일한 신뢰 소스 뷰 — 질의 에코 제외.

    평가가 운영과 다른 신뢰 소스를 쓰면 여기서의 1.0이 운영을 보증하지 않는다.
    """
    return [json.dumps(_strip_query_echo(search_data), ensure_ascii=False)]


def _faithful_answer(search_data: dict) -> str:
    """검색 최상위 근거의 발췌로 '정상(grounded) 답변'을 합성한다."""
    results = search_data.get("results", [])
    if not results:
        return ""
    body = results[0]["text"].split("]\n", 1)[-1].strip()
    return body[:400]


def _replace_claim(text: str, num: str, unit: str, replacement: str) -> str:
    """답변 속 (수치, 단위) 표기를 대체 표기로 바꾼다(공백·선행 0 표기 흡수).

    '120 근무일'처럼 숫자와 단위 사이에 공백이 있는 표기도 잡아야
    변조가 실제로 일어난다 — 치환 실패는 '통과처럼 보이는 미변조'가 된다.
    """
    unit_pat = "주일?" if unit == "주" else re.escape(unit)
    pat = re.compile(rf"(?<!\d)0*{re.escape(num)}\s*{unit_pat}")
    return pat.sub(replacement, text, count=1)


def _swap_value(claim: tuple[str, str], trusted_nums: set[tuple[str, str]],
                pool: set[tuple[str, str]]) -> str | None:
    """같은 단위의 코퍼스 값 중, 이 질의의 근거에는 없는 값을 결정론적으로 고른다."""
    num, unit = claim
    candidates = sorted(
        v for v, u in pool
        if u == unit and v != num and (v, unit) not in trusted_nums
    )
    return candidates[0] if candidates else None


def _offset_value(claim: tuple[str, str], trusted_nums: set[tuple[str, str]]) -> str | None:
    num, unit = claim
    try:
        base = int(num)
    except ValueError:
        return None
    for off in _OFFSETS:
        cand = str(base + off)
        if (cand, unit) not in trusted_nums:
            return cand
    return None


def evaluate() -> dict:
    qa = _load_items("qa_dataset.json")
    pool = _corpus_claim_pool()

    clean_n = clean_pass = 0
    swap_n = swap_detected = 0
    off_n = off_detected = 0
    dir_n = dir_detected = 0
    nat_n = nat_detected = 0
    para_n = para_pass = 0
    no_claim_items = 0

    for item in qa:
        data = search_regulations(item["question"], top_n=3)
        trusted = _trusted_of(data)
        answer = _faithful_answer(data)
        if not answer:
            continue

        # (2) 오탐 측정: 근거 발췌 답변은 통과해야 한다
        clean_n += 1
        if verify_answer(answer, trusted).ok:
            clean_pass += 1

        claims, _ = extract_claims(answer)
        if not claims:
            no_claim_items += 1
            continue
        trusted_nums, _ = extract_claims("\n".join(trusted))

        # (1) 탐지 측정: 클레임마다 변조 답변을 만들어 검증기에 통과시켜 본다
        for claim in sorted(claims):
            num, unit = claim
            swap = _swap_value(claim, trusted_nums, pool)
            if swap is not None:
                corrupted = _replace_claim(answer, num, unit, f"{swap}{unit}")
                if corrupted != answer:  # 치환 실패(고유어 유래 클레임 등)는 표본에서 제외
                    swap_n += 1
                    if not verify_answer(corrupted, trusted).ok:
                        swap_detected += 1
            off = _offset_value(claim, trusted_nums)
            if off is not None:
                corrupted = _replace_claim(answer, num, unit, f"{off}{unit}")
                if corrupted != answer:
                    off_n += 1
                    if not verify_answer(corrupted, trusted).ok:
                        off_detected += 1
            # 고유어 치환: 값이 '다른' 고유어로 — v1의 조용한 사각지대였던 형태
            if unit == "일":
                wrong = next(
                    (w for v, w in sorted(_NATIVE_BY_VALUE.items())
                     if v != num and (v, "일") not in trusted_nums),
                    None,
                )
                if wrong is not None:
                    corrupted = _replace_claim(answer, num, unit, wrong)
                    if corrupted != answer:
                        nat_n += 1
                        if not verify_answer(corrupted, trusted).ok:
                            nat_detected += 1
                # 반대 방향(오탐 감시): 값이 '같은' 고유어 표기는 통과해야 한다
                correct = _NATIVE_BY_VALUE.get(num)
                if correct is not None:
                    paraphrased = _replace_claim(answer, num, unit, correct)
                    if paraphrased != answer:
                        para_n += 1
                        if verify_answer(paraphrased, trusted).ok:
                            para_pass += 1

        # 방향 뒤집기: 근거가 한 방향으로만 말하는 한정어를 반대로 뒤집는다
        trusted_text = "\n".join(trusted)
        from src.verify.verifier import _qualifier_map  # 지역 import: 평가 전용 내부 접근

        src_quals = _qualifier_map(trusted_text)
        flipped_done: set[str] = set()
        for m in _QUAL_RE.finditer(answer):
            unit = "주" if m.group(2) == "주일" else m.group(2)
            key = (m.group(1).lstrip("0") or "0", unit)
            cls = _qual_class(m.group(3))
            opposite_word = "이후" if cls == "상한" else "이내"
            # 근거가 반대 방향으로도 같은 수치를 말하면 충돌이 성립하지 않으므로 제외
            if _qual_class(opposite_word) in src_quals.get(key, set()):
                continue
            span = m.group(0)
            if span in flipped_done:
                continue
            flipped_done.add(span)
            corrupted = answer.replace(span, span[: -len(m.group(3))] + opposite_word, 1)
            dir_n += 1
            if not verify_answer(corrupted, trusted).ok:
                dir_detected += 1

    # (3) 폐지본 인용 감지 — 버전 검증 축
    sup_cite = [{"doc_id": "REG-013", "status": "superseded"}]
    superseded_flagged = not verify_answer("이상사례는 30일 이내 보고한다", ["30일 이내"], sup_cite).ok
    history_allowed = verify_answer(
        "구판 기준은 30일이었다", ["30일"], sup_cite, allow_superseded=True
    ).ok

    # (4) 날짜 시프트 — 도구가 계산한 마감일(YYYY-MM-DD)을 밀어 바꾼다.
    # v1 평가는 수치만 변조해 날짜 축이 미측정이었다.
    date_n = date_detected = date_clean = 0
    for case in _load_items("pv_dataset.json"):
        tool_out = assess_adverse_event(case["case"], awareness_date=_AWARENESS_DATE)
        deadline = tool_out.get("deadline_date")
        if not deadline:
            continue
        trusted = [json.dumps(_strip_query_echo(tool_out), ensure_ascii=False)]
        answer = f"이 케이스의 보고 기한은 {deadline} 입니다 (인지일 {_AWARENESS_DATE} 기준)."
        if verify_answer(answer, trusted).ok:
            date_clean += 1
        shifted = (_dt.date.fromisoformat(deadline) + _dt.timedelta(days=3)).isoformat()
        date_n += 1
        if not verify_answer(answer.replace(deadline, shifted, 1), trusted).ok:
            date_detected += 1

    return {
        "n_clean": clean_n, "clean_pass": clean_pass,
        "n_swap": swap_n, "swap_detected": swap_detected,
        "n_offset": off_n, "off_detected": off_detected,
        "n_direction": dir_n, "dir_detected": dir_detected,
        "n_native": nat_n, "nat_detected": nat_detected,
        "n_paraphrase": para_n, "para_pass": para_pass,
        "n_date": date_n, "date_detected": date_detected, "date_clean": date_clean,
        "SupersededFlagged": superseded_flagged,
        "HistoryModeAllowed": history_allowed,
        "no_claim_items": no_claim_items,
    }


async def _e2e_pass_rate() -> dict:
    """운영 경로 회귀 가드: 오프라인 에이전트의 실제 응답 전수가 검증을 통과하는가.

    포매터가 근거·도구 출력 밖의 수치를 만들어내면(예: 단위 환산, 하드코딩
    문자열) 여기서 깨진다 — '오프라인 모드는 통과가 정상'을 수치로 고정한다.
    """
    from src.agent.agent import RaAgent

    agent = RaAgent()
    questions = [i["question"] for i in _load_items("qa_dataset.json")]
    questions += [i["question"] for i in _load_items("abstention_dataset.json")]
    ok = 0
    failures: list[str] = []
    for q in questions:
        r = await agent.chat(q)
        if r.verification.get("ok"):
            ok += 1
        else:
            failures.append(f"{q} → {r.verification}")
    return {"n": len(questions), "ok": ok, "failures": failures}


def main() -> None:
    from eval.stats import fmt_ci

    res = evaluate()
    e2e = asyncio.run(_e2e_pass_rate())

    def row(label: str, hits: int, n: int) -> str:
        return f"{label}: {fmt_ci(hits, n)}  (n={n})"

    print("=" * 70)
    print("답변 사후 검증기 평가 (메타모픽: 정상 통과 + 합성 변조 탐지 · Wilson 95% CI)")
    print("=" * 70)
    print("— 오탐 감시(통과해야 정상) —")
    print(row("CleanPassRate       (근거 발췌 답변 통과)        ", res["clean_pass"], res["n_clean"]))
    print(row("ParaphrasePassRate  (동치 고유어 표기: 15일=보름) ", res["para_pass"], res["n_paraphrase"]))
    print(row("DateCleanPassRate   (도구 계산 마감일 인용 통과)  ", res["date_clean"], res["n_date"]))
    print("— 탐지(잡아야 정상) —")
    print(row("SwapDetection       (교차문서 수치 치환)          ", res["swap_detected"], res["n_swap"]))
    print(row("OffsetDetection     (오프셋 변조 15일→22일)       ", res["off_detected"], res["n_offset"]))
    print(row("DirectionDetection  (방향 뒤집기 이내→이후)       ", res["dir_detected"], res["n_direction"]))
    print(row("NativeSwapDetection (고유어 치환 15일→열흘)       ", res["nat_detected"], res["n_native"]))
    print(row("DateShiftDetection  (마감일 시프트 +3일)          ", res["date_detected"], res["n_date"]))
    print(f"폐지본 인용 감지: {'✓' if res['SupersededFlagged'] else '✗ 실패'}"
          f" · 이력 조회 모드 허용: {'✓' if res['HistoryModeAllowed'] else '✗ 실패'}")
    print("— 운영 경로 실측 —")
    print(row("E2EPassRate         (오프라인 에이전트 실응답)    ", e2e["ok"], e2e["n"]))
    for f in e2e["failures"]:
        print(f"  - E2E 실패: {f}")
    print("-" * 70)
    print("해석: 탐지율 5축은 '근거에 없는 값·방향'으로 합성한 변조라 1.0이 정상이며,")
    print("      깨지는 순간이 곧 검증기(클레임 추출·대조) 회귀다 — 회귀 고정 핀.")
    print("      CleanPassRate 도 절반은 구성적이다(발췌⊆신뢰 소스): 잡는 것은 답변/근거")
    print("      추출의 비대칭 회귀다. 구성이 개입하지 않는 실측은 E2EPassRate —")
    print("      포매터가 근거 밖 수치를 만들면 여기가 먼저 깨진다.")
    print("      방향·고유어·날짜 축은 v1에서 측정 자체가 없던 사각지대의 가시화이며,")
    print("      ParaphrasePassRate 는 새 사전이 오탐을 만들지 않는지의 반대 방향 감시다.")
    print(f"      (수치 클레임이 없는 문항 {res['no_claim_items']}건은 변조 대상에서 제외 — 은폐가 아니라 명시)")


if __name__ == "__main__":
    main()
