"""검증기의 검증(meta-evaluation) — "그 검증기는 누가 검증하나"에 대한 답.

답변 사후 검증(src/verify)은 운영에서 모든 응답이 통과하는 안전장치다.
안전장치는 두 방향 모두에서 측정돼야 한다:

  (1) 잡아야 할 것을 잡는가 — 근거에 없는 수치가 섞인 답변을 탐지하는가.
  (2) 잡지 말아야 할 것을 흘려보내는가 — 정상(근거 발췌) 답변에 오탐을
      내지 않는가. 오탐이 잦으면 담당자가 경고를 무시하기 시작하고
      (alert fatigue), 그 순간 검증 계층 전체가 죽는다.

정상 답변의 '오류 버전'을 사람이 라벨링하는 대신, 정상 답변에 **결정론적
변조(seeded corruption)**를 가해 오류 케이스를 합성한다(메타모픽 테스트).
변조는 실제 환각의 형태를 모사한다:

  - 교차문서 치환(cross-doc swap): 답변 속 수치를 '코퍼스의 다른 문서에는
    실제로 존재하는' 같은 단위의 값으로 바꾼다. LLM 환각의 전형인
    '그럴듯한 혼동'(의료기기 80일 ↔ 의약품 120일)이며, 검증이 전역
    코퍼스가 아니라 **이 질의의 근거**를 기준으로 하는지를 함께 확인한다.
  - 오프셋 변조(offset): 수치에 상수를 더한다(15일→22일). 날짜 연산
    오류·자릿수 실수 형태의 환각 모사.

정직한 프레이밍: 변조는 '근거에 없는 값'으로 만들어지므로, 탐지율 1.0은
새로운 발견이 아니라 **검증기 회귀를 고정하는 핀**이다 — 누군가 클레임
추출 정규식이나 단위 목록을 건드려 탐지가 새기 시작하면 이 수치가 먼저
떨어진다. 반대로 CleanPassRate 는 실측이다: 포매터·발췌 경로가 근거 밖
수치를 만들어내는 순간 1.0이 깨진다.

실행:  python -m eval.verify_eval
"""
from __future__ import annotations

import asyncio
import json

from src import config
from src.mcp_server.server import search_regulations
from src.rag.loader import load_documents
from src.verify.verifier import extract_claims, verify_answer

_OFFSETS = (7, 11, 13, 17, 23)  # 오프셋 변조 시 근거와의 우연한 충돌을 피하는 결정론적 시퀀스


def _load_qa() -> list[dict]:
    path = config.BASE_DIR / "eval" / "qa_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _corpus_claim_pool() -> set[tuple[str, str]]:
    """전 규제문서에서 (수치, 단위) 풀을 수집 — 교차문서 치환 후보."""
    pool: set[tuple[str, str]] = set()
    for doc in load_documents(config.REG_DIR):
        nums, _ = extract_claims(doc.text)
        pool |= nums
    return pool


def _faithful_answer(search_data: dict) -> str:
    """검색 최상위 근거의 발췌로 '정상(grounded) 답변'을 합성한다."""
    results = search_data.get("results", [])
    if not results:
        return ""
    body = results[0]["text"].split("]\n", 1)[-1].strip()
    return body[:400]


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
    qa = _load_qa()
    pool = _corpus_claim_pool()

    clean_n = clean_pass = 0
    swap_n = swap_detected = 0
    off_n = off_detected = 0
    no_claim_items = 0

    for item in qa:
        data = search_regulations(item["question"], top_n=3)
        trusted = [json.dumps(data, ensure_ascii=False)]
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
                swap_n += 1
                corrupted = answer.replace(f"{num}{unit}", f"{swap}{unit}", 1)
                if not verify_answer(corrupted, trusted).ok:
                    swap_detected += 1
            off = _offset_value(claim, trusted_nums)
            if off is not None:
                off_n += 1
                corrupted = answer.replace(f"{num}{unit}", f"{off}{unit}", 1)
                if not verify_answer(corrupted, trusted).ok:
                    off_detected += 1

    # (3) 폐지본 인용 감지 — 버전 검증 축
    sup_cite = [{"doc_id": "REG-013", "status": "superseded"}]
    superseded_flagged = not verify_answer("이상사례는 30일 이내 보고한다", ["30일"], sup_cite).ok
    history_allowed = verify_answer(
        "구판 기준은 30일이었다", ["30일"], sup_cite, allow_superseded=True
    ).ok

    return {
        "n_clean": clean_n,
        "CleanPassRate": round(clean_pass / clean_n, 3) if clean_n else None,
        "n_swap": swap_n,
        "SwapDetection": round(swap_detected / swap_n, 3) if swap_n else None,
        "n_offset": off_n,
        "OffsetDetection": round(off_detected / off_n, 3) if off_n else None,
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

    def _items(name: str) -> list[dict]:
        return json.loads((config.BASE_DIR / "eval" / name).read_text(encoding="utf-8"))["items"]

    agent = RaAgent()
    questions = [i["question"] for i in _items("qa_dataset.json")]
    questions += [i["question"] for i in _items("abstention_dataset.json")]
    ok = 0
    failures: list[str] = []
    for q in questions:
        r = await agent.chat(q)
        if r.verification.get("ok"):
            ok += 1
        else:
            failures.append(f"{q} → {r.verification}")
    return {"n": len(questions), "E2EPassRate": round(ok / len(questions), 3), "failures": failures}


def main() -> None:
    res = evaluate()
    e2e = asyncio.run(_e2e_pass_rate())
    print("=" * 62)
    print("답변 사후 검증기 평가 (메타모픽: 정상 통과 + 합성 변조 탐지)")
    print("=" * 62)
    print(f"CleanPassRate   (정상 발췌 답변 통과 — 오탐 감시): {res['CleanPassRate']:.3f}  (n={res['n_clean']})")
    print(f"SwapDetection   (교차문서 수치 치환 탐지)        : {res['SwapDetection']:.3f}  (n={res['n_swap']})")
    print(f"OffsetDetection (오프셋 변조 탐지)               : {res['OffsetDetection']:.3f}  (n={res['n_offset']})")
    print(f"폐지본 인용 감지: {'✓' if res['SupersededFlagged'] else '✗ 실패'}"
          f" · 이력 조회 모드 허용: {'✓' if res['HistoryModeAllowed'] else '✗ 실패'}")
    print(f"E2EPassRate     (오프라인 에이전트 실응답 통과)  : {e2e['E2EPassRate']:.3f}  (n={e2e['n']})")
    for f in e2e["failures"]:
        print(f"  - E2E 실패: {f}")
    print("-" * 62)
    print("해석: 탐지율(Swap/Offset)은 '근거에 없는 값'으로 합성한 변조라 1.0이 정상이며,")
    print("      1.0이 깨지는 순간이 곧 검증기(클레임 추출·대조) 회귀다 — 회귀 고정 핀.")
    print("      CleanPassRate·E2EPassRate 는 실측이다: 오탐이 생기면 경고가 노이즈가 되고")
    print("      (alert fatigue), 포매터가 근거 밖 수치를 만들면 E2E 가 먼저 깨진다.")
    print(f"      (수치 클레임이 없는 문항 {res['no_claim_items']}건은 변조 대상에서 제외 — 은폐가 아니라 명시)")


if __name__ == "__main__":
    main()
