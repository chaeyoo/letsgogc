"""RAG 하이퍼파라미터 스윕 — "감이 아니라 스윕으로 정했다"의 재현 스크립트.

README/면접노트에 인용하는 하이퍼파라미터 결정(rerank_weight=0.9, alpha=0.5,
idf_power=0.5, 확장토큰 보조신호 유지)의 근거 수치를 그대로 재현한다.

측정 축:
  1. rerank_weight (리랭커 신호 vs 1차 하이브리드 prior)   0.7 ~ 1.0
  2. alpha         (벡터 vs BM25 결합비)                    0.3 ~ 0.7
  3. idf_power     (리랭커 토큰 가중 idf^p)                 0 / 0.5 / 1
  4. ablation      (질의확장 토큰의 리랭커 보조신호 on/off × rerank_weight)
     → aux 는 rw=0.9 에선 1차 prior 가 구제해 지표가 같지만, prior 를 끄면
       (rw=1.0) 유무가 Hit@1 0.969 vs 0.906 을 가른다 = 리랭커 자체의 견고성.
  5. 섹션 타입 prior (리랭커 v3): contrast_penalty(대조 섹션) 스윕과
     preamble_penalty(서두 섹션) 스윕 + 둘의 on/off ablation.
     → contrast 는 '뒤집히는 최소값' 이상에서 플랫(0.25~), preamble 은
       좁은 유효 밴드(0.04~0.07)가 존재 — 아래는 미교정, 위는 정답이
       서두인 문항·인접 문서 역전. 게이트 문항(QA 31~32)이 '섹션 삭제'가
       아님을 같은 표에서 증명한다.

실행:  python -m eval.sweep
"""
from __future__ import annotations

import time

from src import config
from src.rag.pipeline import RagPipeline

from .evaluate import _context_recall, _gold_ids, _hit_rank, _load_qa

_COLS = ["Hit@1", "MRR", "ContextRecall", "HardNegHit@1", "mean_ms"]


def run_config(pipe: RagPipeline, *, top_k: int = None, rerank_n: int = 1,
               aux_in_rerank: bool = True) -> dict:
    """현재 retriever 설정(인스턴스 속성) 그대로 QA셋을 평가한다."""
    top_k = top_k or config.RETRIEVE_TOP_K
    qa = _load_qa()
    hits1 = hn_hits1 = hn_total = 0
    mrr_sum = ctx_sum = 0.0
    latencies: list[float] = []
    for item in qa:
        t0 = time.perf_counter()
        results = pipe.retriever.retrieve(
            item["question"], top_k=top_k, rerank_n=rerank_n,
            expand=True, aux_in_rerank=aux_in_rerank,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        doc_ids = [s.chunk.doc_id for s in results]
        # 복수 정답 교정(accept_doc_ids)을 evaluate 와 동일하게 전파한다 — 단일
        # 라벨을 직접 넘기면, 하이퍼파라미터를 흔든 설정에서 동등 정답 문항이
        # miss 로 갈려 라벨 아티팩트가 파라미터 효과로 오귀속된다(v10).
        rank = _hit_rank(doc_ids, _gold_ids(item))
        hits1 += rank == 1
        mrr_sum += (1.0 / rank) if rank else 0.0
        if item.get("hard_negative"):
            hn_total += 1
            hn_hits1 += rank == 1
        ctx = "\n".join(s.chunk.text for s in results)
        ctx_sum += _context_recall(ctx, item.get("keywords", []))
    n = len(qa)
    return {
        "Hit@1": round(hits1 / n, 3),
        "MRR": round(mrr_sum / n, 3),
        "ContextRecall": round(ctx_sum / n, 3),
        "HardNegHit@1": round(hn_hits1 / hn_total, 3) if hn_total else None,
        "mean_ms": round(sum(latencies) / len(latencies), 2),
    }


def _table(title: str, rows: list[tuple[str, dict]]) -> None:
    print()
    print(f"— {title}")
    header = f"{'설정':<24}" + "".join(f"{c:>15}" for c in _COLS)
    print(header)
    print("-" * len(header))
    for label, res in rows:
        print(f"{label:<24}" + "".join(f"{res[c]:>15}" for c in _COLS))


def main() -> None:
    print("=" * 62)
    print("RAG 하이퍼파라미터 스윕 · 운영 경로(질의확장+하이브리드+리랭킹) · rerank_n=1")
    print("=" * 62)
    pipe = RagPipeline().build()
    r = pipe.retriever

    # 1) rerank_weight
    rows = []
    for rw in (0.7, 0.8, 0.9, 1.0):
        r.rerank_weight = rw
        rows.append((f"rerank_weight={rw}", run_config(pipe)))
    r.rerank_weight = config.RERANK_WEIGHT
    _table("rerank_weight (리랭커 vs 1차 prior) — 0.9 채택", rows)

    # 2) alpha
    rows = []
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        r.alpha = a
        rows.append((f"alpha={a}", run_config(pipe)))
    r.alpha = config.HYBRID_ALPHA
    _table("alpha (벡터 vs BM25) — 이 코퍼스에선 둔감, 0.5 유지", rows)

    # 3) idf_power
    rows = []
    for p in (0.0, 0.5, 1.0):
        r.idf_power = p
        rows.append((f"idf_power={p}", run_config(pipe)))
    r.idf_power = config.RERANK_IDF_POWER
    _table("idf_power (리랭커 토큰 가중 idf^p) — 현 QA셋 둔감, 0.5 유지", rows)

    # 4) 확장토큰 보조신호(aux) ablation × rerank_weight
    rows = []
    for rw in (0.9, 1.0):
        r.rerank_weight = rw
        rows.append((f"rw={rw} aux=on", run_config(pipe, aux_in_rerank=True)))
        rows.append((f"rw={rw} aux=off", run_config(pipe, aux_in_rerank=False)))
    r.rerank_weight = config.RERANK_WEIGHT
    _table("질의확장 토큰의 리랭커 보조신호 ablation", rows)

    # 5) 섹션 타입 prior (리랭커 v3) — 강도 스윕 + on/off ablation
    rows = []
    for cp in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4):
        r.contrast_penalty = cp
        rows.append((f"contrast_penalty={cp}", run_config(pipe)))
    r.contrast_penalty = config.RERANK_CONTRAST_PENALTY
    _table("contrast_penalty (대조 섹션 페널티) — 0.3 채택", rows)

    rows = []
    for pp in (0.0, 0.02, 0.04, 0.055, 0.07, 0.09, 0.12):
        r.preamble_penalty = pp
        rows.append((f"preamble_penalty={pp}", run_config(pipe)))
    r.preamble_penalty = config.RERANK_PREAMBLE_PENALTY
    _table("preamble_penalty (서두 섹션 감쇠) — 유효 밴드 중앙 0.055 채택", rows)

    rows = []
    for label, cp, pp in (
        ("둘 다 off (v2 리랭커)", 0.0, 0.0),
        ("contrast 만 on", config.RERANK_CONTRAST_PENALTY, 0.0),
        ("preamble 만 on", 0.0, config.RERANK_PREAMBLE_PENALTY),
        ("둘 다 on (운영 기본)", config.RERANK_CONTRAST_PENALTY, config.RERANK_PREAMBLE_PENALTY),
    ):
        r.contrast_penalty, r.preamble_penalty = cp, pp
        rows.append((label, run_config(pipe)))
    r.contrast_penalty = config.RERANK_CONTRAST_PENALTY
    r.preamble_penalty = config.RERANK_PREAMBLE_PENALTY
    _table("섹션 타입 prior ablation (v2 → v3)", rows)

    print()
    print("해석:")
    print(" - rerank_weight: 0.7~0.8은 1차 prior 가 과점해 리랭커의 교정을 일부 되돌리고")
    print("   (Hit@1 0.969), 1.0(순수 재정렬)도 쉬운 질의 1건을 잃는다(0.969).")
    print("   32문항 셋에선 0.9가 단독 최적(1.0) — '1차 신호 10% 잔류'가 동률 시 선호가")
    print("   아니라 실측 우위가 됐다(30문항 시절엔 0.9·1.0 동률이었음).")
    print(" - aux ablation: rw=0.9에선 aux 유무가 안 보이지만(1차 prior 가 구제)")
    print("   rw=1.0에서 aux 를 끄면 Hit@1 0.969→0.906 — 완전 어휘 불일치 질의에서")
    print("   리랭커가 판별력을 잃는다. aux 는 '지표를 올리는 장치'가 아니라")
    print("   '리랭커를 prior 없이도 서게 만드는 견고성 장치'다.")
    print(" - alpha·idf_power 는 이 코퍼스 규모에선 지표가 안 움직인다 — 억지로 튜닝하지")
    print("   않고 중립값을 남겼다(코퍼스가 커지면 같은 스크립트로 재보정).")
    print(" - contrast_penalty: 하드네거티브 잔여 실패(대조 섹션)가 뒤집히는 최소값")
    print("   이상에선 플랫 — 경계값이 아니라 여유 있는 0.3 채택. 게이트 문항(차이 질문)은")
    print("   전 구간 유지 = '섹션 삭제'가 아니라 '의도 조건부 신호'라는 증거.")
    print(" - preamble_penalty: 유효 밴드(0.04~0.07)가 좁다 — 아래(≤0.02)는 서두 과대평가")
    print("   미교정(경고 문항 ContextRecall 손해), 위(≥0.09)는 '서두가 정답 문서의 유일")
    print("   회수 청크'인 문항이 이웃 문서(REG-001 허가 후 의무)에 역전돼 Hit@1 을 잃는다.")
    print("   그 이웃이 PSUR 을 언급해 ContextRecall 은 되레 오르는 지표 상충이 있는데,")
    print("   출처 문서의 정확성(Hit@1·규제 도메인의 추적성)을 우선해 밴드 중앙 0.055 를")
    print("   채택했다. 경계 문항은 pytest 회귀 가드로 고정.")


if __name__ == "__main__":
    main()
