"""RAG 하이퍼파라미터 스윕 — "감이 아니라 스윕으로 정했다"의 재현 스크립트.

README/면접노트에 인용하는 하이퍼파라미터 결정(rerank_weight=0.9, alpha=0.5,
idf_power=0.5, 확장토큰 보조신호 유지)의 근거 수치를 그대로 재현한다.

측정 축:
  1. rerank_weight (리랭커 신호 vs 1차 하이브리드 prior)   0.7 ~ 1.0
  2. alpha         (벡터 vs BM25 결합비)                    0.3 ~ 0.7
  3. idf_power     (리랭커 토큰 가중 idf^p)                 0 / 0.5 / 1
  4. ablation      (질의확장 토큰의 리랭커 보조신호 on/off × rerank_weight)
     → aux 는 rw=0.9 에선 1차 prior 가 구제해 지표가 같지만, prior 를 끄면
       (rw=1.0) 유무가 Hit@1 0.967 vs 0.867 을 가른다 = 리랭커 자체의 견고성.

실행:  python -m eval.sweep
"""
from __future__ import annotations

import time

from src import config
from src.rag.pipeline import RagPipeline

from .evaluate import _context_recall, _hit_rank, _load_qa

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
        rank = _hit_rank(doc_ids, item["relevant_doc_id"])
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

    print()
    print("해석:")
    print(" - rerank_weight: 0.7~0.8은 1차 prior 가 과점해 리랭커의 하드네거티브 교정을")
    print("   되돌린다(Hit@1 0.933). 0.9와 1.0은 동률 — 1차 신호를 완전히 버리기보다")
    print("   10%를 남기는 쪽(0.9)을 채택(쉬운 질의 안정화 + 리랭커 결함 완충).")
    print(" - aux ablation: rw=0.9에선 aux 유무가 안 보이지만(1차 prior 가 구제)")
    print("   rw=1.0에서 aux 를 끄면 Hit@1 0.967→0.867 — 완전 어휘 불일치 질의에서")
    print("   리랭커가 판별력을 잃는다. aux 는 '지표를 올리는 장치'가 아니라")
    print("   '리랭커를 prior 없이도 서게 만드는 견고성 장치'다.")
    print(" - alpha·idf_power 는 이 코퍼스 규모에선 지표가 안 움직인다 — 억지로 튜닝하지")
    print("   않고 중립값을 남겼다(코퍼스가 커지면 같은 스크립트로 재보정).")


if __name__ == "__main__":
    main()
