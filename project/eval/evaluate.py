"""RAG 검색 품질 평가 (RAGAS 스타일 경량 버전).

측정 지표 (검색기 성능):
  - Hit@k           : 정답 문서가 상위 k개 안에 회수됐는가 (Recall 관점)
  - MRR             : 정답 문서의 첫 등장 순위의 역수 평균 (상위에 올렸는가)
  - Context Recall  : 정답 근거에 있어야 할 핵심어(keywords)가 회수된 context에 존재하는 비율
                      (= 생성 단계가 정답을 말할 '재료'를 검색이 제공했는가 → faithfulness 상한)

리랭킹 on/off 를 비교해 'RAG 최적화(리랭킹)'의 효과를 수치로 보여준다.

실행:  python -m eval.evaluate
"""
from __future__ import annotations

import json
import time

from src import config
from src.rag.pipeline import RagPipeline


def _load_qa() -> list[dict]:
    path = config.BASE_DIR / "eval" / "qa_dataset.json"
    return [x for x in json.loads(path.read_text(encoding="utf-8"))["items"]]


def _hit_rank(retrieved_doc_ids: list[str], gold: str) -> int:
    """정답 문서의 순위(1-based). 없으면 0."""
    for i, d in enumerate(retrieved_doc_ids, 1):
        if d == gold:
            return i
    return 0


def _context_recall(context_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k.lower() in context_text.lower())
    return hit / len(keywords)


def _retrieve(pipe: RagPipeline, mode: str, question: str, top_k: int, rerank_n: int):
    """검색 모드별 결과 반환. mode: 'vector' | 'hybrid' | 'rerank' | 'expand'.

    네 모드 모두 동일한 버전 인지 후보군(폐지본 제외)에서 검색해 공정 비교한다.
    """
    if mode == "vector":
        # 벡터(TF-IDF 코사인) 단독
        return pipe.retriever.vector_search(question, top_k)[:rerank_n]
    if mode == "hybrid":
        # 벡터+BM25 하이브리드 (리랭킹 없음)
        return pipe.retriever._hybrid(question, top_k)[:rerank_n]
    if mode == "rerank":
        # 하이브리드 + 리랭킹 (질의 확장 없음)
        return pipe.retriever.retrieve(
            question, top_k=top_k, rerank_n=rerank_n, expand=False
        )
    # 도메인 동의어 질의 확장 + 하이브리드 + 리랭킹 (운영 기본값)
    return pipe.retriever.retrieve(question, top_k=top_k, rerank_n=rerank_n, expand=True)


def evaluate(mode: str, top_k: int, rerank_n: int, pipe: RagPipeline | None = None) -> dict:
    pipe = pipe or RagPipeline().build()
    qa = _load_qa()
    hits1 = hits3 = 0
    hn_hits1 = hn_total = 0          # hard-negative 부분집합 Hit@1
    mrr_sum = 0.0
    ctx_recall_sum = 0.0
    latencies: list[float] = []

    for item in qa:
        t0 = time.perf_counter()
        results = _retrieve(pipe, mode, item["question"], top_k, rerank_n)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        doc_ids = [s.chunk.doc_id for s in results]
        rank = _hit_rank(doc_ids, item["relevant_doc_id"])
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        mrr_sum += (1.0 / rank) if rank else 0.0

        if item.get("hard_negative"):
            hn_total += 1
            if rank == 1:
                hn_hits1 += 1

        ctx_text = "\n".join(s.chunk.text for s in results)
        ctx_recall_sum += _context_recall(ctx_text, item.get("keywords", []))

    n = len(qa)
    latencies.sort()
    return {
        "n": n,
        "counts": {"hits1": hits1, "hn_hits1": hn_hits1, "hn_total": hn_total},
        "Hit@1": round(hits1 / n, 3),
        "Hit@3": round(hits3 / n, 3),
        "MRR": round(mrr_sum / n, 3),
        "ContextRecall": round(ctx_recall_sum / n, 3),
        "HardNegHit@1": round(hn_hits1 / hn_total, 3) if hn_total else None,
        "p50_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def main() -> None:
    # 리랭킹 효과를 순위에 민감하게 보려고 최종 1건(rerank_n=1)으로 평가
    top_k, rerank_n = config.RETRIEVE_TOP_K, 1
    qa_n = len(_load_qa())
    print("=" * 62)
    print(f"RAG 검색 평가 · QA {qa_n}건 · top_k={top_k}, 최종 rerank_n={rerank_n}")
    print("=" * 62)

    # 인덱스는 1회만 구축해 세 모드가 공유(공정 비교 + 빠른 실행)
    pipe = RagPipeline().build()
    cols = [
        ("① 벡터만", "vector"),
        ("② 하이브리드", "hybrid"),
        ("③ +리랭킹", "rerank"),
        ("④ +질의확장", "expand"),
    ]
    res = {name: evaluate(mode, top_k, rerank_n, pipe=pipe) for name, mode in cols}

    header = f"{'지표':<16}" + "".join(f"{name:>14}" for name, _ in cols)
    print(header)
    print("-" * len(header))
    for k in ["Hit@1", "MRR", "ContextRecall", "HardNegHit@1", "mean_ms"]:
        row = f"{k:<16}" + "".join(f"{res[name][k]:>14}" for name, _ in cols)
        print(row)
    print("-" * len(header))
    print("해석:")
    print(" - HardNegHit@1: 어휘가 겹치는 유사문서(하드네거티브)가 섞인 문항에서의 정확도.")
    print("   벡터 단독은 어휘 유사도에 끌려 오답 문서를 고르기 쉽고, 하이브리드(+BM25)와")
    print("   리랭킹이 정답 문서를 1순위로 되돌린다 → 이 컬럼이 'RAG 최적화'의 핵심 근거.")
    print(" - mean_ms: 질의당 평균 검색 지연. 리랭킹은 정밀도를 얻는 대신 지연이 늘어난다")
    print("   (정밀도↔속도 트레이드오프). 폐지본(REG-013)은 버전 필터로 기본 제외된다.")
    print(" - ④ 질의확장: 도메인 동의어 사전('부작용'→'이상사례', '심각'→'중대한')으로")
    print("   어휘 불일치를 메워 Hit@1이 오른다. 확장은 1단계 회수에 전 가중으로,")
    print("   2단계 리랭킹엔 절반 가중의 보조 신호로만 반영(정밀도 희석 방지).")
    print("   하이퍼파라미터 스윕·ablation 근거는 python -m eval.sweep 으로 재현.")

    # --- 통계적 정직성: 소표본 지표에 95% 신뢰구간(Wilson) 병기 ---
    from eval.stats import fmt_ci

    best = res["④ +질의확장"]
    c, n = best["counts"], best["n"]
    print()
    print("통계적 정직성 (운영 기본 ④, Wilson 95% CI):")
    print(f" - Hit@1        : {fmt_ci(c['hits1'], n)}  (n={n})")
    print(f" - HardNegHit@1 : {fmt_ci(c['hn_hits1'], c['hn_total'])}  (n={c['hn_total']})")
    print("   → 1.000은 '완벽'이 아니라 '이 표본에서 실패 관측 0'이라는 뜻이다.")
    print("     구간 하한이 진짜 성능의 보수적 추정 — 개선 주장은 구간이 갈릴 때만 한다.")

    # --- 임베딩 provider 비교(pluggable 실증) ---
    print()
    print("=" * 62)
    print("임베딩 provider 비교 (동일 하이브리드+리랭킹 파이프라인)")
    print("=" * 62)
    prov_header = f"{'provider':<16}" + "".join(f"{k:>14}" for k in ["Hit@1", "MRR", "ContextRecall"])
    print(prov_header)
    print("-" * len(prov_header))
    for kind in ["tfidf", "hashing"]:
        p = RagPipeline(embedder_kind=kind).build()
        r = evaluate("rerank", top_k, rerank_n, pipe=p)
        row = f"{kind:<16}" + "".join(f"{r[k]:>14}" for k in ["Hit@1", "MRR", "ContextRecall"])
        print(row)
    print("-" * len(prov_header))
    print("해석: EmbeddingProvider 인터페이스로 임베더를 교체해도 파이프라인은 무수정 동작한다.")
    print("      실무에선 여기에 상용 임베딩(VoyageEmbedder, VOYAGE_API_KEY)을 끼운다.")


if __name__ == "__main__":
    main()
