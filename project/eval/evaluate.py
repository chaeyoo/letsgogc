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
    """검색 모드별 결과 반환. mode: 'vector' | 'hybrid' | 'rerank'."""
    if mode == "vector":
        # 벡터(TF-IDF 코사인) 단독
        return pipe.retriever.store.search(question, top_k)[:rerank_n]
    if mode == "hybrid":
        # 벡터+BM25 하이브리드 (리랭킹 없음)
        return pipe.retriever._hybrid(question, top_k)[:rerank_n]
    # 하이브리드 + 리랭킹
    return pipe.retriever.retrieve(question, top_k=top_k, rerank_n=rerank_n)


def evaluate(mode: str, top_k: int, rerank_n: int) -> dict:
    pipe = RagPipeline().build()
    qa = _load_qa()
    hits1 = hits3 = 0
    mrr_sum = 0.0
    ctx_recall_sum = 0.0

    for item in qa:
        results = _retrieve(pipe, mode, item["question"], top_k, rerank_n)
        doc_ids = [s.chunk.doc_id for s in results]
        rank = _hit_rank(doc_ids, item["relevant_doc_id"])
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        mrr_sum += (1.0 / rank) if rank else 0.0

        ctx_text = "\n".join(s.chunk.text for s in results)
        ctx_recall_sum += _context_recall(ctx_text, item.get("keywords", []))

    n = len(qa)
    return {
        "n": n,
        "Hit@1": round(hits1 / n, 3),
        "Hit@3": round(hits3 / n, 3),
        "MRR": round(mrr_sum / n, 3),
        "ContextRecall": round(ctx_recall_sum / n, 3),
    }


def main() -> None:
    # 리랭킹 효과를 순위에 민감하게 보려고 최종 1건(rerank_n=1)으로 평가
    top_k, rerank_n = config.RETRIEVE_TOP_K, 1
    qa_n = len(_load_qa())
    print("=" * 62)
    print(f"RAG 검색 평가 · QA {qa_n}건 · top_k={top_k}, 최종 rerank_n={rerank_n}")
    print("=" * 62)

    cols = [
        ("① 벡터만", "vector"),
        ("② 하이브리드", "hybrid"),
        ("③ +리랭킹", "rerank"),
    ]
    res = {name: evaluate(mode, top_k, rerank_n) for name, mode in cols}

    header = f"{'지표':<16}" + "".join(f"{name:>14}" for name, _ in cols)
    print(header)
    print("-" * len(header))
    for k in ["Hit@1", "MRR", "ContextRecall"]:
        row = f"{k:<16}" + "".join(f"{res[name][k]:>14}" for name, _ in cols)
        print(row)
    print("-" * len(header))
    print("해석: 하이브리드는 키워드+의미를 결합해 회수율을, 리랭킹은 정답을")
    print("      1순위로 끌어올려 Hit@1·MRR 을 개선한다 (= RAG '최적화'의 효과).")


if __name__ == "__main__":
    main()
