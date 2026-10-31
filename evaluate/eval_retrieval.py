"""
python -m evaluate.eval_retrieval --dataset QA/datasets/gold/all.jsonl --source-mapping logs/legacy_source_mapping.json --output logs/retrieval_eval_baseline.json
"""


from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rag.schemas import AnswerGuardState, RetrievedChunk
from rag_data.source_mapping import SourceMapping, load_mapping_dict

from .metrics import mrr, precision_at_k, recall_at_k, save_retrieval_plots

TARGET_COLLECTIONS = {
    "Rules_Basic": "rules",
    "Mechanics_Glossary": "rules",
    "Cards_Facts": "cards",
    "Card_FAQ_Interactions": "faq",
    "Tournament_Formats": "rules",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def load_source_mapping(path: Path | None) -> SourceMapping:
    if path is None:
        return SourceMapping()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return load_mapping_dict(raw)


def guard_state_for_score(
    score: float | None,
    refuse_threshold: float,
    warn_threshold: float,
) -> AnswerGuardState:
    if score is None or score < refuse_threshold:
        return AnswerGuardState.REFUSE
    return AnswerGuardState.WARN if score < warn_threshold else AnswerGuardState.OK


def evaluate_case(
    case: dict[str, Any],
    chunks: list[RetrievedChunk],
    source_mapping: SourceMapping,
    ks: tuple[int, ...] = (1, 3, 5),
    refuse_threshold: float = 0.1,
    warn_threshold: float = 0.5,
) -> dict[str, Any]:
    retrieved_ids = [chunk.chunk.id for chunk in chunks]
    top_score = chunks[0].score if chunks else None
    result: dict[str, Any] = {
        "test_id": case["test_id"],
        "category": case["category"],
        "answerability": case["answerability"],
        "retrieved_ids": retrieved_ids,
        "retrieved_collections": sorted({chunk.chunk.collection for chunk in chunks}),
        "top_score": top_score,
        "predicted_guard_state": guard_state_for_score(
            top_score, refuse_threshold, warn_threshold,
        ).name,
    }
    if case["answerability"] == "unanswerable":
        result["status"] = "separate_guard_eval"
        return result

    expected = case["expected_source_ids"]
    missing = sorted(source_id for source_id in expected if source_id not in source_mapping.entries)
    target_collection = TARGET_COLLECTIONS.get(case["category"])
    gold_ids = source_mapping.chunks_for(expected, target_collection)
    result.update({
        "target_collection": target_collection,
        "expected_source_ids": expected,
        "missing_mapping": missing,
        "status": "unmapped" if missing else "evaluated",
    })
    result["metrics"] = (
        {f"recall@{k}": None for k in ks}
        | {f"precision@{k}": None for k in ks}
        | {"mrr": None}
    ) if missing else {
        **{f"recall@{k}": recall_at_k(retrieved_ids, gold_ids, k) for k in ks},
        **{f"precision@{k}": precision_at_k(retrieved_ids, gold_ids, k) for k in ks},
        "mrr": mrr(retrieved_ids, gold_ids),
    }
    return result


async def evaluate_retriever(
    retriever: Any,
    dataset_path: Path,
    source_mapping_path: Path | None = None,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    rrf_limit: int = 20,
    rerank_limit: int = 20,
    use_reranker: bool = True,
    refuse_threshold: float = 0.1,
    warn_threshold: float = 0.5,
) -> dict[str, Any]:
    mapping = load_source_mapping(source_mapping_path)
    results = []
    for case in tqdm(load_jsonl(dataset_path), desc="Retrieval evaluation", unit="case"):
        target = TARGET_COLLECTIONS.get(case["category"])
        kwargs: dict[str, Any] = {
            "top_k": max(max(ks), top_k),
            "dense_chunks_limit": dense_limit,
            "sparse_chunks_limit": sparse_limit,
            "rrf_limit": rrf_limit,
            "rerank_limit": rerank_limit,
            "use_reranker": use_reranker,
        }
        if target:
            kwargs["collections"] = [target]
        chunks = await retriever.retrieve(case["question"], **kwargs)
        results.append(evaluate_case(
            case, chunks, mapping, ks, refuse_threshold, warn_threshold,
        ))

    evaluated = [row for row in results if row["status"] == "evaluated"]
    metric_names = [f"recall@{k}" for k in ks] + [f"precision@{k}" for k in ks] + ["mrr"]
    aggregate = {
        metric: (
            sum(row["metrics"][metric] for row in evaluated) / len(evaluated)
            if evaluated else None
        )
        for metric in metric_names
    }
    return {
        "dataset": str(dataset_path),
        "source_mapping": str(source_mapping_path) if source_mapping_path else None,
        "settings": {
            "top_k": top_k, "dense_limit": dense_limit, "sparse_limit": sparse_limit,
            "rrf_limit": rrf_limit, "rerank_limit": rerank_limit,
            "use_reranker": use_reranker, "refuse_threshold": refuse_threshold,
            "warn_threshold": warn_threshold,
        },
        "counts": {
            "total": len(results),
            "evaluated": len(evaluated),
            "unmapped": sum(row["status"] == "unmapped" for row in results),
            "unanswerable": sum(row["answerability"] == "unanswerable" for row in results),
        },
        "aggregate": aggregate,
        "cases": results,
    }


async def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the QA gold set.")
    parser.add_argument("--dataset", type=Path, default=Path("QA/datasets/gold/all.jsonl"))
    parser.add_argument("--source-mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/retrieval_eval.json"))
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory for detailed validation plots; defaults to <output-stem>_plots.",
    )
    args = parser.parse_args()

    import rag

    await rag.init()
    try:
        report = await evaluate_retriever(rag._retriever, args.dataset, args.source_mapping)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        plots_dir = args.plots_dir or args.output.parent / f"{args.output.stem}_plots"
        saved_plots = save_retrieval_plots(report, plots_dir)
        print(f"Saved {len(saved_plots)} validation plots to {plots_dir}")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(run_cli())
