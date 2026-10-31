"""
python -m evaluate.eval_llm_answers --dataset QA/datasets/gold/all.jsonl --source-mapping logs/legacy_source_mapping.json --output logs/llm_eval_baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from string import Template
from typing import Any, TypedDict

from tqdm import tqdm

import rag
from core.config import get_settings
from rag.llm import LLMClient, LLMResponse, Message, create_llm
from rag.schemas import Answer, AnswerGuardState

from .eval_retrieval import evaluate_case
from .metrics import save_llm_plots
from rag_data.source_mapping import SourceMapping

JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий эксперт по оценке ответов RAG. "
    "Оценивай только факты, которые можно проверить по gold-критериям и контексту. "
    "Верни только корректный JSON без Markdown."
)

JUDGE_USER_TEMPLATE = Template(
    """Вопрос пользователя:
<question>
$question
</question>

Ожидаемый ответ:
<reference>
$reference
</reference>

Обязательные факты:
<expected_facts>
$expected_facts
</expected_facts>

Запрещённые утверждения:
<forbidden_claims>
$forbidden_claims
</forbidden_claims>

Ответ RAG:
<answer>
$answer
</answer>

Состояние Answer Guard: $guard_state
Тип вопроса: $answerability

Контекст RAG:
<context>
$context
</context>

Верни JSON строго такой структуры:
{
  "label": "PASS|PARTIAL|FAIL",
  "score": 0,
  "factual_correctness": 0,
  "faithfulness": 0,
  "answer_relevance": 0,
  "abstention_quality": 0,
  "covered_facts": [],
  "missing_facts": [],
  "unsupported_claims": [],
  "comment": ""
}

Числовые оценки: от 0 до 1, кроме score: от 0 до 100.
Для unanswerable оцени качество отказа и отсутствие выдуманных фактов.
"""
)


class JudgeResult(TypedDict):
    label: str
    score: float
    factual_correctness: float
    faithfulness: float
    answer_relevance: float
    abstention_quality: float
    covered_facts: list[str]
    missing_facts: list[str]
    unsupported_claims: list[str]
    comment: str


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _parse_json_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM judge returned non-JSON response")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM judge JSON must be an object")
    return value


def _judge_result(raw: dict[str, Any]) -> JudgeResult:
    label = str(raw.get("label", "FAIL")).upper()
    if label not in {"PASS", "PARTIAL", "FAIL"}:
        label = "FAIL"

    def as_strings(value: Any) -> list[str]:
        return [str(item) for item in value] if isinstance(value, list) else []

    return {
        "label": label,
        "score": _clamp(raw.get("score"), 0, 100),
        "factual_correctness": _clamp(raw.get("factual_correctness"), 0, 1),
        "faithfulness": _clamp(raw.get("faithfulness"), 0, 1),
        "answer_relevance": _clamp(raw.get("answer_relevance"), 0, 1),
        "abstention_quality": _clamp(raw.get("abstention_quality"), 0, 1),
        "covered_facts": as_strings(raw.get("covered_facts")),
        "missing_facts": as_strings(raw.get("missing_facts")),
        "unsupported_claims": as_strings(raw.get("unsupported_claims")),
        "comment": str(raw.get("comment", "")),
    }


class LLMAsJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def evaluate(self, case: dict[str, Any], answer: Answer) -> JudgeResult:
        context = "\n\n".join(
            f"[{index}] {chunk.chunk.text}"
            for index, chunk in enumerate(answer.chunks, 1)
        )
        prompt = JUDGE_USER_TEMPLATE.substitute(
            question=case["question"],
            reference=case["expected_answer"],
            expected_facts=json.dumps(case["expected_facts"], ensure_ascii=False),
            forbidden_claims=case.get("forbidden_claims", ""),
            answer=answer.text,
            guard_state=answer.guard_state.name,
            answerability=case["answerability"],
            context=context,
        )
        response: LLMResponse = await self.llm.generate(
            [
                Message(role="system", content=JUDGE_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            temperature=0,
            max_tokens=1200,
        )
        return _judge_result(_parse_json_response(response.text))


def _answer_sources(answer: Answer) -> list[dict[str, Any]]:
    return [
        {
            "collection": source.collection,
            "source_type": source.source_type,
            "title": source.title,
            "url": source.url,
            "source_id": source.source_id,
            "chunk_id": source.chunk_id,
        }
        for source in answer.sources
    ]


def _deterministic_metrics(
    case: dict[str, Any],
    answer: Answer,
    source_mapping: SourceMapping,
) -> dict[str, Any]:
    retrieval = evaluate_case(case, answer.chunks, source_mapping)
    expected = case["answerability"] == "answerable"
    guard_ok = answer.guard_state is not AnswerGuardState.REFUSE
    gold_ids = set()
    if expected:
        gold_ids = source_mapping.chunks_for(
            case.get("expected_source_ids", []),
            retrieval.get("target_collection"),
        )
    actual_chunk_ids = {chunk.chunk.id for chunk in answer.chunks}
    cited_chunk_ids = {
        source["chunk_id"]
        for source in _answer_sources(answer)
        if source["chunk_id"]
    }
    return {
        "guard_correct": guard_ok == expected,
        "guard_state": answer.guard_state.name,
        "retrieval": retrieval,
        "source_hit": bool(actual_chunk_ids & gold_ids) if gold_ids else None,
        "citation_hit": bool(cited_chunk_ids & gold_ids) if gold_ids else None,
        "answer_nonempty": bool(answer.text.strip()),
    }


def _load_cases(dataset_path: Path, max_cases: int | None) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"{dataset_path}:{line_number}: expected JSON object")
        cases.append(case)
    return cases[:max_cases] if max_cases is not None else cases


async def evaluate_answers(
    dataset_path: Path,
    *,
    source_mapping: SourceMapping | None = None,
    judge: LLMAsJudge | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    mapping = source_mapping or SourceMapping()
    rows: list[dict[str, Any]] = []

    for case in tqdm(_load_cases(dataset_path, max_cases), desc="Answer evaluation", unit="case"):
        answer = await rag.answer(case["question"])
        row: dict[str, Any] = {
            "test_id": case["test_id"],
            "category": case["category"],
            "answerability": case["answerability"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_facts": case["expected_facts"],
            "answer": answer.text,
            "guard_state": answer.guard_state.name,
            "sources": _answer_sources(answer),
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk.id,
                    "collection": chunk.chunk.collection,
                    "score": chunk.score,
                    "stage": chunk.stage,
                }
                for chunk in answer.chunks
            ],
            "deterministic": _deterministic_metrics(case, answer, mapping),
        }
        if judge is not None:
            try:
                row["judge"] = await judge.evaluate(case, answer)
            except Exception as error:
                row["judge_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    judged = [row["judge"] for row in rows if "judge" in row]
    deterministic = [row["deterministic"] for row in rows]
    labels = Counter(result["label"] for result in judged)
    score_keys = [
        "score",
        "factual_correctness",
        "faithfulness",
        "answer_relevance",
        "abstention_quality",
    ]

    def optional_mean(key: str) -> float | None:
        values = [row[key] for row in deterministic if row[key] is not None]
        return mean(values) if values else None

    return {
        "dataset": str(dataset_path),
        "counts": {
            "total": len(rows),
            "answerable": sum(row["answerability"] == "answerable" for row in rows),
            "unanswerable": sum(row["answerability"] == "unanswerable" for row in rows),
            "judged": len(judged),
            "judge_errors": sum("judge_error" in row for row in rows),
        },
        "deterministic": {
            "guard_accuracy": mean(row["guard_correct"] for row in deterministic) if deterministic else None,
            "source_hit_rate": optional_mean("source_hit"),
            "citation_hit_rate": optional_mean("citation_hit"),
            "nonempty_rate": mean(row["answer_nonempty"] for row in deterministic) if deterministic else None,
        },
        "judge": {
            "labels": dict(labels),
            "pass_rate": labels["PASS"] / len(judged) if judged else None,
            "partial_rate": labels["PARTIAL"] / len(judged) if judged else None,
            "fail_rate": labels["FAIL"] / len(judged) if judged else None,
            "mean": {
                key: mean(result[key] for result in judged) if judged else None
                for key in score_keys
            },
        },
        "cases": rows,
    }


async def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated RAG answers.")
    parser.add_argument("--dataset", type=Path, default=Path("QA/datasets/gold/all.jsonl"))
    parser.add_argument("--source-mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/llm_eval.json"))
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory for detailed validation plots; defaults to <output-stem>_plots.",
    )
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    mapping = None
    if args.source_mapping:
        raw = json.loads(args.source_mapping.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{args.source_mapping}: expected a JSON object")
        from rag_data.source_mapping import load_mapping_dict

        mapping = load_mapping_dict(raw)

    settings = get_settings()
    judge_llm = create_llm(settings.llm) if not args.no_judge else None
    await rag.init(settings)
    try:
        report = await evaluate_answers(
            args.dataset,
            source_mapping=mapping,
            judge=LLMAsJudge(judge_llm) if judge_llm else None,
            max_cases=args.max_cases,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        plots_dir = args.plots_dir or args.output.parent / f"{args.output.stem}_plots"
        saved_plots = save_llm_plots(report, plots_dir)
        print(f"Saved {len(saved_plots)} validation plots to {plots_dir}")
    finally:
        await rag.close()
        if judge_llm is not None:
            await judge_llm.close()


if __name__ == "__main__":
    asyncio.run(run_cli())
