"""Retrieval metrics and visual reports for offline evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    """Fraction of gold chunks present in the first k results."""
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids) if gold_ids else 0.0


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    """Fraction of the first k results that are gold chunks."""
    top_k = retrieved_ids[:k]
    return len(set(top_k) & gold_ids) / len(top_k) if top_k else 0.0


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Reciprocal rank of the first retrieved gold chunk."""
    for rank, document_id in enumerate(retrieved_ids, 1):
        if document_id in gold_ids:
            return 1.0 / rank
    return 0.0


def _plot_metric_bars(
    ax: Any,
    values: dict[str, float],
    *,
    title: str,
    ylabel: str = "Score",
) -> None:
    labels = list(values)
    bars = ax.bar(labels, [values[label] for label in labels], color="#4472C4")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )


def _evaluated_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in report.get("cases", [])
        if case.get("status") == "evaluated" and isinstance(case.get("metrics"), dict)
    ]


def save_retrieval_plots(
    report: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Save detailed PNG charts for a retrieval evaluation report.

    Matplotlib is imported lazily so the basic metric functions stay usable
    without the plotting dependency. Every image is saved with a high DPI and
    a tight bounding box for readable zoomed-in inspection.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    aggregate = {
        name: value
        for name, value in report.get("aggregate", {}).items()
        if value is not None
    }

    if aggregate:
        figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
        _plot_metric_bars(axis, aggregate, title="Retrieval quality: aggregate metrics")
        path = output_dir / "aggregate_metrics.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    evaluated = _evaluated_cases(report)
    if not evaluated:
        return saved

    metric_names = list(aggregate)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in evaluated:
        categories[str(case.get("category", "unknown"))].append(case)

    category_values: dict[str, dict[str, float]] = {}
    for category, cases in sorted(categories.items()):
        category_values[category] = {
            metric: sum(case["metrics"].get(metric, 0.0) for case in cases) / len(cases)
            for metric in metric_names
        }

    if category_values and metric_names:
        figure, axis = plt.subplots(figsize=(max(12, len(category_values) * 2.2), 8))
        x = np.arange(len(category_values))
        width = 0.8 / len(metric_names)
        for index, metric in enumerate(metric_names):
            values = [category_values[category][metric] for category in category_values]
            axis.bar(x + index * width, values, width, label=metric)
        axis.set_title("Retrieval quality by QA category", fontsize=14, pad=12)
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.set_xticks(x + width * (len(metric_names) - 1) / 2, category_values)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncols=2)
        figure.tight_layout()
        path = output_dir / "metrics_by_category.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    ks = sorted(
        int(name.split("@", 1)[1])
        for name in metric_names
        if name.startswith("recall@")
    )
    if ks:
        figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        recalls = [aggregate.get(f"recall@{k}", 0.0) for k in ks]
        precisions = [aggregate.get(f"precision@{k}", 0.0) for k in ks]
        axis.plot(ks, recalls, marker="o", linewidth=2.5, label="Recall@K")
        axis.plot(ks, precisions, marker="o", linewidth=2.5, label="Precision@K")
        axis.set_title("Recall and Precision by K", fontsize=14, pad=12)
        axis.set_xlabel("K")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
        path = output_dir / "recall_precision_by_k.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    mrr_values = [case["metrics"].get("mrr") for case in evaluated]
    mrr_values = [float(value) for value in mrr_values if value is not None]
    if mrr_values:
        figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        axis.hist(
            mrr_values,
            bins=min(10, max(3, len(set(mrr_values)))),
            color="#70AD47",
            edgecolor="white",
        )
        axis.axvline(sum(mrr_values) / len(mrr_values), color="#C00000", linestyle="--", label="Mean MRR")
        axis.set_title("Distribution of per-case MRR", fontsize=14, pad=12)
        axis.set_xlabel("MRR")
        axis.set_ylabel("Number of cases")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        path = output_dir / "mrr_distribution.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    scores = [case.get("top_score") for case in report.get("cases", [])]
    scores = [float(score) for score in scores if score is not None]
    if scores:
        settings = report.get("settings", {})
        figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        axis.hist(
            scores,
            bins=min(20, max(5, len(set(scores)))),
            color="#5B9BD5",
            edgecolor="white",
        )
        for threshold, label, color in (
            (settings.get("refuse_threshold"), "Refuse threshold", "#C00000"),
            (settings.get("warn_threshold"), "Warn threshold", "#ED7D31"),
        ):
            if threshold is not None:
                axis.axvline(float(threshold), color=color, linestyle="--", label=label)
        axis.set_title("Top retrieval score and Answer Guard thresholds", fontsize=14, pad=12)
        axis.set_xlabel("Top retrieved score")
        axis.set_ylabel("Number of cases")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        path = output_dir / "top_score_guard_thresholds.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    return saved


def save_llm_plots(
    report: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Save detailed PNG charts for an LLM answer evaluation report."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    deterministic = {
        name: value
        for name, value in report.get("deterministic", {}).items()
        if isinstance(value, (int, float)) and value is not None
    }
    if deterministic:
        figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
        _plot_metric_bars(axis, deterministic, title="LLM validation: deterministic metrics")
        path = output_dir / "deterministic_metrics.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    judge_mean = {
        name: value
        for name, value in report.get("judge", {}).get("mean", {}).items()
        if isinstance(value, (int, float)) and value is not None
    }
    if judge_mean:
        figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
        _plot_metric_bars(axis, judge_mean, title="LLM-as-a-judge mean scores")
        path = output_dir / "judge_mean_scores.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    labels = {
        str(name): int(value)
        for name, value in report.get("judge", {}).get("labels", {}).items()
    }
    if labels:
        figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
        names = list(labels)
        bars = axis.bar(names, [labels[name] for name in names], color=[
            {"PASS": "#70AD47", "PARTIAL": "#ED7D31", "FAIL": "#C00000"}.get(
                name, "#5B9BD5"
            )
            for name in names
        ])
        axis.set_title("LLM-as-a-judge labels", fontsize=14, pad=12)
        axis.set_ylabel("Number of cases")
        axis.grid(axis="y", alpha=0.25)
        for bar in bars:
            axis.annotate(
                str(int(bar.get_height())),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 4),
                textcoords="offset points",
            )
        path = output_dir / "judge_labels.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    guard_states: dict[str, int] = defaultdict(int)
    judge_scores: list[float] = []
    category_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        guard_state = case.get("guard_state")
        if guard_state:
            guard_states[str(guard_state)] += 1
        judge = case.get("judge")
        if isinstance(judge, dict) and isinstance(judge.get("score"), (int, float)):
            judge_scores.append(float(judge["score"]))
        category_cases[str(case.get("category", "unknown"))].append(case)

    if guard_states:
        figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
        names = list(guard_states)
        bars = axis.bar(names, [guard_states[name] for name in names], color="#8064A2")
        axis.set_title("Answer Guard states", fontsize=14, pad=12)
        axis.set_ylabel("Number of cases")
        axis.grid(axis="y", alpha=0.25)
        for bar in bars:
            axis.annotate(
                str(int(bar.get_height())),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 4),
                textcoords="offset points",
            )
        path = output_dir / "answer_guard_states.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    if judge_scores:
        figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        axis.hist(
            judge_scores,
            bins=min(10, max(3, len(set(judge_scores)))),
            color="#5B9BD5",
            edgecolor="white",
        )
        axis.axvline(
            sum(judge_scores) / len(judge_scores),
            color="#C00000",
            linestyle="--",
            label="Mean score",
        )
        axis.set_title("Distribution of LLM-as-a-judge scores", fontsize=14, pad=12)
        axis.set_xlabel("Judge score")
        axis.set_ylabel("Number of cases")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        path = output_dir / "judge_score_distribution.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    category_values: dict[str, dict[str, float]] = {}
    for category, category_rows in sorted(category_cases.items()):
        values: dict[str, list[float]] = defaultdict(list)
        for case in category_rows:
            for name, value in case.get("deterministic", {}).items():
                if isinstance(value, (int, float)) and value is not None:
                    values[name].append(float(value))
            for name, value in case.get("judge", {}).items():
                if name != "score" and isinstance(value, (int, float)) and value is not None:
                    values[f"judge_{name}"].append(float(value))
        category_values[category] = {
            name: sum(items) / len(items) for name, items in values.items() if items
        }

    category_metrics = sorted({name for values in category_values.values() for name in values})
    if category_values and category_metrics:
        figure, axis = plt.subplots(figsize=(max(13, len(category_values) * 2.3), 8))
        x = np.arange(len(category_values))
        width = 0.8 / len(category_metrics)
        for index, metric in enumerate(category_metrics):
            values = [category_values[category].get(metric, 0.0) for category in category_values]
            axis.bar(x + index * width, values, width, label=metric)
        axis.set_title("LLM validation metrics by QA category", fontsize=14, pad=12)
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.set_xticks(x + width * (len(category_metrics) - 1) / 2, category_values)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncols=2, fontsize=8)
        figure.tight_layout()
        path = output_dir / "llm_metrics_by_category.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    return saved
