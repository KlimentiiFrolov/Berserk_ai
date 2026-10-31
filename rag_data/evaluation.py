"""Compatibility imports for the evaluation package.

New code should import from evaluate.eval_retrieval. This module remains only
so existing integrations do not break during the package reorganization.
"""

from evaluate.eval_retrieval import *
from evaluate.eval_retrieval import run_cli

__all__ = [
    "TARGET_COLLECTIONS",
    "evaluate_case",
    "evaluate_retriever",
    "guard_state_for_score",
    "load_jsonl",
    "load_source_mapping",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "run_cli",
]
