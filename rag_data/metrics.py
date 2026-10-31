"""Compatibility imports for metrics moved to evaluate.metrics."""

from evaluate.metrics import mrr, precision_at_k, recall_at_k

__all__ = ["mrr", "precision_at_k", "recall_at_k"]
