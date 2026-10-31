"""Shared data contracts and legacy index-ready ingestion."""

from .documents import COLLECTIONS, IndexReadyDocument, load_index_ready
from .metrics import mrr, precision_at_k, recall_at_k

__all__ = [
    "COLLECTIONS",
    "IndexReadyDocument",
    "SourceMapping",
    "SourceMappingEntry",
    "build_legacy_source_mapping",
    "load_index_ready",
    "mrr",
    "precision_at_k",
    "recall_at_k",
]


def __getattr__(name: str):
    """Load source-mapping exports only when they are explicitly requested."""
    if name in {"SourceMapping", "SourceMappingEntry", "build_legacy_source_mapping"}:
        from . import source_mapping

        return getattr(source_mapping, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
