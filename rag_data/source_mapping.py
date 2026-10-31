"""Mapping between QA source IDs and legacy index-ready chunks.

The QA dataset uses logical source IDs (``SRC-001``, ``CARD-001``), while
legacy index-ready documents use IDs such as ``knowledge_card_*``.  This
module keeps that translation outside DE and QA code and preserves the
collection dimension, because the same logical card source can have both a
card record and FAQ records.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .documents import COLLECTIONS, IndexReadyDocument

QA_CATEGORY_COLLECTIONS = {
    "Rules_Basic": "rules",
    "Mechanics_Glossary": "rules",
    "Cards_Facts": "cards",
    "Card_FAQ_Interactions": "faq",
    "Tournament_Formats": "rules",
}

# These names are the stable names used by the QA source registry.  Matching
# is deliberately exact after whitespace normalization; fuzzy matching would
# silently turn a wrong source into a gold chunk.
CARD_SOURCE_NAMES = {
    "CARD-001": "Дхайа",
    "CARD-002": "Лёгкая кавалерия",
    "CARD-003": "Древний огр",
    "CARD-004": "Викарий",
    "CARD-005": "Жрец штормов",
    "CARD-006": "Землекоп",
    "CARD-007": "Древляна",
    "CARD-008": "Азенаэль",
    "CARD-009": "Мастер разума",
    "CARD-010": "Кицунэ",
}


@dataclass(frozen=True)
class SourceMappingEntry:
    """Resolved chunks for one logical QA source."""

    source_id: str
    collection_chunks: dict[str, frozenset[str]]
    match_type: str
    status: str
    notes: tuple[str, ...] = ()

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset().union(*self.collection_chunks.values())


@dataclass(frozen=True)
class SourceMapping:
    """Collection-aware source mapping used by retrieval evaluation."""

    entries: dict[str, SourceMappingEntry] = field(default_factory=dict)

    def chunks_for(self, source_ids: Iterable[str], collection: str | None = None) -> set[str]:
        """Return legacy chunk IDs allowed for the given QA source IDs."""
        chunks: set[str] = set()
        for source_id in source_ids:
            entry = self.entries.get(source_id)
            if entry is None:
                continue
            if collection is None:
                chunks.update(entry.chunk_ids)
            else:
                chunks.update(entry.collection_chunks.get(collection, ()))
                chunks.update(entry.collection_chunks.get("*", ()))
        return chunks

    def as_flat_dict(self) -> dict[str, list[str]]:
        """Export the evaluator's legacy ``source_id -> chunk IDs`` format."""
        return {
            source_id: sorted(entry.chunk_ids)
            for source_id, entry in self.entries.items()
        }

    def as_dict(self) -> dict[str, Any]:
        """Export mapping plus diagnostics for review and reproducibility."""
        return {
            source_id: {
                "status": entry.status,
                "match_type": entry.match_type,
                "notes": list(entry.notes),
                "collections": {
                    collection: sorted(chunk_ids)
                    for collection, chunk_ids in entry.collection_chunks.items()
                },
            }
            for source_id, entry in self.entries.items()
        }


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _metadata(document: IndexReadyDocument) -> Mapping[str, Any]:
    metadata = document.payload.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _card_name_matches(document: IndexReadyDocument, expected_name: str) -> bool:
    metadata = _metadata(document)
    entities = metadata.get("entities")
    card_names = entities.get("cards", ()) if isinstance(entities, Mapping) else ()
    candidates = [document.payload.get("title"), *card_names]
    return any(_normalize_name(candidate) == _normalize_name(expected_name) for candidate in candidates)


def _entry(
    source_id: str,
    collection_chunks: dict[str, set[str]],
    match_type: str,
    notes: Iterable[str] = (),
) -> SourceMappingEntry:
    frozen = {
        collection: frozenset(chunks)
        for collection, chunks in collection_chunks.items()
        if chunks
    }
    status = "mapped" if frozen else "missing"
    if any(len(chunks) > 1 for chunks in frozen.values()):
        status = "ambiguous"
    return SourceMappingEntry(
        source_id=source_id,
        collection_chunks=frozen,
        match_type=match_type,
        status=status,
        notes=tuple(notes),
    )


def build_legacy_source_mapping(
    documents: Mapping[str, Iterable[IndexReadyDocument]],
) -> SourceMapping:
    """Build the QA-to-legacy mapping from loaded index-ready documents.

    The function only uses explicit metadata or exact card names.  It does
    not use embeddings or lexical similarity, so its result is safe to use as
    a gold set for Recall@K/Precision@K.
    """
    by_collection = {
        collection: list(documents.get(collection, ()))
        for collection in COLLECTIONS
    }
    entries: dict[str, SourceMappingEntry] = {}

    rule_documents = by_collection["rules"]
    entries["SRC-001"] = _entry(
        "SRC-001",
        {
            "rules": {
                document.chunk_id
                for document in rule_documents
                if _metadata(document).get("source_family") == "rulebook"
                and _metadata(document).get("source_type") != "rulebook_glossary"
            }
        },
        "metadata_filter",
        ("Legacy rulebook records; exact official URL is not retained in the legacy rows.",),
    )
    entries["SRC-002"] = _entry(
        "SRC-002",
        {
            "rules": {
                document.chunk_id
                for document in rule_documents
                if _metadata(document).get("source_type") == "rulebook_glossary"
            }
        },
        "metadata_filter",
        ("Legacy glossary records matched by source_type=rulebook_glossary.",),
    )
    entries["SRC-003"] = _entry(
        "SRC-003", {}, "unavailable", ("No tournament chunks were found in the legacy authoritative snapshot.",)
    )
    entries["SRC-004"] = _entry(
        "SRC-004", {}, "not_gold_source", ("Used only as an out-of-scope QA reference.",)
    )

    for source_id, card_name in CARD_SOURCE_NAMES.items():
        card_chunks = {
            document.chunk_id
            for document in by_collection["cards"]
            if _card_name_matches(document, card_name)
        }
        faq_chunks = {
            document.chunk_id
            for document in by_collection["faq"]
            if _card_name_matches(document, card_name)
        }
        notes: list[str] = []
        if len(card_chunks) > 1:
            notes.append("Multiple card records matched; likely reprint or duplicate legacy records.")
        if not card_chunks:
            notes.append("No exact card record matched in the legacy cards collection.")
        if not faq_chunks:
            notes.append("No exact FAQ record matched in the legacy FAQ collection.")
        entries[source_id] = _entry(
            source_id,
            {"cards": card_chunks, "faq": faq_chunks},
            "exact_card_name",
            notes,
        )

    return SourceMapping(entries)


def load_mapping_dict(raw: Mapping[str, Any]) -> SourceMapping:
    """Load either the diagnostic format or the old flat JSON format."""
    entries: dict[str, SourceMappingEntry] = {}
    for source_id, value in raw.items():
        if isinstance(value, list):
            entries[source_id] = _entry(source_id, {"*": set(map(str, value))}, "external", ())
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid source mapping entry for {source_id!r}")
        collections = value.get("collections", {})
        if not isinstance(collections, Mapping):
            raise ValueError(f"Invalid collections for {source_id!r}")
        entries[source_id] = SourceMappingEntry(
            source_id=source_id,
            collection_chunks={
                str(collection): frozenset(map(str, chunk_ids))
                for collection, chunk_ids in collections.items()
                if isinstance(chunk_ids, list)
            },
            match_type=str(value.get("match_type", "external")),
            status=str(value.get("status", "mapped")),
            notes=tuple(map(str, value.get("notes", ()))),
        )
    return SourceMapping(entries)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build collection-aware QA source mapping from legacy index-ready JSONL."
    )
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from .documents import load_index_ready

    args = _parse_args()
    documents = load_index_ready(args.authoritative, args.community)
    mapping = build_legacy_source_mapping(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(mapping.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
