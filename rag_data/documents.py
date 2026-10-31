from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

COLLECTIONS = ("cards", "rules", "faq", "community")
AUTHORITATIVE_FAMILIES = {
    "cards": "cards",
    "rulebook": "rules",
    "faq": "faq",
    # Official VK articles have no separate collection in the current contract.
    "vk": "community",
}


@dataclass(frozen=True)
class IndexReadyDocument:
    collection: str
    chunk_id: str
    source_id: str
    text: str
    retrieval_text: str
    payload: dict[str, Any]
    point_id: str


def load_jsonl(path: Path, *, show_progress: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        lines = tqdm(file, desc=f"Reading {path.name}", unit="line") if show_progress else file
        for line_number, line in enumerate(lines, 1):
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


def collection_for(row: dict[str, Any], corpus: str) -> str:
    if corpus == "community":
        return "community"
    family = (row.get("metadata") or {}).get("source_family")
    try:
        return AUTHORITATIVE_FAMILIES[str(family)]
    except KeyError as error:
        raise ValueError(f"Unsupported authoritative source_family: {family!r}") from error


def _stable_point_id(collection: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"berserk-rag:{collection}:{chunk_id}"))


def _document(row: dict[str, Any], corpus: str) -> IndexReadyDocument:
    collection = collection_for(row, corpus)
    source_id = str(row.get("source_id") or "")
    if not source_id:
        raise ValueError("index-ready row has no source_id")
    text = str(row.get("text") or "")
    retrieval_text = str(row.get("embedding_text") or row.get("retrieval_text") or text)
    if not retrieval_text.strip():
        raise ValueError(f"index-ready row {source_id!r} has empty retrieval text")
    payload = dict(row)
    payload["collection"] = collection
    payload["chunk_id"] = source_id
    payload["text"] = text
    return IndexReadyDocument(
        collection=collection,
        chunk_id=source_id,
        source_id=source_id,
        text=text,
        retrieval_text=retrieval_text,
        payload=payload,
        point_id=str(row.get("point_id") or _stable_point_id(collection, source_id)),
    )


def load_index_ready(
    authoritative_path: Path,
    community_path: Path,
) -> dict[str, list[IndexReadyDocument]]:
    documents = {collection: [] for collection in COLLECTIONS}
    seen: set[tuple[str, str]] = set()
    sources = ((authoritative_path, "authoritative"), (community_path, "community"))
    for path, corpus in sources:
        for row in load_jsonl(path):
            document = _document(row, corpus)
            key = (document.collection, document.chunk_id)
            if key in seen:
                raise ValueError(f"duplicate chunk_id in collection {key[0]!r}: {key[1]!r}")
            seen.add(key)
            documents[document.collection].append(document)
    return documents
