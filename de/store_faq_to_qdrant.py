# store_faq_to_qdrant.py
from pathlib import Path
from datetime import datetime, timezone
import uuid

from google_sheets_sources import SOURCES, collect_source
from google_sheets_normalizer import (
    load_jsonl, clean_text, normalize_number,
    approval_status, stable_id, release_name_from_faq_sheet,
)
from schema import Chunk, ChunkMetadata

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier, PointStruct,
)
from config import DENSE_MODEL, DENSE_SIZE, SPARSE_MODEL, COLLECTION_NAME

SHEET_NAME = "4. Ложные боги"


def normalize_faq_single_sheet(sheet_name: str, rows: list[dict]) -> list[dict]:
    records = []
    current_number, current_title = None, None
    release = release_name_from_faq_sheet(sheet_name)
    for row in rows[1:]:
        values = (row.get("values") or []) + [""] * 5
        number = normalize_number(values[0])
        title = clean_text(values[1])
        question = clean_text(values[2])
        answer = clean_text(values[3])
        if title and title.startswith("Это FAQ "):
            continue
        if number is not None:
            current_number = number
        if title:
            current_title = title
        if not question and not answer:
            continue
        status, rank = approval_status(values[4])
        parts = [f"FAQ: {release}"]
        if current_title: parts.append(f"Карта: {current_title}")
        if question: parts.append(f"Вопрос: {question}")
        if answer: parts.append(f"Ответ: {answer}")
        records.append({
            "faq_id": stable_id("faq", sheet_name, row["row_number"], question, answer),
            "card_number": current_number,
            "card_name": current_title,
            "question": question,
            "answer": answer,
            "approval_status": status,
            "text": "\n".join(parts),
        })
    return records


def to_chunk(record: dict) -> Chunk:
    priority = "official" if record["approval_status"] == "approved" else "community"
    return Chunk(
        text=record["text"],
        metadata=ChunkMetadata(
            chunk_id=record["faq_id"],
            doc_id="faq_4_lozhnye_bogi",
            source_type="faq",
            priority=priority,
            card_name=record["card_name"],
            card_number=record["card_number"],
            set_name=SHEET_NAME,
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


# --- 1. load + normalize + chunk ---
faq_source = next(s for s in SOURCES if s.key == "faq")
metadata = collect_source(faq_source, force=True)
rows = load_jsonl(Path(metadata["raw"]["rows_jsonl"]["path"]))
records = normalize_faq_single_sheet(SHEET_NAME, rows)
chunks = [to_chunk(r) for r in records]
print(f"Чанков получено: {len(chunks)}")


# --- 2. embedding (dense + sparse) ---
dense_embedder = TextEmbedding(DENSE_MODEL)
sparse_embedder = SparseTextEmbedding(SPARSE_MODEL, language="russian")

texts = [c.text for c in chunks]

dense_vectors = [
    v.tolist() for v in dense_embedder.embed([f"passage: {t}" for t in texts])
]
sparse_vectors = [
    models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    for s in sparse_embedder.embed(texts)
]
print(f"Dense: {len(dense_vectors)} x {len(dense_vectors[0])}, "
      f"sparse: {len(sparse_vectors)}")


# --- 3. upsert в Qdrant ---
client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
    },
)

points = [
    PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.metadata.chunk_id)),
        vector={"dense": dv, "sparse": sv},
        payload={**c.metadata.model_dump(), "text": c.text},
    )
    for c, dv, sv in zip(chunks, dense_vectors, sparse_vectors)
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Записано точек в Qdrant: {len(points)}")