"""
python -m rag_data.ingest --authoritative data/index_ready/authoritative_index_docs.jsonl --community data/index_ready/community_index_docs.jsonl --qdrant-url http://localhost:6333
"""


from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .documents import COLLECTIONS, IndexReadyDocument, load_index_ready

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _embedder(model_name: str, cache_dir: Path):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir / model_name))


def _sparse_embedder(model_name: str, cache_dir: Path):
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=model_name, cache_dir=str(cache_dir / model_name))


def _payload(document: IndexReadyDocument) -> dict[str, Any]:
    payload = dict(document.payload)
    payload.update({"chunk_id": document.chunk_id, "collection": document.collection})
    return payload


def _create_collection(client: Any, collection: str, first_document: IndexReadyDocument, dense: Any, models: Any) -> None:
    if client.collection_exists(collection):
        return
    first_vector = next(dense.embed(["passage: " + first_document.retrieval_text]))
    client.create_collection(
        collection_name=collection,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(
            size=len(first_vector), distance=models.Distance.COSINE,
        )},
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )},
    )


def _build_points(batch: list[IndexReadyDocument], dense: Any, sparse: Any, models: Any) -> list[Any]:
    dense_vectors = list(dense.embed(["passage: " + document.retrieval_text for document in batch]))
    sparse_vectors = list(sparse.embed([document.retrieval_text for document in batch]))
    return [
        models.PointStruct(
            id=document.point_id,
            vector={
                DENSE_VECTOR_NAME: dense_vector.tolist(),
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
            },
            payload=_payload(document),
        )
        for document, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors)
    ]


def _ingest_collection(
    client: Any,
    collection: str,
    documents: list[IndexReadyDocument],
    dense: Any,
    sparse: Any,
    models: Any,
    batch_size: int,
    recreate: bool,
) -> int:
    if not documents:
        return 0
    if client.collection_exists(collection) and recreate:
        client.delete_collection(collection)
    _create_collection(client, collection, documents[0], dense, models)

    batches = range(0, len(documents), batch_size)
    for start in tqdm(
        batches,
        total=math.ceil(len(documents) / batch_size),
        desc=f"Uploading {collection}",
        unit="batch",
    ):
        batch = documents[start:start + batch_size]
        client.upsert(
            collection_name=collection,
            points=_build_points(batch, dense, sparse, models),
            wait=True,
        )
    return int(client.count(collection_name=collection, exact=True).count)


def ingest_index_ready(
    documents: dict[str, list[IndexReadyDocument]],
    *,
    qdrant_url: str,
    dense_model: str = "intfloat/multilingual-e5-large",
    sparse_model: str = "Qdrant/bm25",
    model_cache_dir: Path = Path("models"),
    batch_size: int = 64,
    recreate: bool = True,
) -> dict[str, int]:
    """Re-embed legacy documents and upload them to the four RAG collections."""
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=qdrant_url)
    dense = _embedder(dense_model, model_cache_dir)
    sparse = _sparse_embedder(sparse_model, model_cache_dir)
    counts: dict[str, int] = {}

    for collection in tqdm(COLLECTIONS, desc="Collections", unit="collection"):
        counts[collection] = _ingest_collection(
            client,
            collection,
            documents[collection],
            dense,
            sparse,
            models,
            batch_size,
            recreate,
        )
        print(f"{collection}: {counts[collection]} points")
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load old index-ready JSONL into cards/rules/faq/community.")
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--sparse-model", default="Qdrant/bm25")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_index_ready(args.authoritative, args.community)
    print("Loaded documents:")
    for collection in COLLECTIONS:
        print(f"  {collection}: {len(documents[collection])}")
    counts = ingest_index_ready(
        documents,
        qdrant_url=args.qdrant_url,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        model_cache_dir=args.model_cache_dir,
        batch_size=args.batch_size,
        recreate=not args.no_recreate,
    )
    print(counts)


if __name__ == "__main__":
    main()
