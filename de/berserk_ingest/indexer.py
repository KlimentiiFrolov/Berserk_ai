"""Индексатор Qdrant: коллекция «dense (+ sparse BM25)», идемпотентная заливка, поиск.

Использование (embedder — любая модель с интерфейсом FastEmbed `.embed(texts)`):
    from fastembed import TextEmbedding, SparseTextEmbedding
    loader = Loader()                                   # QdrantClient(url=QDRANT_URL)
    loader.load_data(chunks,
                     embedder=TextEmbedding("intfloat/multilingual-e5-large"),
                     sparse_embedder=SparseTextEmbedding("Qdrant/bm25", language="russian"))
Без sparse_embedder коллекция будет только dense. Готовый Embedder из embedders.py тоже подходит:
    loader.load_data(chunks, embedder=get_embedder("e5-large-bm25"))
"""
from __future__ import annotations

import re
from typing import Sequence

from .embedders import Embedder, as_embedder
from .schema import Chunk, point_id

DENSE = "dense"      # имена именованных векторов: на них опирается retrieval (Prefetch using=...)
SPARSE = "sparse"
KEYWORD_INDEXES = ("source_type", "priority", "doc_id", "approval_status", "card_name")


def default_collection_name(embedder: Embedder) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", embedder.name.lower()).strip("_")
    return f"berserk_{slug}"


def _models():
    from qdrant_client import models   # ленивый импорт: пакет работает и без qdrant-client (команда `chunks`)
    return models


class Indexer:
    def __init__(self, client=None, *, url: str | None = None):
        if client is None:
            from qdrant_client import QdrantClient
            from .config import QDRANT_URL
            client = QdrantClient(url=url or QDRANT_URL)
        self.client = client

    # ---------- коллекция ----------
    def ensure_collection(self, name: str, embedder: Embedder, *, recreate: bool) -> None:
        m = _models()
        exists = self.client.collection_exists(name)
        if exists and recreate:
            self.client.delete_collection(name)
            exists = False
        if exists:
            self._check_compatible(name, embedder)
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={DENSE: m.VectorParams(size=embedder.dense_size, distance=m.Distance.COSINE)},
            sparse_vectors_config=(
                {SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)} if embedder.has_sparse else None
            ),
        )
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(name, field_name=field, field_schema=m.PayloadSchemaType.KEYWORD)

    def _check_compatible(self, name: str, embedder: Embedder) -> None:
        params = self.client.get_collection(name).config.params
        vectors = params.vectors
        size = vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None
        has_sparse = bool(params.sparse_vectors) and SPARSE in params.sparse_vectors
        if size != embedder.dense_size or has_sparse != embedder.has_sparse:
            raise RuntimeError(
                f"Коллекция {name!r} несовместима с эмбеддером {embedder.name!r}: "
                f"в коллекции dense={size}, sparse={has_sparse}; нужно dense={embedder.dense_size}, "
                f"sparse={embedder.has_sparse}. Используйте другое имя или recreate=True."
            )

    def delete_docs(self, name: str, doc_ids: Sequence[str]) -> None:
        m = _models()
        self.client.delete(
            collection_name=name,
            points_selector=m.FilterSelector(filter=m.Filter(
                must=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(doc_ids)))]
            )),
        )

    def count(self, name: str) -> int:
        return int(self.client.count(collection_name=name, exact=True).count)

    # ---------- заливка ----------
    def load_data(
        self,
        chunks: list[Chunk],
        embedder,
        collection_name: str | None = None,
        *,
        sparse_embedder=None,
        recreate: bool = True,
        batch_size: int = 32,
    ) -> str:
        """Эмбеддит чанки и заливает в коллекцию; возвращает её имя.

        embedder — модель с интерфейсом FastEmbed (`.embed`) или готовый Embedder; sparse_embedder — опциональная
        sparse-модель (BM25). Имя коллекции по умолчанию строится из имён моделей, поэтому разные
        модели не перетирают друг друга.

        recreate=True  — коллекция создаётся заново (воспроизводимый прогон на тестовом наборе);
        recreate=False — «синхронизация»: старые точки тех же doc_id удаляются, затем заливаются новые,
                         поэтому правка/удаление строки в источнике не оставляет в базе устаревших чанков.
        """
        m = _models()
        embedder = as_embedder(embedder, sparse_embedder)
        name = collection_name or default_collection_name(embedder)
        self.ensure_collection(name, embedder, recreate=recreate)
        if not recreate and chunks:
            self.delete_docs(name, sorted({c.metadata.doc_id for c in chunks}))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            dense, sparse = embedder.embed_passages([c.text for c in batch])
            points = []
            for j, chunk in enumerate(batch):
                vector = {DENSE: dense[j]}
                if sparse is not None:
                    idx, val = sparse[j]
                    vector[SPARSE] = m.SparseVector(indices=idx, values=val)
                points.append(m.PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=chunk.payload()))
            self.client.upsert(collection_name=name, points=points)
        print(f"[indexer] {embedder.name} -> {name}: {len(chunks)} чанков")
        return name

    # ---------- поиск (для smoke-теста и ручной проверки) ----------
    def search(
        self,
        name: str,
        embedder,
        query: str,
        *,
        sparse_embedder=None,
        limit: int = 5,
        mode: str = "hybrid",
        source_type: str | None = None,
        priority: str | None = None,
    ) -> list[dict]:
        m = _models()
        embedder = as_embedder(embedder, sparse_embedder)
        conds = []
        if source_type:
            conds.append(m.FieldCondition(key="source_type", match=m.MatchValue(value=source_type)))
        if priority:
            conds.append(m.FieldCondition(key="priority", match=m.MatchValue(value=priority)))
        flt = m.Filter(must=conds) if conds else None

        dense_q, sparse_q = embedder.embed_query(query)
        if mode == "hybrid" and sparse_q is not None:
            sv = m.SparseVector(indices=sparse_q[0], values=sparse_q[1])
            res = self.client.query_points(
                collection_name=name,
                prefetch=[
                    m.Prefetch(query=dense_q, using=DENSE, limit=limit * 4, filter=flt),
                    m.Prefetch(query=sv, using=SPARSE, limit=limit * 4, filter=flt),
                ],
                query=m.FusionQuery(fusion=m.Fusion.RRF),
                limit=limit, with_payload=True,
            )
        elif mode == "sparse":
            if sparse_q is None:
                raise ValueError("у этого эмбеддера нет sparse-части")
            res = self.client.query_points(
                collection_name=name, query=m.SparseVector(indices=sparse_q[0], values=sparse_q[1]),
                using=SPARSE, limit=limit, with_payload=True, query_filter=flt)
        else:
            res = self.client.query_points(
                collection_name=name, query=dense_q, using=DENSE, limit=limit,
                with_payload=True, query_filter=flt)
        return [{"score": p.score, **(p.payload or {})} for p in res.points]


Loader = Indexer   # алиас под привычное имя: Loader(client).load_data(chunks, embedder=...)
