import asyncio
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import AsyncQdrantClient, models

from core.config import MODELS_DIR, Settings, get_settings
from core.logger import prepare_logger
from rag.schemas import Chunk, RetrievedChunk

logger = prepare_logger(__name__, get_settings().log)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        logger.debug("Модели будут сохранены в %s", MODELS_DIR)
        self.client = AsyncQdrantClient(
            url=self.settings.qdrant.url,
            api_key=self.settings.qdrant.api_key.get_secret_value() or None,
        )
        self.dense_embedder = TextEmbedding(
            model_name=self.settings.models.dense,
            cache_dir=str(MODELS_DIR / self.settings.models.dense),
        )
        self.sparse_embedder = SparseTextEmbedding(
            model_name=self.settings.models.sparse,
            cache_dir=str(MODELS_DIR / self.settings.models.sparse),
        )
        self.reranker = TextCrossEncoder(
            model_name=self.settings.models.rerank,
            cache_dir=str(MODELS_DIR / self.settings.models.rerank),
        )
        self.router = None
        self.collection_names = self.settings.qdrant.collections

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "Retriever":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    
    async def _extract_collections(self, query: str) -> list[str]:
        # извлекает коллекции из запроса, тут в будущем будет работать роутер
        return self.collection_names
    
    async def retrieve(self, query: str, 
                       top_k: int = 5, 
                       collections: list[str] | None = None,
                       dense_chunks_limit: int = 20,
                       sparse_chunks_limit: int = 20,
                       rrf_limit: int = 20,
                       rerank_limit: int = 20,
                       use_reranker: bool = True) -> list[RetrievedChunk]:
        # точка входа
        collections = collections if collections else await self._extract_collections(query)
        chunks = await self._search(query, 
                                   collections=collections,
                                   dense_chunks_limit=dense_chunks_limit,
                                   sparse_chunks_limit=sparse_chunks_limit,
                                   rrf_limit=rrf_limit)
        if use_reranker:
            rerank_count = min(rerank_limit, len(chunks))
            reranked = await self._rerank(query, chunks[:rerank_count])
            chunks = reranked + chunks[rerank_count:]
        else:
            chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)

        return chunks[:top_k]
    
    async def _embed_dense(self, texts) -> list[list]:
        dense_vectors = await asyncio.to_thread(
            lambda: list(self.dense_embedder.embed(["query: " + text 
                                                    for text in texts])))
        return [dense_vector.tolist()
                for dense_vector in dense_vectors]
            
    async def _embed_sparse(self, texts) -> list[models.SparseVector]:
        sparse_vectors = await asyncio.to_thread(
            lambda: list(self.sparse_embedder.embed(texts))
        )
        return [models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist())
                for sparse_vector in sparse_vectors]
        
    async def _search(self, query: str, 
                     collections: list[str],
                     dense_chunks_limit: int = 20,
                     sparse_chunks_limit: int = 20,
                     rrf_limit: int = 20) -> list[RetrievedChunk]:
        """
        Поиск документов в векторной базе данных
        """
        dense_vec, sparse_vec = await asyncio.gather(
                    self._embed_dense([query]),
                    self._embed_sparse([query]))
        
        dense_vec, sparse_vec = dense_vec[0], sparse_vec[0]
        
        tasks = [self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=dense_chunks_limit,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=sparse_chunks_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=rrf_limit,
        ) for collection_name in collections]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_chunks: list[RetrievedChunk] = []
        for collection_name, result in zip(collections, results):
            if isinstance(result, BaseException):
                logger.warning("Коллекция %s недоступна: %s", collection_name, result)
                continue

            for point in result.points:
                total_chunks.append(RetrievedChunk(
                    chunk=self._chunk_from_point(collection_name, point),
                    score=point.score,
                    stage="hybrid_rrf",
                    rrf_score=point.score,
                ))

        return total_chunks

    @staticmethod
    def _chunk_from_point(collection_name: str, point: Any) -> Chunk:
        payload = point.payload or {}
        return Chunk(
            id=str(payload.get("chunk_id") or payload.get("source_id") or point.id),
            text=str(payload.get("text") or ""),
            source_type=str(payload.get("source_type") or ""),
            collection=collection_name,
            metadata=payload,
        )
    
    
    async def _rerank(self, query: str, chunks: list[RetrievedChunk]):
        texts = [chunk.chunk.text for chunk in chunks]
        scores = list(await asyncio.to_thread(
                lambda: self.reranker.rerank(query, texts)
            ))
        #print('rerank:score', scores)
        for chunk, score in zip(chunks, scores):
            chunk.stage = "reranked"
            chunk.rerank_score = float(score)
            chunk.score = float(score)
            
        return sorted(chunks, key=lambda x: x.score, reverse=True)
