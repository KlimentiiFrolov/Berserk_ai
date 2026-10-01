from collections.abc import Sequence
from typing import List
import asyncio

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import AsyncQdrantClient, models

from core.config import RAGSettings, get_rag_settings
from rag.schemas import RetrievedChunk, Chunk

from logging import Logger

logger = Logger(__name__)


class Retriever:
    def __init__(
        self,
        settings: RAGSettings | None = None
    ) -> None:
        self.settings = settings or get_rag_settings()
        logger.debug(f"Модели будут сохранены в {self.settings.models_cache_dir}")
        self.client = AsyncQdrantClient(url=self.settings.qdrant_url)
        self.dense_embedder = TextEmbedding(
            model_name=self.settings.dense_model_name,
            cache_dir=str(self.settings.models_cache_dir),
        )
        self.sparse_embedder = SparseTextEmbedding(
            model_name=self.settings.sparse_model_name,
            cache_dir=str(self.settings.models_cache_dir),
        )
        self.reranker = TextCrossEncoder(
            model_name=self.settings.rerank_model_name,
            cache_dir=str(self.settings.models_cache_dir),
        )
        self.router = None
        self.collection_names = ["cards", "rules", "faq", "community"]

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "Retriever":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    
    async def _extract_collections(self, query: str) -> List[str]:
        # извлекает коллекции из запроса, тут в будущем будет работать роутер
        return self.collection_names
    
    async def retrieve(self, query: str, 
                       top_k: int = 5, 
                       collections: list | None = None,
                       dense_chunks_limit: int = 20,
                       sparse_chunks_limit: int = 20,
                       rrf_limit: int = 20,
                       use_reranker: bool = True) -> List[RetrievedChunk]:
        # точка входа
        collections = collections if collections else await self._extract_collections(query)
        chunks = await self._search(query, 
                                   collections=collections,
                                   dense_chunks_limit=dense_chunks_limit,
                                   sparse_chunks_limit=sparse_chunks_limit,
                                   rrf_limit=rrf_limit)
        #print("retrieve:chunks:", chunks)
        if use_reranker:
            chunks = await self._rerank(query, chunks)
            
        return chunks[:top_k]
    
    async def _embed_dense(self, texts) -> list[list]:
        dense_vectors = await asyncio.to_thread(
            lambda: list(self.dense_embedder.embed(["query: " + text 
                                                    for text in texts])))
        return [dense_vector.tolist()
                for dense_vector in dense_vectors]
            
    async def _embed_sparse(self, texts) -> List[models.SparseVector]:
        sparse_vectors = await asyncio.to_thread(
            lambda: list(self.sparse_embedder.embed(texts))
        )
        return [models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist())
                for sparse_vector in sparse_vectors]
        
    async def _search(self, query: str, 
                     collections: List[str], 
                     dense_chunks_limit: int = 20,
                     sparse_chunks_limit: int = 20,
                     rrf_limit: int = 20) -> List[RetrievedChunk]:
        """
        Поиск документов в векторной базе данных
        """
        dense_vec, sparse_vec = await asyncio.gather(
                    self._embed_dense([query]),
                    self._embed_sparse([query]))
        
        dense_vec, sparse_vec = dense_vec[0], sparse_vec[0]
        
        tasks = [self.client.query_points(
                collection_name,
                prefetch=[
                        models.Prefetch(query=dense_vec, using="dense", 
                                        limit=dense_chunks_limit),
                        models.Prefetch(query=sparse_vec, using="sparse", 
                                        limit=sparse_chunks_limit),
                    ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=rrf_limit,
            ) for collection_name in collections]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_chunks = []
        for collection_name, result in zip(collections, results):
            if isinstance(result, BaseException):
                logger.warning(f"Коллекция {collection_name} недоступна: {result}")
                continue
            
            chunks = [RetrievedChunk(
                        chunk=Chunk(
                                id=point.payload.get('id', ''),
                                text=point.payload.get('text', ''),
                                source_type=point.payload.get('source_type', ''),
                                collection=collection_name,
                                metadata=point.payload
                            ),
                            score=point.score,
                            stage="hybrid_rrf"
                        ) 
                    for point in result.points if point.payload is not None]
            
            total_chunks.extend(chunks)
            
        return total_chunks
    
    
    async def _rerank(self, query: str, chunks: List[RetrievedChunk]):
        texts = [chunk.chunk.text for chunk in chunks]
        scores = list(await asyncio.to_thread(
                lambda: self.reranker.rerank(query, texts)
            ))
        #print('rerank:score', scores)
        for chunk, score in zip(chunks, scores):
            chunk.stage = "reranked"
            chunk.score = score
            
        return sorted(chunks, key=lambda x: x.score, reverse=True)
