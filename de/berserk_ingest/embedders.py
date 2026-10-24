"""Эмбеддеры за единым интерфейсом: смена модели = смена одного аргумента.

Embedder = dense-модель + (опционально) sparse BM25. Префиксы e5 («passage: » / «query: »)
живут внутри dense-бэкенда, чтобы документы и запросы нельзя было посчитать по-разному.
Тяжёлые библиотеки (fastembed, sentence-transformers) импортируются лениво.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Protocol

SparseVec = tuple[list[int], list[float]]   # (indices, values), без привязки к qdrant_client


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


class DenseBackend(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SparseBackend(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> list[SparseVec]: ...
    def embed_query(self, text: str) -> SparseVec: ...


class FastEmbedDense:
    def __init__(self, model_name: str, passage_prefix: str = "passage: ", query_prefix: str = "query: "):
        from fastembed import TextEmbedding

        self.name = model_name
        self.passage_prefix, self.query_prefix = passage_prefix, query_prefix
        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed([self.passage_prefix + t for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self.query_prefix + text]))).tolist()


class SentenceTransformerDense:
    """Для моделей, которых нет в fastembed (например multilingual-e5-base)."""

    def __init__(self, model_name: str, passage_prefix: str = "passage: ", query_prefix: str = "query: "):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.passage_prefix, self.query_prefix = passage_prefix, query_prefix
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode([self.passage_prefix + t for t in texts], normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


class FastEmbedBM25:
    def __init__(self, model_name: str = "Qdrant/bm25", language: str = "russian"):
        from fastembed import SparseTextEmbedding

        self.name = model_name
        try:   # язык (стеммер и стоп-слова) поддерживается не во всех версиях fastembed
            self._model = SparseTextEmbedding(model_name, language=language)
            self.language = language
        except TypeError:
            print("[embedders] ВНИМАНИЕ: эта версия fastembed не принимает language=, BM25 без русского стеммера")
            self._model = SparseTextEmbedding(model_name)
            self.language = None

    @staticmethod
    def _conv(s) -> SparseVec:
        return s.indices.tolist(), s.values.tolist()

    def embed_passages(self, texts: list[str]) -> list[SparseVec]:
        return [self._conv(s) for s in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        return self._conv(next(iter(self._model.query_embed(text))))


@dataclass
class Embedder:
    dense: DenseBackend
    sparse: SparseBackend | None = None

    @property
    def name(self) -> str:
        return self.dense.name + (f"+{self.sparse.name}" if self.sparse else "")

    @property
    def dense_size(self) -> int:
        return self.dense.dim

    @property
    def has_sparse(self) -> bool:
        return self.sparse is not None

    def embed_passages(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec] | None]:
        return self.dense.embed_passages(texts), (self.sparse.embed_passages(texts) if self.sparse else None)

    def embed_query(self, text: str) -> tuple[list[float], SparseVec | None]:
        return self.dense.embed_query(text), (self.sparse.embed_query(text) if self.sparse else None)

    def info(self) -> dict:
        return {
            "name": self.name, "dense_model": self.dense.name, "dense_dim": self.dense.dim,
            "sparse_model": self.sparse.name if self.sparse else None,
            "sparse_language": getattr(self.sparse, "language", None) if self.sparse else None,
            "libs": {p: _version(p) for p in ("fastembed", "sentence-transformers", "qdrant-client")},
        }


# Новый эмбеддер = новая строка здесь (или свой класс DenseBackend/SparseBackend).
PRESETS: dict[str, Callable[[], Embedder]] = {
    "e5-base": lambda: Embedder(SentenceTransformerDense("intfloat/multilingual-e5-base")),
    "e5-base-bm25": lambda: Embedder(SentenceTransformerDense("intfloat/multilingual-e5-base"), FastEmbedBM25()),
    "e5-large": lambda: Embedder(FastEmbedDense("intfloat/multilingual-e5-large")),
    "e5-large-bm25": lambda: Embedder(FastEmbedDense("intfloat/multilingual-e5-large"), FastEmbedBM25()),
}


def get_embedder(name: str) -> Embedder:
    try:
        factory = PRESETS[name]
    except KeyError:
        raise SystemExit(f"Неизвестный эмбеддер {name!r}. Доступно: {', '.join(PRESETS)}") from None
    return factory()
