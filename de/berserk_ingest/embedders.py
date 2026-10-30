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


def _auto_prefix(model_name: str, kind: str) -> str:
    """Префиксы нужны только семейству e5; для остальных моделей (bge, jina, ...) пустая строка."""
    if "e5" in model_name.lower():
        return {"passage": "passage: ", "query": "query: "}[kind]
    return ""


def _to_list(vec) -> list[float]:
    return vec.tolist() if hasattr(vec, "tolist") else [float(x) for x in vec]


def _to_sparse(s) -> SparseVec:
    return [int(i) for i in s.indices], [float(v) for v in s.values]


class FastEmbedDense:
    def __init__(self, model_name: str, passage_prefix: str | None = None, query_prefix: str | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self.passage_prefix = _auto_prefix(model_name, "passage") if passage_prefix is None else passage_prefix
        self.query_prefix = _auto_prefix(model_name, "query") if query_prefix is None else query_prefix
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


# ---------- адаптеры для «любой модели с интерфейсом FastEmbed» ----------
class FastEmbedLikeDense:
    """Оборачивает готовый объект с методом `.embed(texts) -> Iterable[array]` (fastembed.TextEmbedding и аналоги).

    Префиксы e5 подставляются автоматически по имени модели (`model_name`); задать вручную:
    passage_prefix="..." / query_prefix="..." (пустая строка = без префикса).
    """

    def __init__(self, model, *, name: str | None = None,
                 passage_prefix: str | None = None, query_prefix: str | None = None):
        self._model = model
        self.name = name or getattr(model, "model_name", None) or type(model).__name__
        self.passage_prefix = _auto_prefix(self.name, "passage") if passage_prefix is None else passage_prefix
        self.query_prefix = _auto_prefix(self.name, "query") if query_prefix is None else query_prefix
        self.dim = len(_to_list(next(iter(model.embed(["probe"])))))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [_to_list(v) for v in self._model.embed([self.passage_prefix + t for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return _to_list(next(iter(self._model.embed([self.query_prefix + text]))))


class FastEmbedLikeSparse:
    """Оборачивает объект с `.embed(texts)`, элементы которого имеют `.indices` и `.values`
    (fastembed.SparseTextEmbedding). Для запросов берётся `query_embed`, если он есть (у BM25 он нужен)."""

    def __init__(self, model, *, name: str | None = None):
        self._model = model
        self.name = name or getattr(model, "model_name", None) or type(model).__name__

    def embed_passages(self, texts: list[str]) -> list[SparseVec]:
        return [_to_sparse(s) for s in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        query_embed = getattr(self._model, "query_embed", None)
        out = query_embed(text) if query_embed else self._model.embed([text])
        return _to_sparse(next(iter(out)))


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


def wrap_fastembed(dense, sparse=None, *, passage_prefix: str | None = None,
                   query_prefix: str | None = None) -> Embedder:
    """Собрать Embedder из готовых моделей FastEmbed: wrap_fastembed(TextEmbedding(...), SparseTextEmbedding(...))."""
    return Embedder(
        FastEmbedLikeDense(dense, passage_prefix=passage_prefix, query_prefix=query_prefix),
        FastEmbedLikeSparse(sparse) if sparse is not None else None,
    )


def as_embedder(embedder, sparse_embedder=None):
    """Единая точка входа для Indexer: принимает готовый Embedder (embed_passages/embed_query)
    ИЛИ модель с интерфейсом FastEmbed (`.embed`), к которой можно добавить sparse_embedder."""
    if hasattr(embedder, "embed_passages"):
        if sparse_embedder is not None:
            raise ValueError("sparse_embedder нельзя передавать вместе с готовым Embedder: он уже содержит sparse-часть")
        return embedder
    if not hasattr(embedder, "embed"):
        raise TypeError(f"{type(embedder).__name__}: ожидается объект с методом .embed(texts) (интерфейс FastEmbed) "
                        "или Embedder из berserk_ingest.embedders")
    return wrap_fastembed(embedder, sparse_embedder)


def custom_embedder(dense_model: str, sparse_model: str | None = "Qdrant/bm25") -> Embedder:
    """Эмбеддер по именам моделей FastEmbed, без правки PRESETS (для CLI --dense-model/--sparse-model)."""
    return Embedder(FastEmbedDense(dense_model), FastEmbedBM25(sparse_model) if sparse_model else None)


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
