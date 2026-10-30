"""Тесты индексатора.

FakeClientTests проверяют НАШУ логику (имена векторов, батчи, идемпотентность, синхронизация по doc_id,
защита от смешивания моделей) на заглушке клиента и работают без qdrant-client.
LocalQdrantTests прогоняют то же на настоящем qdrant-client в локальном режиме (без сервера):
запускаются, только если библиотека установлена. Именно они проверяют соответствие реальному API.
"""
import hashlib
import math
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from berserk_ingest import indexer as indexer_mod
from berserk_ingest.indexer import DENSE, SPARSE, Indexer, default_collection_name
from berserk_ingest.schema import Chunk, ChunkMetadata, point_id


class FakeEmbedder:
    """Детерминированный эмбеддер без моделей: dense = хеш, sparse = мешок слов."""

    def __init__(self, dim=8, sparse=True, name="fake"):
        self.dense_size, self.has_sparse, self.name = dim, sparse, name

    def _dense(self, text):
        h = hashlib.sha256(text.encode()).digest()
        v = [b / 255 - 0.5 for b in h[: self.dense_size]]
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def _sparse(self, text):
        idx = sorted({int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in re.findall(r"\w+", text.lower())})
        return idx, [1.0] * len(idx)

    def embed_passages(self, texts):
        return [self._dense(t) for t in texts], ([self._sparse(t) for t in texts] if self.has_sparse else None)

    def embed_query(self, text):
        return self._dense(text), (self._sparse(text) if self.has_sparse else None)

    def info(self):
        return {"name": self.name}


def make_chunks(n=5, doc_id="faq_4"):
    return [Chunk(text=f"Карта: Карта{i}\nВопрос: Что делает карта {i}?\nОтвет: Делает {i}.",
                  metadata=ChunkMetadata(chunk_id=f"{doc_id}_{i}", doc_id=doc_id, source_type="faq",
                                         priority="community", source_version="v", snapshot_date="2026-10-10",
                                         card_name=f"Карта{i}", card_number=i, set_name="4. Ложные боги",
                                         faq_kind="qna", approval_status="unknown")) for i in range(n)]


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeModels:
    Distance = SimpleNamespace(COSINE="Cosine")
    Modifier = SimpleNamespace(IDF="idf")
    PayloadSchemaType = SimpleNamespace(KEYWORD="keyword")
    Fusion = SimpleNamespace(RRF="rrf")
    _cache = {}

    def __getattr__(self, name):
        return self._cache.setdefault(name, type(name, (_Obj,), {}))


class FakeClient:
    def __init__(self):
        self.cols, self.upserts = {}, 0

    def collection_exists(self, n):
        return n in self.cols

    def delete_collection(self, n):
        del self.cols[n]

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        self.cols[collection_name] = {"vectors": vectors_config, "sparse": sparse_vectors_config, "points": {}, "indexes": []}

    def create_payload_index(self, name, field_name, field_schema):
        self.cols[name]["indexes"].append(field_name)

    def get_collection(self, n):
        c = self.cols[n]
        vec = {k: SimpleNamespace(size=v.size) for k, v in c["vectors"].items()}
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vec, sparse_vectors=c["sparse"])))

    def upsert(self, collection_name, points):
        self.upserts += 1
        for p in points:
            self.cols[collection_name]["points"][p.id] = p

    def delete(self, collection_name, points_selector):
        ids = set(points_selector.filter.must[0].match.any)
        pts = self.cols[collection_name]["points"]
        for k in [k for k, p in pts.items() if p.payload["doc_id"] in ids]:
            del pts[k]

    def count(self, collection_name, exact):
        return SimpleNamespace(count=len(self.cols[collection_name]["points"]))


class FakeClientTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(indexer_mod, "_models", lambda: FakeModels())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = FakeClient()
        self.ix = Indexer(self.client)

    def test_creates_hybrid_collection_with_named_vectors_and_indexes(self):
        name = self.ix.load_data(make_chunks(), FakeEmbedder())
        col = self.client.cols[name]
        self.assertEqual(list(col["vectors"]), [DENSE])
        self.assertEqual(col["vectors"][DENSE].size, 8)
        self.assertEqual(list(col["sparse"]), [SPARSE])
        self.assertEqual(col["sparse"][SPARSE].modifier, "idf")
        self.assertIn("source_type", col["indexes"])
        self.assertIn("priority", col["indexes"])

    def test_points_have_deterministic_ids_and_full_payload(self):
        chunks = make_chunks(2)
        name = self.ix.load_data(chunks, FakeEmbedder())
        pt = self.client.cols[name]["points"][point_id("faq_4_1")]
        self.assertEqual(set(pt.vector), {DENSE, SPARSE})
        self.assertEqual(pt.payload["card_name"], "Карта1")
        self.assertEqual(pt.payload["text"], chunks[1].text)
        self.assertNotIn("section_id", pt.payload)        # None-поля в payload не попадают

    def test_dense_only_embedder_has_no_sparse_config(self):
        name = self.ix.load_data(make_chunks(), FakeEmbedder(sparse=False))
        col = self.client.cols[name]
        self.assertIsNone(col["sparse"])
        self.assertTrue(all(set(p.vector) == {DENSE} for p in col["points"].values()))

    def test_reload_is_idempotent(self):
        name = self.ix.load_data(make_chunks(5), FakeEmbedder())
        self.ix.load_data(make_chunks(5), FakeEmbedder(), name, recreate=False)
        self.assertEqual(self.ix.count(name), 5)

    def test_sync_mode_removes_stale_points_of_same_doc_only(self):
        name = self.ix.load_data(make_chunks(5, "faq_4") + make_chunks(2, "faq_5"), FakeEmbedder())
        self.assertEqual(self.ix.count(name), 7)
        self.ix.load_data(make_chunks(3, "faq_4"), FakeEmbedder(), name, recreate=False)   # 2 строки удалили из таблицы
        self.assertEqual(self.ix.count(name), 5)                                           # 3 + нетронутые faq_5

    def test_recreate_drops_everything(self):
        name = self.ix.load_data(make_chunks(5), FakeEmbedder())
        self.ix.load_data(make_chunks(2), FakeEmbedder(), name, recreate=True)
        self.assertEqual(self.ix.count(name), 2)

    def test_incompatible_embedder_is_rejected(self):
        name = self.ix.load_data(make_chunks(), FakeEmbedder(dim=8))
        with self.assertRaises(RuntimeError):
            self.ix.load_data(make_chunks(), FakeEmbedder(dim=16), name, recreate=False)
        with self.assertRaises(RuntimeError):
            self.ix.load_data(make_chunks(), FakeEmbedder(dim=8, sparse=False), name, recreate=False)

    def test_batching(self):
        self.ix.load_data(make_chunks(10), FakeEmbedder(), batch_size=4)
        self.assertEqual(self.client.upserts, 3)

    def test_default_collection_name_is_per_model(self):
        self.assertEqual(default_collection_name(FakeEmbedder(name="intfloat/multilingual-e5-large+Qdrant/bm25")),
                         "berserk_intfloat_multilingual_e5_large_qdrant_bm25")


try:
    from qdrant_client import QdrantClient
except ImportError:   # pragma: no cover
    QdrantClient = None


@unittest.skipIf(QdrantClient is None, "qdrant-client не установлен")
class LocalQdrantTests(unittest.TestCase):
    def setUp(self):
        self.ix = Indexer(QdrantClient(":memory:"))
        self.emb = FakeEmbedder()

    def test_load_count_and_idempotency(self):
        name = self.ix.load_data(make_chunks(6), self.emb)
        self.assertEqual(self.ix.count(name), 6)
        self.ix.load_data(make_chunks(6), self.emb, name, recreate=False)
        self.assertEqual(self.ix.count(name), 6)

    def test_sync_removes_stale(self):
        name = self.ix.load_data(make_chunks(6), self.emb)
        self.ix.load_data(make_chunks(4), self.emb, name, recreate=False)
        self.assertEqual(self.ix.count(name), 4)

    def test_hybrid_dense_and_sparse_search(self):
        chunks = make_chunks(8)
        name = self.ix.load_data(chunks, self.emb)
        for mode in ("hybrid", "dense", "sparse"):
            hits = self.ix.search(name, self.emb, chunks[3].text, limit=3, mode=mode)
            self.assertEqual(hits[0]["chunk_id"], "faq_4_3", mode)

    def test_filter_by_priority(self):
        name = self.ix.load_data(make_chunks(4), self.emb)
        self.assertEqual(self.ix.search(name, self.emb, "карта", limit=3, priority="official"), [])
        self.assertTrue(self.ix.search(name, self.emb, "карта", limit=3, priority="community"))


if __name__ == "__main__":
    unittest.main()
