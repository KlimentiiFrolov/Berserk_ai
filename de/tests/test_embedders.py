"""Тесты адаптеров «модель с интерфейсом FastEmbed -> Embedder» и Loader.load_data(embedder=...).

Настоящие fastembed/qdrant-client не нужны: используются заглушки с тем же интерфейсом
(`.embed(texts)` -> итератор numpy-подобных векторов; у sparse элементы с `.indices` и `.values`).
"""
import unittest
from unittest import mock

from berserk_ingest import indexer as indexer_mod
from berserk_ingest.embedders import (
    Embedder, FastEmbedLikeDense, FastEmbedLikeSparse, as_embedder, wrap_fastembed,
)
from berserk_ingest.indexer import DENSE, SPARSE, Loader, Indexer
from berserk_ingest.schema import point_id
from tests.test_indexer import FakeClient, FakeEmbedder, FakeModels, make_chunks


class _Arr(list):
    """Как numpy-массив: есть .tolist()."""
    def tolist(self):
        return list(self)


class StubDense:
    """Интерфейс fastembed.TextEmbedding: model_name + embed(Iterable[str]) -> генератор."""

    def __init__(self, model_name="intfloat/multilingual-e5-large", dim=4):
        self.model_name, self.dim, self.seen = model_name, dim, []

    def embed(self, texts, **kw):
        for t in texts:
            self.seen.append(t)
            yield _Arr([float(len(t) % 7)] * self.dim)


class _Sp:
    def __init__(self, indices, values):
        self.indices, self.values = _Arr(indices), _Arr(values)


class StubSparse:
    """Интерфейс fastembed.SparseTextEmbedding: embed + query_embed."""

    model_name = "Qdrant/bm25"

    def __init__(self):
        self.query_calls = []

    def embed(self, texts, **kw):
        for t in texts:
            yield _Sp(sorted({len(w) for w in t.split()}), [1.0] * len({len(w) for w in t.split()}))

    def query_embed(self, query, **kw):
        self.query_calls.append(query)
        yield _Sp([1], [2.0])


class PrefixTests(unittest.TestCase):
    def test_e5_gets_passage_and_query_prefixes(self):
        d = StubDense("intfloat/multilingual-e5-large")
        be = FastEmbedLikeDense(d)
        d.seen.clear()
        be.embed_passages(["текст"])
        be.embed_query("вопрос")
        self.assertEqual(d.seen, ["passage: текст", "query: вопрос"])

    def test_non_e5_model_gets_no_prefix(self):
        d = StubDense("BAAI/bge-m3")
        be = FastEmbedLikeDense(d)
        d.seen.clear()
        be.embed_passages(["текст"])
        self.assertEqual(d.seen, ["текст"])

    def test_explicit_prefix_overrides_auto(self):
        d = StubDense("BAAI/bge-m3")
        be = FastEmbedLikeDense(d, passage_prefix="doc: ")
        d.seen.clear()
        be.embed_passages(["x"])
        self.assertEqual(d.seen, ["doc: x"])

    def test_dim_and_name_are_taken_from_model(self):
        be = FastEmbedLikeDense(StubDense("m/x", dim=6))
        self.assertEqual((be.name, be.dim), ("m/x", 6))
        self.assertIsInstance(be.embed_passages(["a"])[0][0], float)


class SparseTests(unittest.TestCase):
    def test_passages_use_embed_and_queries_use_query_embed(self):
        sp = StubSparse()
        be = FastEmbedLikeSparse(sp)
        self.assertEqual(be.embed_passages(["аб вгд"])[0], ([2, 3], [1.0, 1.0]))
        self.assertEqual(be.embed_query("запрос"), ([1], [2.0]))
        self.assertEqual(sp.query_calls, ["запрос"])

    def test_indices_are_plain_ints(self):
        idx, val = FastEmbedLikeSparse(StubSparse()).embed_passages(["аб"])[0]
        self.assertTrue(all(type(i) is int for i in idx) and all(type(v) is float for v in val))


class AsEmbedderTests(unittest.TestCase):
    def test_fastembed_like_objects_are_wrapped(self):
        emb = as_embedder(StubDense(), StubSparse())
        self.assertIsInstance(emb, Embedder)
        self.assertTrue(emb.has_sparse)
        self.assertEqual(emb.name, "intfloat/multilingual-e5-large+Qdrant/bm25")

    def test_dense_only(self):
        self.assertFalse(as_embedder(StubDense()).has_sparse)

    def test_ready_embedder_passes_through(self):
        fake = FakeEmbedder()
        self.assertIs(as_embedder(fake), fake)

    def test_ready_embedder_plus_sparse_is_an_error(self):
        with self.assertRaises(ValueError):
            as_embedder(FakeEmbedder(), StubSparse())

    def test_object_without_embed_is_rejected(self):
        with self.assertRaises(TypeError):
            as_embedder(object())

    def test_wrap_fastembed_matches_as_embedder(self):
        self.assertEqual(wrap_fastembed(StubDense(), StubSparse()).name, as_embedder(StubDense(), StubSparse()).name)


class LoaderWithFastEmbedLikeModelsTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(indexer_mod, "_models", lambda: FakeModels())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = FakeClient()
        self.loader = Loader(self.client)

    def test_hybrid_load_with_raw_models(self):
        chunks = make_chunks(3)
        name = self.loader.load_data(chunks, embedder=StubDense(dim=4), sparse_embedder=StubSparse())
        col = self.client.cols[name]
        self.assertEqual(name, "berserk_intfloat_multilingual_e5_large_qdrant_bm25")
        self.assertEqual(col["vectors"][DENSE].size, 4)
        self.assertEqual(list(col["sparse"]), [SPARSE])
        pt = col["points"][point_id("faq_4_0")]
        self.assertEqual(set(pt.vector), {DENSE, SPARSE})
        self.assertEqual(len(pt.vector[DENSE]), 4)

    def test_dense_only_with_raw_model(self):
        name = self.loader.load_data(make_chunks(2), embedder=StubDense("BAAI/bge-m3"))
        self.assertIsNone(self.client.cols[name]["sparse"])
        self.assertEqual(name, "berserk_baai_bge_m3")

    def test_passages_are_embedded_with_prefix_but_payload_text_is_clean(self):
        dense = StubDense()
        chunks = make_chunks(1)
        name = self.loader.load_data(chunks, embedder=dense)
        self.assertIn("passage: " + chunks[0].text, dense.seen)
        self.assertEqual(self.client.cols[name]["points"][point_id("faq_4_0")].payload["text"], chunks[0].text)

    def test_different_models_go_to_different_collections(self):
        a = self.loader.load_data(make_chunks(2), embedder=StubDense("m/a"))
        b = self.loader.load_data(make_chunks(2), embedder=StubDense("m/b"))
        self.assertNotEqual(a, b)
        self.assertEqual(set(self.client.cols), {a, b})

    def test_loader_is_indexer(self):
        self.assertIs(Loader, Indexer)


if __name__ == "__main__":
    unittest.main()
