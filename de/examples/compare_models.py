"""Сравнение эмбеддинг-моделей на одних и тех же чанках.

Для каждой пары (dense, sparse) создаётся своя коллекция, затем прогоняются запросы из data/smoke_queries.json
и считается, в скольких случаях ожидаемый чанк попал в топ-3. Запуск (Qdrant должен быть поднят):

    python examples/compare_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import SparseTextEmbedding, TextEmbedding   # noqa: E402

from berserk_ingest.config import DATA_DIR                 # noqa: E402
from berserk_ingest.indexer import Loader                  # noqa: E402
from berserk_ingest.pipeline import build                  # noqa: E402

# Что сравниваем: (dense-модель, sparse-модель или None). Добавить модель = добавить строку.
CANDIDATES = [
    ("intfloat/multilingual-e5-large", None),
    ("intfloat/multilingual-e5-large", "Qdrant/bm25"),
]

result, _ = build(["faq", "rules"])
queries = json.loads((DATA_DIR / "smoke_queries.json").read_text(encoding="utf-8"))
present = {c.metadata.source_type for c in result.chunks}
queries = [q for q in queries if q.get("source_type") in present]

loader = Loader()   # QdrantClient(url=QDRANT_URL)
for dense_name, sparse_name in CANDIDATES:
    dense = TextEmbedding(dense_name)
    sparse = SparseTextEmbedding(sparse_name, language="russian") if sparse_name else None

    collection = loader.load_data(result.chunks, embedder=dense, sparse_embedder=sparse)
    hits = 0
    for q in queries:
        found = loader.search(collection, dense, q["query"], sparse_embedder=sparse, limit=3,
                              source_type=q.get("source_type"))
        hits += any(all(str(h.get(k)) == str(v) for k, v in q["expect"].items()) for h in found)
    print(f"{collection}: hit@3 = {hits}/{len(queries)}")
