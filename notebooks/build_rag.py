import asyncio
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from core.config import ModelsSettings, get_settings, MODELS_DIR, DATA_DIR, PROJECT_DIR

from load_fixtures import load_all_fixtures

settings = get_settings()

COLLECTIONS = settings.qdrant.collections
DENSE_MODEL = settings.models.dense
SPARSE_MODEL = settings.models.sparse
BATCH_SIZE = 25

print(PROJECT_DIR)
load_dotenv(PROJECT_DIR / ".env")
NAMESPACE = uuid.UUID(os.environ["NAMESPACE_UUID"])


def main():
    client = QdrantClient(url=settings.qdrant.url)

    # Пересоздание коллекций
    for name in COLLECTIONS:
        client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        print(f"{name}: коллекция создана")

    # Эмбеддеры
    dense = TextEmbedding(DENSE_MODEL, cache_dir=str(MODELS_DIR / DENSE_MODEL))
    sparse = SparseTextEmbedding(SPARSE_MODEL, cache_dir=str(MODELS_DIR / SPARSE_MODEL))

    # Загрузка данных
    fixtures = load_all_fixtures(DATA_DIR)
    for name in COLLECTIONS:
        chunks = list(fixtures[name])
        texts = [c.text for c in chunks]
        dense_vecs = dense.embed(["passage: " + t for t in texts])
        sparse_vecs = sparse.embed(texts)

        points = (
            models.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, c.id)),
                vector={
                    "dense": d.tolist(),
                    "sparse": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload=c.metadata,
            )
            for c, d, s in zip(chunks, dense_vecs, sparse_vecs)
        )
        client.upload_points(collection_name=name, points=points, batch_size=BATCH_SIZE)
        print(f"{name}: загружено {len(chunks)}")


async def test(query: str):
    from rag.retrieval import Retriever

    for item in await Retriever().retrieve(query=query, top_k=5):
        print(item)


if __name__ == "__main__":
    main()
    if "--test" in sys.argv:
        idx = sys.argv.index("--test") + 1
        asyncio.run(test(sys.argv[idx] if idx < len(sys.argv) else "нельзя блокировать без Дозора"))