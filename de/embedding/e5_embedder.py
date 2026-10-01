from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("intfloat/multilingual-e5-base")

def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return _model.encode(prefixed, normalize_embeddings=True).tolist()

def embed_query(text: str) -> list[float]:
    return _model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()