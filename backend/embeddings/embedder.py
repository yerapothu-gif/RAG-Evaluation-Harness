"""
Embedding wrapper using sentence-transformers (all-MiniLM-L6-v2).
Local, free, no API cost. 384-dimensional vectors.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"[embedder] Loading {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[embedder] Model loaded. Dim={_model.get_sentence_embedding_dimension()}")
    return _model


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.

    Returns:
        List of 384 floats (unit-normalized).
    """
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
    """
    Embed a list of texts in batches.

    Args:
        texts: Input strings
        batch_size: How many to embed at once
        show_progress: Whether to show a tqdm progress bar

    Returns:
        List of embedding vectors (each 384 floats)
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return [v.tolist() for v in vectors]


def get_embedding_dim() -> int:
    """Return the dimensionality of the embedding model."""
    return _get_model().get_sentence_embedding_dimension()


if __name__ == "__main__":
    vec = embed_text("Who is Kattappa?")
    print(f"Embedding dim: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
