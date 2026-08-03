"""
ChromaDB vector store for the Baahubali RAG pipeline.
Handles collection create/upsert/query for both Setting A and B.
Uses local sentence-transformers embeddings (no API cost).
"""
import os
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

from backend.config import EMBEDDING_MODEL, CHROMA_DB_DIR, SETTING_A, SETTING_B

_client: Optional[chromadb.PersistentClient] = None
_embed_fn = None


def _get_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client (singleton)."""
    global _client
    if _client is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def _get_embed_fn():
    """Get or create the SentenceTransformer embedding function (singleton)."""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embed_fn


def build_collection(
    collection_name: str,
    chunks: List[Dict],
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Build or retrieve a ChromaDB collection from text chunks.

    Args:
        collection_name: Name for the Chroma collection
        chunks: List of {id, text, char_count, metadata} dicts
        force_rebuild: If True, drop existing collection and rebuild

    Returns:
        The ChromaDB Collection object
    """
    client = _get_client()
    embed_fn = _get_embed_fn()

    if force_rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    # Return existing collection if already populated
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
        if collection.count() > 0 and not force_rebuild:
            print(f"[chroma_store] Collection '{collection_name}' already exists with {collection.count()} docs.")
            return collection
    except Exception:
        pass

    # Drop and recreate
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches of 50
    batch_size = 50
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"char_count": c.get("char_count", len(c["text"]))} for c in batch],
        )
    print(f"[chroma_store] Built '{collection_name}' with {total} chunks.")
    return collection


def query_collection(
    collection_name: str,
    query_text: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Query a ChromaDB collection and return ranked results.

    Returns:
        List of {id, text, similarity, metadata} dicts
    """
    client = _get_client()
    embed_fn = _get_embed_fn()

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    n = min(top_k, collection.count())
    if n == 0:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=n,
        include=["documents", "distances", "metadatas"],
    )

    formatted = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = max(0.0, 1.0 - distance)  # cosine distance to similarity
            formatted.append({
                "id": results["ids"][0][i],
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

    return formatted


def collection_count(collection_name: str) -> int:
    """Return the number of documents in a collection (0 if not found)."""
    try:
        client = _get_client()
        embed_fn = _get_embed_fn()
        col = client.get_collection(name=collection_name, embedding_function=embed_fn)
        return col.count()
    except Exception:
        return 0


def build_both_indexes(
    chunks_a: List[Dict],
    chunks_b: List[Dict],
    force_rebuild: bool = False,
) -> None:
    """Build both Setting A and Setting B ChromaDB collections."""
    print("[chroma_store] Building Setting A collection...")
    build_collection(SETTING_A["collection_name"], chunks_a, force_rebuild)
    print("[chroma_store] Building Setting B collection...")
    build_collection(SETTING_B["collection_name"], chunks_b, force_rebuild)
    print("[chroma_store] Both collections ready.")
