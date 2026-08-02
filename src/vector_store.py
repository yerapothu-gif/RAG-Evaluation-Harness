"""
Vector store module using ChromaDB for dense vector retrieval.
Uses sentence-transformers for local embeddings (no API cost).
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from src.config import EMBEDDING_MODEL, CHROMA_DB_DIR


_client: Optional[chromadb.PersistentClient] = None
_embed_fn = None


def _get_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
    global _client
    if _client is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def _get_embed_fn():
    """Get or create the embedding function."""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embed_fn


def build_collection(
    collection_name: str,
    chunks: List[Dict[str, str]],
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Build or get a ChromaDB collection from text chunks.
    """
    client = _get_client()
    embed_fn = _get_embed_fn()

    if force_rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    # Check if collection already exists with data
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
        if collection.count() > 0 and not force_rebuild:
            return collection
    except Exception:
        pass

    # Create fresh collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Add chunks in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"char_count": c["char_count"]} for c in batch],
        )

    return collection


def query_collection(
    collection_name: str,
    query_text: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Query a ChromaDB collection and return ranked results.
    """
    client = _get_client()
    embed_fn = _get_embed_fn()

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    formatted = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1 - distance  # cosine distance to similarity
            formatted.append({
                "id": results["ids"][0][i],
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

    return formatted


def build_index(force_rebuild: bool = False):
    """Build both Setting A and Setting B vector indexes."""
    from src.config import SETTING_A, SETTING_B, KNOWLEDGE_BASE_PATH
    from src.document_loader import load_and_split

    print("Loading knowledge base...")
    chunks_a = load_and_split(
        KNOWLEDGE_BASE_PATH,
        SETTING_A["chunk_size"],
        SETTING_A["chunk_overlap"],
    )
    print(f"  Setting A: {len(chunks_a)} chunks (size={SETTING_A['chunk_size']})")

    chunks_b = load_and_split(
        KNOWLEDGE_BASE_PATH,
        SETTING_B["chunk_size"],
        SETTING_B["chunk_overlap"],
    )
    print(f"  Setting B: {len(chunks_b)} chunks (size={SETTING_B['chunk_size']})")

    print("Building Setting A collection (dense)...")
    build_collection(SETTING_A["collection_name"], chunks_a, force_rebuild)

    print("Building Setting B collection (for hybrid)...")
    build_collection(SETTING_B["collection_name"], chunks_b, force_rebuild)

    print("Index build complete!")
    return chunks_a, chunks_b
