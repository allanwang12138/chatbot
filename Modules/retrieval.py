# retrieval.py
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant as LcQdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


# ---------- Client & Embeddings ----------
@st.cache_resource(show_spinner=False)
def build_clients(openai_api_key: str, qdrant_url: str, qdrant_api_key: str | None = None) -> tuple[OpenAIEmbeddings, QdrantClient]:
    """
    Create and cache heavy resources: OpenAI embeddings + Qdrant client.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # QdrantClient accepts api_key=None for public/local instances
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)

    return embeddings, client


# ---------- Vector stores ----------
@st.cache_resource(show_spinner=False)
def get_textbook_store(client: QdrantClient, collection_name: str, embeddings: OpenAIEmbeddings) -> LcQdrant:
    """
    LangChain Qdrant store for textbook collection.
    """
    return LcQdrant(client=client, collection_name=collection_name, embeddings=embeddings)


@st.cache_resource(show_spinner=False)
def get_user_memory_store(client: QdrantClient, embeddings: OpenAIEmbeddings, collection_name: str = "user_memory") -> LcQdrant:
    """
    LangChain Qdrant store for user-specific memory collection shared across subjects.
    """
    return LcQdrant(client=client, collection_name=collection_name, embeddings=embeddings)


# ---------- Retriever factory with routing probe ----------
def make_retrievers(
    query: str,
    username: str,
    textbook: str,
    level: str,
    db: LcQdrant,
    memory_db: LcQdrant,
    *,
    memory_k: int = 3,
    textbook_k: int = 5,
) -> tuple[Any, bool, Any]:
    """
    Build retrievers for user memory (strict payload filter) and textbook.
    Probe memory first; if no hit, route to textbook.
    Returns: (chosen_retriever, has_memory_match, textbook_retriever)
    """
    # --- memory filter payload (aligns with your upload payload) ---
    user_filter = {
        "username": username,
        "textbook": textbook,
        "experience_level": level,
        # "source": "user_memory",  # if you set this on upload, uncomment to be extra strict
    }

    # Memory retriever (strict)
    memory_retriever = memory_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": memory_k, "filter": user_filter},
    )

    # Probe for a quick route decision
    has_memory_match = False
    try:
        probe = memory_db.similarity_search_with_score(query, k=1, filter=user_filter)
        has_memory_match = len(probe) > 0
    except Exception:
        # Swallow & route to textbook gracefully
        has_memory_match = False

    # Textbook retriever (prefer filtered by textbook field if present in payload)
    try:
        # quick test to see if filter works on the collection
        _ = db.similarity_search_with_score(query, k=1, filter={"textbook": textbook})
        textbook_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": textbook_k, "filter": {"textbook": textbook}},
        )
    except Exception:
        # If no payload or filter not supported, fall back to unfiltered similarity
        textbook_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": textbook_k},
        )

    chosen = memory_retriever if has_memory_match else textbook_retriever
    return chosen, has_memory_match, textbook_retriever


# ---------- Small utilities ----------
def top_context_from_sources(source_docs: list[Document], max_chars: int = 1200) -> str:
    """
    Extract a single concise snippet from source docs for logging.
    """
    if not source_docs:
        return ""
    text = (source_docs[0].page_content or "").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def ping_collection(client: QdrantClient, collection_name: str) -> str:
    """
    Returns a short human-readable status string for UI captions.
    """
    try:
        info = client.get_collection(collection_name)
        return f"🗂️ Using collection **{collection_name}** · status: Active"
    except Exception:
        return f"🗂️ Using collection **{collection_name}**"
