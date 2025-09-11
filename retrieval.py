# retrieval.py
from __future__ import annotations
from typing import Any, Tuple
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant as LcQdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Human-name -> Qdrant collection
COLLECTION_MAP: dict[str, str] = {
    "Introductory Macroeconomics": "introductory_macroeconomics_collection",
    "Introductory Microeconomics": "introductory_microeconomics_collection",
    "Statistics For Economics": "statistics_for_economics_collection",
    "MATHEMATICS Textbook for Class IX": "mathematics_textbook_for_class_ix_collection",
    "MATHEMATICS Textbook for Class X": "mathematics_textbook_for_class_x_collection",
    "MATHEMATICS Textbook for Class XI": "mathematics_textbook_for_class_xi_collection",
    "MATHEMATICS Textbook for Class XII PART I": "mathematics_textbook_for_class_xii_part_i_collection",
    "MATHEMATICS Textbook for Class XII PART II": "mathematics_textbook_for_class_xii_part_ii_collection",
}

@st.cache_resource(show_spinner=False)
def build_clients(openai_api_key: str, qdrant_url: str, qdrant_api_key: str | None = None) -> tuple[OpenAIEmbeddings, QdrantClient]:
    return OpenAIEmbeddings(openai_api_key=openai_api_key), QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)

# retrieval.py

@st.cache_resource(show_spinner=False)
def get_textbook_store(_client: QdrantClient, collection_name: str, _embeddings: OpenAIEmbeddings) -> LcQdrant:
    """LangChain Qdrant store for textbook collection."""
    return LcQdrant(client=_client, collection_name=collection_name, embeddings=_embeddings)


@st.cache_resource(show_spinner=False)
def get_user_memory_store(_client: QdrantClient, _embeddings: OpenAIEmbeddings, collection_name: str = "user_memory") -> LcQdrant:
    """LangChain Qdrant store for user-specific memory collection."""
    return LcQdrant(client=_client, collection_name=collection_name, embeddings=_embeddings)


def make_retrievers(
    query: str, username: str, textbook: str, level: str,
    db: LcQdrant, memory_db: LcQdrant, *, memory_k: int = 3, textbook_k: int = 5
) -> tuple[Any, bool, Any]:
    user_filter = {"username": username, "textbook": textbook, "experience_level": level}
    memory_retriever = memory_db.as_retriever(search_type="similarity", search_kwargs={"k": memory_k, "filter": user_filter})
    has_memory_match = False
    try:
        has_memory_match = len(memory_db.similarity_search_with_score(query, k=1, filter=user_filter)) > 0
    except Exception:
        has_memory_match = False

    try:
        _ = db.similarity_search_with_score(query, k=1, filter={"textbook": textbook})
        textbook_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": textbook_k, "filter": {"textbook": textbook}})
    except Exception:
        textbook_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": textbook_k})

    return (memory_retriever if has_memory_match else textbook_retriever), has_memory_match, textbook_retriever

def top_context_from_sources(source_docs: list[Document], max_chars: int = 1200) -> str:
    if not source_docs: return ""
    text = (source_docs[0].page_content or "").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

def ping_collection(client: QdrantClient, collection_name: str) -> str:
    try:
        client.get_collection(collection_name)
        return f"🗂️ Using collection **{collection_name}** · status: Active"
    except Exception:
        return f"🗂️ Using collection **{collection_name}**"


def is_in_scope(
    query: str,
    textbook: str,
    db: LcQdrant,
    *,
    k: int = 3,
    sim_threshold: float = 0.72,   # tune 0.68–0.78 for your data
) -> tuple[bool, float]:
    """
    Quick semantic gate: if no textbook doc is close enough, treat query as out-of-scope.
    Assumes Qdrant returns cosine *distance* as score; we convert to similarity ≈ 1 - distance.
    Returns (in_scope_bool, max_similarity).
    """
    try:
        hits = db.similarity_search_with_score(query, k=k, filter={"textbook": textbook})
    except Exception:
        hits = db.similarity_search_with_score(query, k=k)

    if not hits:
        return False, 0.0

    sims = []
    for _, score in hits:
        try:
            sims.append(1.0 - float(score))  # distance → similarity
        except Exception:
            sims.append(float(score) if isinstance(score, (int, float)) else 0.0)

    max_sim = max(sims) if sims else 0.0
    return (max_sim >= sim_threshold, max_sim)
