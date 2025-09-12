# retrieval.py
from __future__ import annotations
from typing import Any, Tuple, Dict
import re
import streamlit as st

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant as LcQdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


# -------------------- Collections --------------------
COLLECTION_MAP: Dict[str, str] = {
    "Introductory Macroeconomics": "introductory_macroeconomics_collection",
    "Introductory Microeconomics": "introductory_microeconomics_collection",
    "Statistics For Economics": "statistics_for_economics_collection",
    "MATHEMATICS Textbook for Class IX": "mathematics_textbook_for_class_ix_collection",
    "MATHEMATICS Textbook for Class X": "mathematics_textbook_for_class_x_collection",
    "MATHEMATICS Textbook for Class XI": "mathematics_textbook_for_class_xi_collection",
    "MATHEMATICS Textbook for Class XII PART I": "mathematics_textbook_for_class_xii_part_i_collection",
    "MATHEMATICS Textbook for Class XII PART II": "mathematics_textbook_for_class_xii_part_ii_collection",
}


# -------------------- Clients / Stores --------------------
@st.cache_resource(show_spinner=False)
def build_clients(
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str | None = None,
) -> tuple[OpenAIEmbeddings, QdrantClient]:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    return embeddings, client


@st.cache_resource(show_spinner=False)
def get_textbook_store(_client: QdrantClient, collection_name: str, _embeddings: OpenAIEmbeddings) -> LcQdrant:
    return LcQdrant(client=_client, collection_name=collection_name, embeddings=_embeddings)


@st.cache_resource(show_spinner=False)
def get_user_memory_store(_client: QdrantClient, _embeddings: OpenAIEmbeddings, collection_name: str = "user_memory") -> LcQdrant:
    return LcQdrant(client=_client, collection_name=collection_name, embeddings=_embeddings)


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
    Build retrievers and decide route.
    - Memory retriever uses strict payload filter.
    - Textbook retriever: try filtered; if the probe returns 0 hits, fall back to unfiltered.
    """
    user_filter = {"username": username, "textbook": textbook, "experience_level": level}

    # Memory retriever + quick probe
    memory_retriever = memory_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": memory_k, "filter": user_filter},
    )
    try:
        has_memory_match = len(memory_db.similarity_search_with_score(query, k=1, filter=user_filter)) > 0
    except Exception:
        has_memory_match = False

    # Textbook retriever: filtered probe; if empty, unfiltered
    try:
        probe = db.similarity_search_with_score(query, k=1, filter={"textbook": textbook})
        if probe and len(probe) > 0:
            textbook_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": textbook_k, "filter": {"textbook": textbook}},
            )
        else:
            textbook_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": textbook_k},
            )
    except Exception:
        textbook_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": textbook_k},
        )

    chosen = memory_retriever if has_memory_match else textbook_retriever
    return chosen, has_memory_match, textbook_retriever


def top_context_from_sources(source_docs: list[Document], max_chars: int = 1200) -> str:
    if not source_docs:
        return ""
    text = (source_docs[0].page_content or "").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def ping_collection(client: QdrantClient, collection_name: str) -> str:
    try:
        client.get_collection(collection_name)
        return f"🗂️ Using collection **{collection_name}** · status: Active"
    except Exception:
        return f"🗂️ Using collection **{collection_name}**"


# -------------------- In-scope Gate --------------------
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _score_to_similarity(score: float) -> float:
    """Convert Qdrant/LC scores to [0,1] similarity. (Distance → 1 - distance.)"""
    try:
        s = float(score)
    except Exception:
        return 0.0
    if 0.0 <= s <= 2.0:           # cosine distance in [0,1] or [0,2]
        return max(0.0, min(1.0, 1.0 - s))
    if -1.0 <= s <= 1.0:          # already similarity-ish
        return max(0.0, min(1.0, s))
    return 0.0


def is_in_scope(
    query: str,
    textbook: str,
    db: LcQdrant,
    *,
    k: int = 4,
    sim_cutoff: float = 0.05,   # VERY lenient – only block truly off-topic queries
) -> Tuple[bool, float]:
    """
    Decide if the query is in scope based on textbook content only.
    1) Try filtered probe; if it returns 0 hits, retry unfiltered.
    2) If we get any hits at all, treat it as in-scope unless the top similarity is *extremely* low.
    """
    # Probe filtered; if empty, try unfiltered
    try:
        hits = db.similarity_search_with_score(query, k=k, filter={"textbook": textbook})
        if not hits:
            hits = db.similarity_search_with_score(query, k=k)
    except Exception:
        hits = db.similarity_search_with_score(query, k=k)

    if not hits:
        return False, 0.0

    sims = [_score_to_similarity(score) for _, score in hits]
    max_sim = max(sims) if sims else 0.0

    # Lenient gate: treat as in-scope if we have any reasonable hit
    return (max_sim >= sim_cutoff, max_sim)
