# retrieval.py
from __future__ import annotations
from typing import Any, Tuple, Dict, List
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
    """Choose memory vs textbook retriever; textbook has filter→unfiltered fallback."""
    user_filter = {"username": username, "textbook": textbook, "experience_level": level}

    # Memory retriever + probe
    memory_retriever = memory_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": memory_k, "filter": user_filter},
    )
    try:
        has_memory_match = len(memory_db.similarity_search_with_score(query, k=1, filter=user_filter)) > 0
    except Exception:
        has_memory_match = False

    # Textbook retriever: probe filtered; if empty, build unfiltered
    try:
        probe = db.similarity_search_with_score(query, k=1, filter={"textbook": textbook})
        if probe:
            textbook_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": textbook_k, "filter": {"textbook": textbook}},
            )
        else:
            textbook_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": textbook_k})
    except Exception:
        textbook_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": textbook_k})

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
_STOPWORDS = {
    "textbook", "for", "class", "part", "i", "ii", "iii", "the", "of", "and",
    "introduction", "introductory", "a", "an", "to", "in", "on",
}

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if len(t) >= 4 and t not in _STOPWORDS]

def _score_to_similarity(score: float) -> float:
    """Map distance→similarity. Cosine distance in [0,1] (or [0,2]) → 1 - d."""
    try:
        s = float(score)
    except Exception:
        return 0.0
    if 0.0 <= s <= 2.0:
        return max(0.0, min(1.0, 1.0 - s))
    if -1.0 <= s <= 1.0:
        return max(0.0, min(1.0, s))
    return 0.0

def is_in_scope(
    query: str,
    textbook: str,
    db: LcQdrant,
    *,
    k: int = 5,
) -> Tuple[bool, float]:
    """
    Scope gate against textbook content only.
    - Try filtered by textbook; if empty, retry unfiltered.
    - Multi-signal rule: require decent top hit + some supporting evidence.
    """
    # 1) Probe textbook store
    try:
        hits = db.similarity_search_with_score(query, k=k, filter={"textbook": textbook})
        if not hits:
            hits = db.similarity_search_with_score(query, k=k)  # fallback if filter yields 0
    except Exception:
        hits = db.similarity_search_with_score(query, k=k)

    if not hits:
        return False, 0.0

    # 2) Convert scores to similarities
    sims = [_score_to_similarity(score) for _, score in hits]
    sims_sorted = sorted(sims, reverse=True)
    max_sim = sims_sorted[0]
    avg_top3 = sum(sims_sorted[:3]) / min(3, len(sims_sorted))
    # weak-but-related count (low bar)
    count_lo = sum(1 for s in sims_sorted[:5] if s >= 0.06)

    # Optional one-line debug
    if st.session_state.get("debug_scope"):
        st.caption(f"[scope] sims={ [round(x,4) for x in sims_sorted] } "
                   f"max={max_sim:.3f} avg3={avg_top3:.3f} count≥0.06={count_lo}")

    # 3) Dynamic thresholds
    q_tokens = set(_tokens(query))
    title_tokens = set(_tokens(textbook))
    overlap = bool(q_tokens & title_tokens)

    # Baselines tuned for small sim ranges
    T1 = 0.06   # top must clear this
    T2 = 0.05   # avg of top-3 must be reasonably related

    # Short queries are noisier → nudge up slightly
    qlen = max(1, len(_norm(query).split()))
    if qlen <= 4:
        T1 += 0.02
        T2 += 0.01

    # No topical overlap with the title? be stricter
    if not overlap:
        T1 += 0.03
        T2 += 0.02

    # Extra signal: clear dominance over #2
    gap_ok = (len(sims_sorted) > 1) and (max_sim - sims_sorted[1] >= 0.02)

    in_scope = (max_sim >= T1) and (avg_top3 >= T2 or count_lo >= 2 or gap_ok)
    return in_scope, max_sim
