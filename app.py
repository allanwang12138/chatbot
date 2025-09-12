# app.py
import os, re, datetime
from dotenv import load_dotenv
import streamlit as st
import openai

from auth import load_credentials, require_auth
from retrieval import (
    build_clients,
    get_textbook_store,
    get_user_memory_store,
    COLLECTION_MAP,
    ping_collection,
)
import qa
from memory_store import upload_session_to_user_memory
from voice_speech import synthesize, play_in_streamlit

from langchain.schema import Document
from qdrant_client.http import models as qmodels  # Filter types

# ------------------ ENV ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL     = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # may be None/empty for local/public Qdrant
openai.api_key = OPENAI_API_KEY  # for voice_speech

# ------------------ AUTH ------------------
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS)  # no GitHub logs

# ------------------ STORES ------------------
textbook = st.session_state.get("textbook")
username = st.session_state.get("username")
level    = st.session_state.get("experience_level", "Intermediate")

embeddings, client = build_clients(OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
collection = COLLECTION_MAP.get(textbook)
if not collection:
    st.error(f"❌ No collection configured for textbook: {textbook}")
    st.stop()

db = get_textbook_store(client, collection, embeddings)
memory_db = get_user_memory_store(client, embeddings)
st.caption(ping_collection(client, collection))

USER_MEMORY_COLLECTION = "user_memory"

# ------------------ HISTORY HELPERS (Qdrant) ------------------
def _parse_qa_from_memory(text: str) -> tuple[str, str]:
    """Extract Q and A from 'Q: ...\\nA: ...'; be forgiving if format differs."""
    if not text:
        return "", ""
    m = re.search(r"Q:\s*(.*?)\nA:\s*(.*)\Z", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()  # fallback: treat all as answer

def _safe_iso(ts: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return datetime.datetime.min

def fetch_user_history_from_qdrant(
    _client,
    username: str,
    textbook: str,
    level: str,
    limit: int = 100,
) -> list[dict]:
    """Return recent Q/A items for the sidebar from the user_memory collection."""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="username", match=qmodels.MatchValue(value=username)),
            qmodels.FieldCondition(key="textbook", match=qmodels.MatchValue(value=textbook)),
            qmodels.FieldCondition(key="experience_level", match=qmodels.MatchValue(value=level)),
        ]
    )
    try:
        records, _ = _client.scroll(
            collection_name=USER_MEMORY_COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
    except Exception as e:
        st.warning(f"⚠️ Could not load history from Qdrant: {e}")
        return []

    items = []
    for rec in records or []:
        payload = (rec.payload or {})
        # LangChain Qdrant store flattens page_content + metadata into payload
        content = payload.get("page_content") or payload.get("text") or payload.get("content") or ""
        q, a = _parse_qa_from_memory(content)
        items.append({
            "question": q,
            "answer": a,
            "timestamp": payload.get("timestamp", ""),
            "option": payload.get("option", ""),
        })
    items.sort(key=lambda x: _safe_iso(x.get("timestamp", "")), reverse=True)
    return items

def upsert_interaction_to_memory(
    memory_db,
    *,
    username: str,
    textbook: str,
    level: str,
    option: str,
    question: str,
    answer: str,
    timestamp_iso: str,
):
    """Immediately add this Q/A to user_memory so it is retrievable right away."""
    content = f"Q: {question}\nA: {answer}"
    doc = Document(
        page_content=content,
        metadata={
            "username": username,
            "textbook": textbook,
            "experience_level": level,
            "option": option,
            "timestamp": timestamp_iso,
            "source": "user_memory",
        },
    )
    try:
        memory_db.add_documents([doc])
        # Prevent exit uploader from dupl
