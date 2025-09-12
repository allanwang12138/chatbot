# app.py
import os
import datetime
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

# ------------------ ENV ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL     = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # may be None for local/public
openai.api_key = OPENAI_API_KEY  # for voice_speech

# ------------------ AUTH ------------------
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS)  # sets: username, textbook, experience_level, voice, chat_history_enabled, buffer_memory

# Safety: ensure buffer_memory exists
if "buffer_memory" not in st.session_state:
    from langchain.memory import ConversationBufferMemory
    st.session_state.buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
        output_key="answer",
    )

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

# ------------------ HISTORY (robust fetch) ------------------
def _extract_field(payload: dict, key: str, default=None):
    """Return payload[key] or payload['metadata'][key] if nested."""
    if not isinstance(payload, dict):
        return default
    if key in payload:
        return payload.get(key, default)
    md = payload.get("metadata")
    if isinstance(md, dict):
        return md.get(key, default)
    return default

def _parse_qa_block(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    # Very forgiving parse of "Q: ...\nA: ..."
    try:
        parts = text.splitlines()
        q, a = "", ""
        for line in parts:
            if line.strip().startswith("Q:"):
                q = line.split(":", 1)[1].strip()
            elif line.strip().startswith("A:"):
                a = line.split(":", 1)[1].strip()
        return q, a
    except Exception:
        return "", text

def load_history_from_qdrant_anyshape(client, username: str, textbook: str, limit: int = 10):
    """
    Read recent items from user_memory without server-side filter,
    then filter in Python to tolerate nested/flat payloads.
    """
    # Pull a reasonably large batch and sort newest-first in Python.
    try:
        points, _ = client.scroll(
            collection_name=USER_MEMORY_COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=512,
        )
    except Exception as e:
        st.warning(f"⚠️ Could not load history from Qdrant: {e}")
        return []

    items = []
    for p in points or []:
        pl = p.payload or {}
        u  = _extract_field(pl, "username", "")
        tb = _extract_field(pl, "textbook", "")
        if u != username or tb != textbook:
            continue

        ts = _extract_field(pl, "timestamp", "")
        opt = _extract_field(pl, "option", "")
        # page content may be stored under different keys; try all
        content = (pl.get("page_content") or pl.get("text") or pl.get("content") or
                   (_extract_field(pl, "page_content", "") or _extract_field(pl, "content", "")))
        q, a = _parse_qa_block(content)

        items.append({
            "timestamp": ts,
            "option": opt,
            "question": q,
            "answer": a,
            "oos": bool(_extract_field(pl, "oos", False)),
        })

    def _safe_iso(s: str):
        try:
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return datetime.datetime.min

    items.sort(key=lambda x: _safe_iso(x["timestamp"]), reverse=True)
    return items[:limit]

def upsert_interaction_to_memory(
    *,
    memory_db,
    username: str,
    textbook: str,
    level: str,
    option: str,
    question: str,
    answer: str,
    timestamp_iso: str,
    oos: bool,
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
            "oos": oos,        # reuse should ignore these
            "content": content # duplicate for robust history parsing
        },
    )
    try:
        memory_db.add_documents([doc])
        # Optional debug line:
        # st.caption("✅ Upserted to user_memory.")
    except Exception as e:
        st.warning(f"⚠️ Could not update personal memory: {e}")

# ------------------ UI: Chat history toggle (permissioned) ------------------
if "show_chat_history" not in st.session_state:
    st.session_state["show_chat_history"] = False

if st.session_state.get("chat_history_enabled", False):
    if st.button("📜 Show/Hide Chat History", key="toggle_chat_history"):
        st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

# ------------------ MAIN UI ------------------
st.title(f"📄 {textbook} Q&A App")
raw_query = st.text_input(f"Ask a question about {textbook}:")
col1, col2, col3 = st.columns(3)
option = None
with col1:
    if st.button("📖 Detailed Answer"): option = "Detailed Answer"
with col2:
    if st.button("✂️ Concise Answer"): option = "Concise Answer"
with col3:
    if st.button("🔊 Voice Answer"):   option = "Concise Answer + Voice"

# ------------------ ANSWER FLOW ------------------
if raw_query and option:
    is_voice = ("Voice" in option)

    result = qa.answer(
        raw_question=raw_query,
        option=option,
        username=username,
        level=level,
        textbook=textbook,
        memory=st.session_state.buffer_memory,
        db=db,
        memory_db=memory_db,
        logs=[],  # vector-only
        openai_api_key=OPENAI_API_KEY,
    )

    # Route hint
    if result.route == "reuse":
        st.info("🔁 Reused a similar answer from your personal memory.")
    elif result.route == "memory":
        st.caption("🧠 Using your personal memory context.")
    elif result.route == "textbook":
        st.caption("📚 Using textbook context.")
    elif result.route == "oos":
        st.warning(f"🚫 Outside the scope of **{textbook}**.")

    # Render: hide text if Voice Answer was chosen
    if is_voice:
        st.markdown("### 🔊 Voice Answer")
    else:
        st.markdown("### 📘 Answer")
        st.write(result.answer)

    # Supporting context (hide if out-of-scope)
    if result.route != "oos":
        with st.expander("📚 Show Supporting Context"):
            if not result.sources:
                st.write("No supporting documents returned.")
            else:
                top = result.sources[0]
                meta = getattr(top, "metadata", {}) or {}
                st.markdown(
                    f"**Source:** `{meta.get('source','unknown')}` — "
                    f"**Textbook:** {meta.get('textbook','N/A')}"
                )
                snippet = (top.page_content or "")
                st.write(snippet[:1200] + ("..." if len(snippet) > 1200 else ""))

    # Voice playback
    if is_voice:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"🎙️ Generating voice with '{voice_choice}'..."):
            audio_bytes = synthesize(result.answer, voice_choice)
            play_in_streamlit(audio_bytes)

    # Log interaction locally + immediate upsert (store OOS too, but flagged)
    st.session_state.setdefault("session_log", {"interactions": []})
    ts_iso = datetime.datetime.utcnow().isoformat()
    st.session_state["session_log"]["interactions"].append({
        "timestamp": ts_iso,
        "experience_level": level,
        "question": raw_query,
        "option": option,
        "answer": result.answer,
        "context": result.context_snippet,
        "textbook": textbook,
        "score": None,
    })

    upsert_interaction_to_memory(
        memory_db=memory_db,
        username=username,
        textbook=textbook,
        level=level,
        option=option,
        question=raw_query,
        answer=result.answer,
        timestamp_iso=ts_iso,
        oos=(result.route == "oos"),
    )

st.markdown("---")

# ------------------ SIDEBAR HISTORY (after potential upsert) ------------------
if st.session_state.get("chat_history_enabled", False) and st.session_state["show_chat_history"]:
    st.sidebar.title("📜 Chat History")
    hist_items = load_history_from_qdrant_anyshape(client, username, textbook, limit=10)
    if not hist_items:
        st.sidebar.info("No previous interactions found.")
    else:
        for item in hist_items:
            ts = item.get("timestamp","")
            st.sidebar.markdown(f"**Q:** {item.get('question','')}  \n🕒 {ts}")
            st.sidebar.markdown(f"**A ({item.get('option','')}):** {item.get('answer','')}")
            st.sidebar.markdown("---")

# ------------------ EXIT ------------------
if st.button("🚪 Exit", key="exit_session"):
    uploaded = 0
    if "session_log" in st.session_state:
        # Most items were already upserted live; this is a safety net (dedupe-friendly).
        uploaded = upload_session_to_user_memory(st.session_state["session_log"], memory_db)
        if uploaded:
            st.success(f"📤 Added {uploaded} Q&A item(s) to your personal memory.")
        else:
            st.info("No new Q&A to upload to personal memory.")
    st.session_state.clear()
    st.rerun()
