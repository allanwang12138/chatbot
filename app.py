# app.py (vector-only; no GitHub logs/TF-IDF)
import os, datetime
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

# ENV
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL     = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # may be None/empty for local/public Qdrant
openai.api_key = OPENAI_API_KEY  # for voice_speech

# Auth (no GitHub logs refresh)
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS)  # 👈 no refresh_logs_fn

# Stores
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

# ---- UI ----
st.title(f"📄 {textbook} Q&A App")
raw_query = st.text_input(f"Ask a question about {textbook}:")
col1, col2, col3 = st.columns(3)
option = None
with col1:
    if st.button("📖 Detailed Answer"):
        option = "Detailed Answer"
with col2:
    if st.button("✂️ Concise Answer"):
        option = "Concise Answer"
with col3:
    if st.button("🔊 Voice Answer"):
        option = "Concise Answer + Voice"

# ---- Answer flow (Qdrant-only) ----
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
        logs=[],                      # 👈 no JSON logs; ignored by qa.answer
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

    # Log interaction locally (for memory upload)
    st.session_state.setdefault("session_log", {"interactions": []})
    st.session_state["session_log"]["interactions"].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "experience_level": level,
        "question": raw_query,
        "option": option,
        "answer": result.answer,
        "context": result.context_snippet,
        "textbook": textbook,
        "score": None,
    })

st.markdown("---")

# ---- Exit: upload Q&A pairs to user_memory only (no GitHub) ----
if st.button("🚪 Exit", key="exit_session"):
    uploaded = 0
    if "session_log" in st.session_state:
        uploaded = upload_session_to_user_memory(st.session_state["session_log"], memory_db)
        if uploaded:
            st.success(f"📤 Added {uploaded} Q&A item(s) to your personal memory.")
        else:
            st.info("No new Q&A to upload to personal memory.")
    st.session_state.clear()
    st.rerun()
