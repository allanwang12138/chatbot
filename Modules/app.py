# app.py
import datetime
import streamlit as st

from config import load_env, COLLECTION_MAP
from auth import load_credentials, require_auth
import history
import retrieval
import ui
import qa
from models import QAResult
from voice_speech import synthesize, play_in_streamlit
from memory_store import upload_session_to_user_memory
from retrieval import ping_collection  # optional caption

# 0) env & creds
env = load_env()  # <-- load once; we no longer import a global `env`
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS, refresh_logs_fn=history.load_existing_logs)

# 1) resolve subject & build stores
textbook = st.session_state.get("textbook")
username = st.session_state.get("username")
level = st.session_state.get("experience_level", "Intermediate")

embeddings, client = retrieval.build_clients(
    env.OPENAI_API_KEY,
    env.QDRANT_URL,
    env.QDRANT_API_KEY,
)

collection = COLLECTION_MAP.get(textbook)
if not collection:
    st.error(f"❌ No collection configured for textbook: {textbook}")
    st.stop()

db = retrieval.get_textbook_store(client, collection, embeddings)
memory_db = retrieval.get_user_memory_store(client, embeddings)

# (optional) Caption with collection status
st.caption(ping_collection(client, collection))

# 2) UI
ui.render_history_toggle()
if st.session_state.get("show_chat_history"):
    logs = st.session_state.get("SESSION_LOGS", [])
    hist = history.get_user_chat_history(logs, username, textbook)
    ui.render_history_sidebar(hist)
else:
    logs = st.session_state.get("SESSION_LOGS", [])

raw_query, option = ui.render_controls(textbook)

if raw_query and option:
    # Orchestrate Q&A (prompts are selected inside qa.answer via prompts.get_prompt)
    result: QAResult = qa.answer(
        raw_question=raw_query,
        option=option,
        username=username,
        level=level,
        textbook=textbook,
        memory=st.session_state.buffer_memory,
        db=db,
        memory_db=memory_db,
        logs=logs,
        openai_api_key=env.OPENAI_API_KEY,
    )

    # Small route hint
    if result.route == "reuse":
        st.info("🔁 Reused answer from previous session.")
    elif result.route == "memory":
        st.caption("🧠 Using your personal memory context.")
    else:
        st.caption("📚 Using textbook context.")

    # Show text answer + sources
    ui.render_answer(result.answer)
    ui.render_sources(result.sources)

    # Voice option
    if "Voice" in option:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"🎙️ Generating voice with '{voice_choice}'..."):
            audio_bytes = synthesize(result.answer, voice_choice)
            play_in_streamlit(audio_bytes)

    # Append interaction to session log (UTC)
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

# 3) Exit button (uses new on_click pattern)
def _on_exit():
    uploaded = 0
    if "session_log" in st.session_state:
        # 1) Embed to user_memory
        uploaded = upload_session_to_user_memory(st.session_state["session_log"], memory_db)
        # 2) Push JSON log to GitHub
        try:
            ok = history.append_log_to_github(st.session_state["session_log"])
            if ok:
                st.success(f"📤 Session log uploaded to GitHub. (Memory docs: {uploaded})")
            else:
                st.warning("⚠️ Failed to upload session log to GitHub.")
        except Exception as e:
            st.warning(f"⚠️ GitHub upload error: {e}")
    # Fresh start next login (buffer rebuilt on login)
    st.session_state.clear()
    st.rerun()

ui.render_exit_button(_on_exit)
