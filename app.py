import os, datetime
from dotenv import load_dotenv
import streamlit as st

from auth import load_credentials, require_auth
from retrieval import build_clients, get_textbook_store, get_user_memory_store, COLLECTION_MAP, ping_collection
import history, qa
from memory_store import upload_session_to_user_memory
from voice_speech import synthesize, play_in_streamlit

# ENV
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Auth
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS, refresh_logs_fn=history.load_existing_logs)

# Stores
textbook = st.session_state.get("textbook")
username = st.session_state.get("username")
level    = st.session_state.get("experience_level", "Intermediate")

embeddings, client = build_clients(OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
collection = COLLECTION_MAP.get(textbook)
if not collection:
    st.error(f"❌ No collection configured for textbook: {textbook}"); st.stop()

db = get_textbook_store(client, collection, embeddings)
memory_db = get_user_memory_store(client, embeddings)
st.caption(ping_collection(client, collection))

# UI (inline)
if "show_chat_history" not in st.session_state:
    st.session_state["show_chat_history"] = False
if st.session_state.get("chat_history_enabled", False):
    if st.button("📜 Show/Hide Chat History", key="toggle_chat_history"):
        st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

logs = st.session_state.get("SESSION_LOGS", [])
if st.session_state["show_chat_history"]:
    st.sidebar.title("📜 Chat History")
    for item in reversed(history.get_user_chat_history(logs, username, textbook)[-10:]):
        st.sidebar.markdown(f"**Q:** {item.get('question','')}  \n🕒 {item.get('timestamp','')}")
        st.sidebar.markdown(f"**A ({item.get('option','')}):** {item.get('answer','')}")
        st.sidebar.markdown('---')

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
        logs=logs,
        openai_api_key=OPENAI_API_KEY,
    )

    # Route hint
    if result.route == "reuse":
        st.info("🔁 Reused answer from previous session.")
    elif result.route == "memory":
        st.caption("🧠 Using your personal memory context.")
    else:
        st.caption("📚 Using textbook context.")

    # Render: hide text if Voice Answer was chosen
    if is_voice:
        st.markdown("### 🔊 Voice Answer")
    else:
        ui.render_answer(result.answer)

    # Sources still shown (keep this if you want the context expander even in voice mode)
    ui.render_sources(result.sources)

    # Voice playback
    if is_voice:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"🎙️ Generating voice with '{voice_choice}'..."):
            audio_bytes = synthesize(result.answer, voice_choice)
            play_in_streamlit(audio_bytes)

    # Log interaction (unchanged)
    st.session_state.setdefault("session_log", {"interactions": []})
    st.session_state["session_log"]["interactions"].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "experience_level": level,
        "question": raw_query,
        "option": option,
        "answer": result.answer,                 # keep logging full text for memory/upload
        "context": result.context_snippet,
        "textbook": textbook,
        "score": None,
    })

    st.session_state.setdefault("session_log", {"interactions": []})
    st.session_state["session_log"]["interactions"].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "experience_level": level, "question": raw_query, "option": option,
        "answer": result.answer, "context": result.context_snippet, "textbook": textbook, "score": None,
    })

st.markdown("---")
if st.button("🚪 Exit", key="exit_session"):
    uploaded = 0
    if "session_log" in st.session_state:
        uploaded = upload_session_to_user_memory(st.session_state["session_log"], memory_db)
        try:
            ok = history.append_log_to_github(st.session_state["session_log"])
            if ok: st.success(f"📤 Session log uploaded. (Memory docs: {uploaded})")
            else:  st.warning("⚠️ Failed to upload session log.")
        except Exception as e:
            st.warning(f"⚠️ GitHub upload error: {e}")
    st.session_state.clear(); st.rerun()
