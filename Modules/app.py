# app.py
import streamlit as st
from config import load_env, env, COLLECTION_MAP
from auth import load_credentials, require_auth
import history
import retrieval
import ui
import qa
from models import QAResult

# 0) env & creds
load_env()
CREDS = load_credentials("sample_credentials_with_levels.csv")
require_auth(CREDS, refresh_logs_fn=history.load_existing_logs)

# 1) resolve subject & build stores
textbook = st.session_state.get("textbook")
username = st.session_state.get("username")
level = st.session_state.get("experience_level", "Intermediate")

embeddings, client = retrieval.build_clients(env.OPENAI_API_KEY, env.QDRANT_URL, env.QDRANT_API_KEY)
collection = COLLECTION_MAP.get(textbook)
db = retrieval.get_textbook_store(client, collection, embeddings)
memory_db = retrieval.get_user_memory_store(client, embeddings)

# 2) UI
ui.render_history_toggle()
if st.session_state.get("show_chat_history"):
    hist = history.get_user_chat_history(st.session_state.get("SESSION_LOGS", []), username, textbook)
    ui.render_history_sidebar(hist)

raw_query, option = ui.render_controls(textbook)
if raw_query and option:
    result: QAResult = qa.answer(raw_query, option, st.session_state, (db, memory_db), st.session_state.get("SESSION_LOGS", []))
    ui.render_answer(result.answer)
    ui.render_sources(result.sources)

# 3) Exit button
ui.render_exit_button(db=memory_db)  # internally calls memory_store + history and clears session
