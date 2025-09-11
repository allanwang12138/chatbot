# ui.py
from __future__ import annotations
import datetime
import streamlit as st

def render_history_toggle():
    if "show_chat_history" not in st.session_state:
        st.session_state["show_chat_history"] = False
    if st.session_state.get("chat_history_enabled", False):
        if st.button("📜 Show/Hide Chat History", key="toggle_chat_history"):
            st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

def render_history_sidebar(history: list):
    st.sidebar.title("📜 Chat History")
    if not history:
        st.sidebar.info("No previous interactions found.")
        return
    for item in reversed(history[-10:]):
        ts = item.get("timestamp")
        try:
            dt = datetime.datetime.fromisoformat(ts)
            tstr = dt.strftime("%m/%d/%Y %H:%M")
        except Exception:
            tstr = ts or ""
        st.sidebar.markdown(f"**Q:** {item.get('question','')}  \n🕒 {tstr}")
        st.sidebar.markdown(f"**A ({item.get('option','')}):** {item.get('answer','')}")
        st.sidebar.markdown("---")

def render_controls(textbook: str) -> tuple[str, str | None]:
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
    return raw_query, option

def render_answer(text: str):
    st.markdown("### 📘 Answer")
    st.write(text)

def render_sources(source_docs: list):
    with st.expander("📚 Show Supporting Context"):
        if not source_docs:
            st.write("No supporting documents returned.")
            return
        top = source_docs[0]
        meta = getattr(top, "metadata", {}) or {}
        origin = meta.get("source", "unknown")
        st.markdown(f"**Source:** `{origin}` — **Textbook:** {meta.get('textbook','N/A')}")
        snippet = (top.page_content or "").strip()
        st.write(snippet[:1200] + ("..." if len(snippet) > 1200 else ""))

def render_exit_button(on_click):
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        custom_css = """
        <style>
        div.stButton > button:first-child {
            font-size: 18px; padding: 0.6em 2em; width: 100%;
            border-radius: 10px; background-color: #ffb47a; color: white;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        if st.button("🚪 Exit", key="exit_session"):
            on_click()
