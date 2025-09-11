# auth.py
from __future__ import annotations
import re, datetime
from dataclasses import dataclass
from typing import Dict, Optional, Callable
import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferMemory

@dataclass(frozen=True)
class UserProfile:
    username: str
    assigned_subject: str
    experience_level: str   # 'Beginner'|'Intermediate'|'Advanced'
    voice: str
    chat_history_enabled: bool
    dynamic_level_key: str

def _slugify(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

@st.cache_data
def load_credentials(csv_path: str = "sample_credentials_with_levels.csv") -> Dict[str, Dict]:
    df = pd.read_csv(csv_path)
    creds, cols = {}, set(df.columns)
    for _, row in df.iterrows():
        username = str(row["username"]).strip()
        subject  = str(row["assigned_subject"]).strip()
        dyn_key  = f"{_slugify(subject)}_level"
        level    = str(row[dyn_key]).strip().title() if dyn_key in cols and pd.notna(row[dyn_key]) else "Intermediate"
        if dyn_key not in cols or pd.isna(row.get(dyn_key)):
            st.warning(f"Level column '{dyn_key}' missing for '{username}'. Defaulting to Intermediate.")
        creds[username] = {
            "password": str(row["password"]),
            "voice": str(row["voice"]),
            "assigned_subject": subject,
            "experience_level": level,
            "chat_history": str(row.get("chat_history","")).strip().lower() == "yes",
            "dynamic_level_key": dyn_key,
        }
    return creds

def authenticate(username: str, password: str, creds: Dict[str, Dict]) -> Optional[UserProfile]:
    rec = creds.get(username)
    if not rec or rec["password"] != password:
        return None
    return UserProfile(
        username=username,
        assigned_subject=rec["assigned_subject"],
        experience_level=rec["experience_level"],
        voice=rec["voice"],
        chat_history_enabled=rec["chat_history"],
        dynamic_level_key=rec["dynamic_level_key"],
    )

def _default_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history", input_key="question",
        return_messages=True, output_key="answer"
    )

def start_session(
    user: UserProfile,
    refresh_logs_fn: Optional[Callable[[], list]] = None,
    memory_factory: Optional[Callable[[], ConversationBufferMemory]] = None,
) -> None:
    st.session_state["authenticated"] = True
    st.cache_data.clear()
    if refresh_logs_fn:
        st.session_state["SESSION_LOGS"] = refresh_logs_fn()
    st.session_state["username"] = user.username
    st.session_state["voice"] = user.voice
    st.session_state["textbook"] = user.assigned_subject
    st.session_state["experience_level"] = user.experience_level
    st.session_state["chat_history_enabled"] = user.chat_history_enabled
    st.session_state.buffer_memory = (memory_factory or _default_memory)()
    st.session_state["session_log"] = {
        "username": user.username,
        "login_time": str(datetime.datetime.now()),
        "experience_level": user.experience_level,
        "textbook": user.assigned_subject,
        "interactions": [],
    }

def login_ui(creds: Dict[str, Dict], refresh_logs_fn=None, memory_factory=None) -> None:
    st.title("🔐 Login")
    u = st.text_input("Username"); p = st.text_input("Password", type="password")
    if st.button("Login"):
        user = authenticate(u, p, creds)
        if user:
            start_session(user, refresh_logs_fn, memory_factory)
            st.success(f"✅ Welcome, {user.username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

def require_auth(creds: Dict[str, Dict], refresh_logs_fn=None, memory_factory=None) -> None:
    if not st.session_state.get("authenticated"):
        login_ui(creds, refresh_logs_fn, memory_factory)
        st.stop()
