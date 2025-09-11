# auth.py
from __future__ import annotations
import re
import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Callable

import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferMemory


# ---------- Public data model ----------
@dataclass(frozen=True)
class UserProfile:
    username: str
    assigned_subject: str
    experience_level: str     # 'Beginner' | 'Intermediate' | 'Advanced'
    voice: str
    chat_history_enabled: bool
    dynamic_level_key: str    # e.g., 'introductory_macroeconomics_level'


# ---------- Helpers ----------
def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ---------- Credentials loader ----------
@st.cache_data
def load_credentials(csv_path: str = "sample_credentials_with_levels.csv") -> Dict[str, Dict[str, str]]:
    """
    Load users from CSV and compute the dynamic `<subject>_level` column per user.
    Returns: { username: {password, voice, assigned_subject, experience_level, chat_history, dynamic_level_key} }
    """
    df = pd.read_csv(csv_path)
    creds: Dict[str, Dict[str, str]] = {}
    cols = set(df.columns)

    for _, row in df.iterrows():
        username = str(row["username"]).strip()
        assigned_subject = str(row["assigned_subject"]).strip()
        dyn_key = f"{_slugify(assigned_subject)}_level"

        if dyn_key in cols and pd.notna(row[dyn_key]):
            level_val = str(row[dyn_key]).strip().title()
        else:
            level_val = "Intermediate"
            st.warning(
                f"Level column '{dyn_key}' not found for user '{username}'. Defaulting to 'Intermediate'."
            )

        chat_hist = str(row.get("chat_history", "")).strip().lower() == "yes"

        creds[username] = {
            "password": str(row["password"]),
            "voice": str(row["voice"]),
            "assigned_subject": assigned_subject,
            "experience_level": level_val,
            "chat_history": chat_hist,
            "dynamic_level_key": dyn_key,
        }

    return creds


# ---------- Auth core ----------
def authenticate(username: str, password: str, creds: Dict[str, Dict[str, str]]) -> Optional[UserProfile]:
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


# ---------- Session bootstrap / teardown ----------
def _default_memory_factory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
        output_key="answer",
    )


def start_session(
    user: UserProfile,
    refresh_logs_fn: Optional[Callable[[], list]] = None,
    memory_factory: Optional[Callable[[], ConversationBufferMemory]] = None,
) -> None:
    """
    Idempotently populate st.session_state with everything the rest of the app expects.
    - refresh_logs_fn: your existing load_existing_logs (optional)
    - memory_factory: builds a ConversationBufferMemory (optional)
    """
    st.session_state["authenticated"] = True

    # Clear cached data (e.g., session logs) the way your current code does
    st.cache_data.clear()

    if refresh_logs_fn:
        st.session_state["SESSION_LOGS"] = refresh_logs_fn()

    st.session_state["username"] = user.username
    st.session_state["voice"] = user.voice
    st.session_state["textbook"] = user.assigned_subject
    st.session_state["experience_level"] = user.experience_level
    st.session_state["chat_history_enabled"] = user.chat_history_enabled

    st.session_state.buffer_memory = (
        memory_factory() if memory_factory else _default_memory_factory()
    )

    st.session_state["session_log"] = {
        "username": user.username,
        "login_time": str(datetime.datetime.now()),
        "experience_level": user.experience_level,
        "textbook": user.assigned_subject,
        "interactions": [],
    }


def logout() -> None:
    for k in [
        "authenticated",
        "username",
        "voice",
        "textbook",
        "experience_level",
        "chat_history_enabled",
        "buffer_memory",
        "session_log",
        "SESSION_LOGS",
    ]:
        if k in st.session_state:
            del st.session_state[k]


# ---------- UI ----------
def login_ui(
    creds: Dict[str, Dict[str, str]],
    refresh_logs_fn: Optional[Callable[[], list]] = None,
    memory_factory: Optional[Callable[[], ConversationBufferMemory]] = None,
) -> None:
    """
    Minimal, self-contained login form. On success, sets session_state and reruns.
    """
    st.title("🔐 Login")
    st.write("Enter your username and password")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(username, password, creds)
        if user:
            start_session(user, refresh_logs_fn=refresh_logs_fn, memory_factory=memory_factory)
            st.success(f"✅ Login successful. Welcome, {user.username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")


def require_auth(
    creds: Dict[str, Dict[str, str]],
    refresh_logs_fn: Optional[Callable[[], list]] = None,
    memory_factory: Optional[Callable[[], ConversationBufferMemory]] = None,
) -> None:
    """
    Gatekeeper for any page. If not authenticated, render login and stop execution.
    """
    if not st.session_state.get("authenticated"):
        login_ui(creds, refresh_logs_fn=refresh_logs_fn, memory_factory=memory_factory)
        st.stop()
