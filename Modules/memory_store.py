# memory_store.py
from __future__ import annotations
from typing import Dict, Any
from langchain.schema import Document
import streamlit as st

def upload_session_to_user_memory(session_log: Dict[str, Any], memory_db) -> int:
    interactions = (session_log or {}).get("interactions", []) or []
    if not interactions: return 0
    if st.session_state.get("uploaded_to_memory"): return 0

    docs = []
    for e in interactions:
        q = (e.get("question") or "").strip()
        a = (e.get("answer") or "").strip()
        if not (q and a): continue
        content = (f"Q: {q}\nA: {a}")[:3000] + (" …" if len(f"Q: {q}\nA: {a}")>3000 else "")
        docs.append(Document(page_content=content, metadata={
            "username": session_log.get("username"),
            "textbook": session_log.get("textbook"),
            "experience_level": e.get("experience_level"),
            "option": e.get("option"),
            "timestamp": e.get("timestamp"),
            "source": "user_memory",
        }))
    try:
        if docs:
            memory_db.add_documents(docs)
            st.session_state["uploaded_to_memory"] = True
        return len(docs)
    except Exception as ex:
        st.warning(f"⚠️ Failed to write user_memory: {ex}")
        return 0
