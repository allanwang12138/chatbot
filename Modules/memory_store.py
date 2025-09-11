# memory_store.py
from __future__ import annotations
from typing import Dict, Any
from langchain.schema import Document
import streamlit as st

def upload_session_to_user_memory(session_log: Dict[str, Any], memory_db) -> int:
    """Return count uploaded. Expects memory_db to be a LangChain Qdrant store."""
    interactions = (session_log or {}).get("interactions", []) or []
    if not interactions:
        return 0
    if st.session_state.get("uploaded_to_memory"):
        return 0

    docs = []
    for entry in interactions:
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        if not (q and a): 
            continue
        content = f"Q: {q}\nA: {a}"
        if len(content) > 3000:
            content = content[:3000] + " …"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "username": session_log.get("username"),
                    "textbook": session_log.get("textbook"),
                    "experience_level": entry.get("experience_level"),
                    "option": entry.get("option"),
                    "timestamp": entry.get("timestamp"),
                    "source": "user_memory",
                },
            )
        )
    try:
        if docs:
            memory_db.add_documents(docs)
            st.session_state["uploaded_to_memory"] = True
        return len(docs)
    except Exception as e:
        st.warning(f"⚠️ Failed to write to user_memory: {e}")
        return 0
