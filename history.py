# history.py
from __future__ import annotations
import os, json, base64, requests
import streamlit as st
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

@st.cache_data
def load_existing_logs() -> list:
    token = os.getenv("GITHUB_TOKEN"); repo = os.getenv("GITHUB_REPO"); path = os.getenv("GITHUB_FILE_PATH")
    if not (token and repo and path): return []
    h = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=h)
    if r.status_code == 200:
        content = r.json()
        return json.loads(base64.b64decode(content["content"]).decode())
    return []

def append_log_to_github(log_entry: dict) -> bool:
    token = os.getenv("GITHUB_TOKEN"); repo = os.getenv("GITHUB_REPO"); path = os.getenv("GITHUB_FILE_PATH")
    if not (token and repo and path):
        st.error("❌ Missing GitHub env vars."); return False
    h = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/contents/{path}"

    r = requests.get(url, headers=h)
    if r.status_code == 200:
        content = r.json(); existing = json.loads(base64.b64decode(content["content"]).decode()); sha = content["sha"]
    elif r.status_code == 404:
        existing, sha = [], None
    else:
        st.error(f"❌ GitHub error: {r.status_code} — {r.text}"); return False

    existing.append(log_entry)
    payload = {"message": f"Append log for {log_entry.get('username')}",
               "content": base64.b64encode(json.dumps(existing, indent=2).encode()).decode(),
               "branch": "main"}
    if sha: payload["sha"] = sha
    pr = requests.put(url, headers=h, data=json.dumps(payload))
    return pr.status_code in (200, 201)

def get_user_chat_history(logs: list, username: str, textbook: str) -> list:
    hist = []
    for s in logs or []:
        if s.get("username")==username and s.get("textbook")==textbook:
            hist.extend(s.get("interactions", []))
    return hist
def get_user_chat_history_from_memory(
    client: QdrantClient,
    collection_name: str,
    username: str,
    textbook: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Read past Q&A from the user_memory collection (all sessions), newest first.
    Assumes each point has payload: username, textbook, timestamp, option, and page_content "Q: ...\\nA: ...".
    """
    # Pull a reasonable batch and sort by payload timestamp in Python
    flt = Filter(
        must=[
            FieldCondition(key="username", match=MatchValue(value=username)),
            FieldCondition(key="textbook", match=MatchValue(value=textbook)),
        ]
    )

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=flt,
        limit=512,
        with_payload=True,
        with_vectors=False,
    )

    def _parse_qa(content: str) -> tuple[str, str]:
        q, a = "", ""
        if content:
            parts = content.splitlines()
            for line in parts:
                if line.startswith("Q:"):
                    q = line[2:].strip()
                elif line.startswith("A:"):
                    a = line[2:].strip()
        return q, a

    items = []
    for p in points or []:
        pl = p.payload or {}
        q, a = _parse_qa(pl.get("content") or pl.get("text") or pl.get("page_content") or "")
        if not q or not a:
            # Try page_content if stored via LangChain Document
            q2, a2 = _parse_qa(pl.get("page_content", ""))
            q = q or q2
            a = a or a2
        items.append({
            "timestamp": pl.get("timestamp", ""),
            "option": pl.get("option", ""),
            "question": q,
            "answer": a,
            "oos": bool(pl.get("oos", False)),
        })

    # Sort by timestamp (ISO strings) and return newest first, limited
    items.sort(key=lambda x: x.get("timestamp",""), reverse=True)
    return items[:limit]
