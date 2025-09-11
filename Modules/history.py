# history.py
from __future__ import annotations
import os, json, base64, requests
import streamlit as st

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
