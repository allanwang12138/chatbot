# history.py
from __future__ import annotations
import os, json, base64, requests
import streamlit as st

@st.cache_data
def load_existing_logs() -> list:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    path = os.getenv("GITHUB_FILE_PATH")
    if not (token and repo and path):
        return []

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    r = requests.get(api_url, headers=headers)
    if r.status_code == 200:
        content = r.json()
        return json.loads(base64.b64decode(content["content"]).decode())
    return []

def append_log_to_github(log_entry: dict) -> bool:
    token = os.getenv("GITHUB_TOKEN"); repo = os.getenv("GITHUB_REPO"); path = os.getenv("GITHUB_FILE_PATH")
    if not (token and repo and path):
        st.error("❌ Missing GitHub env vars.")
        return False

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    r = requests.get(api_url, headers=headers)
    if r.status_code == 200:
        content = r.json()
        existing_data = json.loads(base64.b64decode(content["content"]).decode())
        sha = content["sha"]
    elif r.status_code == 404:
        existing_data, sha = [], None
    else:
        st.error(f"❌ GitHub error: {r.status_code} — {r.text}")
        return False

    existing_data.append(log_entry)
    updated_content = base64.b64encode(json.dumps(existing_data, indent=2).encode()).decode()

    payload = {"message": f"Append log for {log_entry.get('username')}", "content": updated_content, "branch": "main"}
    if sha: payload["sha"] = sha

    pr = requests.put(api_url, headers=headers, data=json.dumps(payload))
    return pr.status_code in (200, 201)

def get_user_chat_history(logs: list, username: str, textbook: str) -> list:
    history = []
    for session in logs or []:
        if session.get("username") == username and session.get("textbook") == textbook:
            history.extend(session.get("interactions", []))
    return history
