# RAG Tutor App (Streamlit)

A modular, Streamlit-based RAG (Retrieval-Augmented Generation) tutor that answers questions from textbook embeddings and personal learning memory. The app supports login-based profiles, GitHub JSON logging, and optional text-to-speech (TTS) voice playback.

---

## Project Structure (final)

```
project/
├── app.py                # Streamlit entrypoint (thin coordinator + simple UI)
├── auth.py               # Authentication + session bootstrap
├── retrieval.py          # Qdrant clients, vector stores, retriever routing
├── qa.py                 # Q&A orchestration (TF‑IDF reuse → retrieval → LLM → optional compress)  [prompts inlined]
├── history.py            # GitHub JSON logs (load + append) + history helpers
├── memory_store.py       # Upload session Q&A pairs to 'user_memory' collection
├── voice_speech.py       # Text → Speech synthesis & Streamlit playback
└── README.md             # This file
```

> **Note:** `config.py`, `models.py`, and `ui.py` have been removed. Prompt templates are now **inlined** in `qa.py`. `COLLECTION_MAP` lives in `retrieval.py`.

---

## Quickstart

### 1) Environment
- Python 3.10+ recommended
- Create a virtual environment and install packages:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> If you don't have a requirements file yet, see the **Suggested requirements** at the end of this README.

### 2) .env
Create a `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333            # or your Qdrant endpoint
QDRANT_API_KEY=                             # optional if Qdrant is public/local
GITHUB_TOKEN=ghp_...                        # required for logging to GitHub
GITHUB_REPO=username/repo                   # e.g., "yourname/learning-logs"
GITHUB_FILE_PATH=logs/session_logs.json     # path within the repo
```

### 3) Credentials CSV
Place a `sample_credentials_with_levels.csv` with columns like:

```
username,password,voice,assigned_subject,chat_history,introductory_macroeconomics_level,introductory_microeconomics_level,statistics_for_economics_level
alice,pass123,alloy,Introductory Macroeconomics,yes,Beginner,Intermediate,Advanced
```

- `assigned_subject` must match a key in `retrieval.COLLECTION_MAP`.
- The app computes a dynamic level column name: `<slug(subject)>_level`. If missing, it defaults to **Intermediate**.

### 4) Run
```bash
streamlit run app.py
```

Log in with a username + password from the CSV.

---

## How it works

### Authentication (`auth.py`)
- Loads credentials from CSV (`load_credentials`).
- Authenticates users and sets `st.session_state` with profile fields:
  - `username`, `voice`, `textbook` (assigned subject), `experience_level`, `chat_history_enabled`
- Creates a `ConversationBufferMemory` per session.
- Provides `require_auth(...)` gate and a minimal login UI.

### Retrieval (`retrieval.py`)
- `build_clients(...)` creates cached OpenAI embeddings + Qdrant client.
- `COLLECTION_MAP` maps human textbook names → Qdrant collection names.
- `get_textbook_store(...)`, `get_user_memory_store(...)` return LangChain Qdrant stores.
- `make_retrievers(...)` builds:
  1. **User memory** retriever with strict payload filter (`username`, `textbook`, `experience_level`).
  2. **Textbook** retriever (prefers payload filter `{"textbook": <name>}`; falls back unfiltered).
  - Probes memory for a quick route decision (memory-first; otherwise textbook).

### Q&A Orchestration (`qa.py`)
1. **Reuse**: TF‑IDF similarity over prior GitHub logs. If a close match exists, reuse the answer (optionally compress for concise).
2. **Route**: build memory + textbook retrievers and choose route based on memory hit.
3. **Prompt**: select a template **inlined in `qa.py`** based on level (`Beginner/Intermediate/Advanced`) and answer type (`Concise/Detailed`).
4. **LLM**: run a `ConversationalRetrievalChain` with memory and the chosen retriever.
5. **Compress**: if `Concise`, rewrite the answer down to ≤ 2 sentences.
6. **Return**: `QAResult` (answer text, sources, a short context snippet, and route).

### Logs (`history.py`)
- `load_existing_logs()` fetches and caches JSON at `GITHUB_REPO/GITHUB_FILE_PATH`.
- `append_log_to_github(log_entry)` appends the session JSON.
- `get_user_chat_history(logs, username, textbook)` extracts prior Q&A for sidebar display.

### Memory Upload (`memory_store.py`)
- On Exit, converts session Q&A pairs to LangChain `Document`s and writes them into the **`user_memory`** collection with payload fields used in filtering.

### Voice Speech (`voice_speech.py`)
- `synthesize(text, voice)` uses OpenAI TTS (`tts-1`) to produce MP3 bytes.
- `play_in_streamlit(audio_bytes)` plays audio in Streamlit.
- In `app.py`, when the user selects **Voice**, the app synthesizes and plays the final answer.

---

## `app.py` Responsibilities

- Load environment variables via `dotenv`.
- `require_auth(...)` and then build Qdrant stores.
- Minimal UI: question input + buttons, optional chat-history panel.
- Call `qa.answer(...)` and display the result (text + sources).
- If voice selected: call `voice_speech.synthesize(...)` and play audio.
- On Exit: upload session Q&A to user memory and append JSON logs to GitHub.

---

## Adding a textbook

1. Upload/prepare the embedded collection in Qdrant.
2. Add a mapping in `retrieval.COLLECTION_MAP`:
   ```python
   COLLECTION_MAP["Your New Subject"] = "your_new_subject_collection"
   ```
3. Ensure the user’s `assigned_subject` matches `"Your New Subject"` exactly.

---

## Troubleshooting

- **Missing OPENAI_API_KEY / QDRANT_URL**: set them in `.env` and restart.
- **Qdrant 401/403**: set `QDRANT_API_KEY` or allow your IP / make sure the URL is correct.
- **GitHub 404**: ensure `GITHUB_TOKEN` has `repo` scope and that `GITHUB_REPO` and `GITHUB_FILE_PATH` exist (the code will create the file if missing).
- **No chat history**: `chat_history_enabled` must be `"yes"` in the CSV for that user.
- **Memory not used**: No hits yet or payload filters didn’t match; confirm the session actually uploaded to `user_memory` and that the filters align (`username`, `textbook`, `experience_level`).

---

## Suggested requirements (example)

```
streamlit>=1.34
python-dotenv>=1.0
openai>=1.40
qdrant-client>=1.8
langchain>=0.2
langchain-openai>=0.1
langchain-qdrant>=0.1
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
requests>=2.31
```

> Versions are indicative; pin to the exact versions that work in your environment.

---

## Privacy & Notes

- Session transcripts are uploaded to your GitHub repo and to the Qdrant `user_memory` collection. Treat both as **persistent** storage.
- Streamlit caching (`@st.cache_data` / `@st.cache_resource`) is used for logs and heavy clients.
- The TTS feature calls OpenAI’s API and returns MP3 bytes.

---

## License

Choose a license that fits your use (MIT/Apache-2.0/etc.) and add it to the repository.
