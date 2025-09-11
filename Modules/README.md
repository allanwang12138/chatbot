# RAG Tutor App

## Quickstart
1. Create .env (see .env.example)
2. pip install -r requirements.txt
3. Prepare credentials CSV (sample_credentials_with_levels.csv)
4. Start: streamlit run app.py

## How it works
- Authentication: auth.py
- Retrieval (Qdrant + user_memory): retrieval.py
- Q&A orchestration: qa.py (TF-IDF reuse → retrieval route → LLM → optional compress)
- Logs: history.py (GitHub JSON)
- Memory upload: memory_store.py (stores Q&A pairs to user_memory)
- Prompts: prompts.py

## Environment variables
OPENAI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
GITHUB_TOKEN=...
GITHUB_REPO=username/repo
GITHUB_FILE_PATH=logs/session_logs.json

## Collections
See config.COLLECTION_MAP
