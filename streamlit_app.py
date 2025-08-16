import streamlit as st
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
import json
import base64
import requests
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import unicodedata
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage

# ------------------- Normalize Response -------------
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)  # handles full-width characters
    text = re.sub(r"[^\w\s]", "", text.lower()).strip()  # lowercasing + remove punctuation
    return text
     
# Initialize memory buffer per session (store in session_state)
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question", 
        return_messages=True,
        output_key="answer",
    )

# ------------------- Load secrets -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

openai.api_key = OPENAI_API_KEY

# ------------------- Load session logs from GitHub -------------------
@st.cache_data
def load_existing_logs():
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")  # e.g., "yourusername/yourrepo"
    path = os.getenv("GITHUB_FILE_PATH")  # e.g., "logs/session_logs.json"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return json.loads(base64.b64decode(content["content"]).decode())
    else:
        return []  # Return empty list if not found or failed

SESSION_LOGS = load_existing_logs()
# ------------------- Create a function to retrieve chat history -------------------
def get_user_chat_history(logs, username, textbook):
    """
    Filter full session logs to return only past Q&A entries 
    for the specified user and textbook.
    """
    history = []
    for session in logs:
        if session.get("username") == username and session.get("textbook") == textbook:
            history.extend(session.get("interactions", []))
    return history

# ------------------- Create a function to find similar question asked before -------------------
def find_similar_answer(logs, new_question, level, textbook, answer_type, threshold=0.75):
    def preprocess(text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    new_question_clean = preprocess(new_question)
    questions = []
    answers = []

    for session in logs:
        for entry in session.get("interactions", []):
            is_same_type = (
                ("Concise" in entry.get("option", "") and answer_type == "Concise") or
                ("Detailed" in entry.get("option", "") and answer_type == "Detailed")
            )
            if (
                entry.get("experience_level") == level and
                entry.get("textbook") == textbook and
                is_same_type
            ):
                questions.append(preprocess(entry["question"]))
                answers.append(entry["answer"])

    if not questions:
        return None

    vectorizer = TfidfVectorizer().fit(questions + [new_question_clean])
    vectors = vectorizer.transform(questions + [new_question_clean])
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]

    best_idx = similarities.argmax()
    if similarities[best_idx] >= threshold:
        return answers[best_idx]
    return None



# ------------------- Create a function to store log -------------------
def append_log_to_github(log_entry):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")  # e.g., "yourusername/yourrepo"
    path = os.getenv("GITHUB_FILE_PATH")  # e.g., "logs/session_logs.json"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    # Try to get existing file
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        existing_data = json.loads(base64.b64decode(content["content"]).decode())
        sha = content["sha"]
    elif response.status_code == 404:
        existing_data = []
        sha = None
    else:
        st.error(f"❌ GitHub error: {response.status_code} — {response.text}")
        return False

    # Append the new log
    existing_data.append(log_entry)
    updated_content = base64.b64encode(json.dumps(existing_data, indent=2).encode()).decode()

    payload = {
        "message": f"Append log for {log_entry['username']}",
        "content": updated_content,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha  # include only when updating

    put_response = requests.put(api_url, headers=headers, data=json.dumps(payload))
    return put_response.status_code in [200, 201]

# ------------------- Load credentials (dynamic {subject}_level) -------------------
def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

@st.cache_data
def load_credentials():
    df = pd.read_csv("sample_credentials_with_levels.csv")

    creds = {}
    cols = set(df.columns)  # faster membership check
    for _, row in df.iterrows():
        username = str(row["username"]).strip()
        assigned_subject = str(row["assigned_subject"]).strip()

        dyn_key = f"{_slugify(assigned_subject)}_level"

        # Robust key check + normalization
        if dyn_key in cols and pd.notna(row[dyn_key]):
            level_val = str(row[dyn_key]).strip().title()  # -> 'Beginner'/'Intermediate'/'Advanced'
        else:
            level_val = "Intermediate"  # safe default
            st.warning(f"Level column '{dyn_key}' not found for user '{username}'. Defaulting to 'Intermediate'.")

        chat_hist = str(row.get("chat_history", "")).strip().lower() == "yes"

        creds[username] = {
            "password": str(row["password"]),
            "voice": str(row["voice"]),
            "assigned_subject": assigned_subject,
            "experience_level": level_val,   # <-- use this later at login
            "chat_history": chat_hist,
            "dynamic_level_key": dyn_key,    # optional for debugging
        }

    return creds

CREDENTIALS = load_credentials()

def login():
    st.title("🔐 Login")
    st.write("Enter your username and password")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = CREDENTIALS.get(username)
        if user and user["password"] == password:
            assigned_subject = user["assigned_subject"]
            level = user["experience_level"]  # <-- use exactly what we loaded

            st.session_state["authenticated"] = True
            global SESSION_LOGS
            st.cache_data.clear()
            SESSION_LOGS = load_existing_logs()

            st.session_state["username"] = username
            st.session_state["voice"] = user["voice"]
            st.session_state["textbook"] = assigned_subject
            st.session_state["experience_level"] = level              # <-- set here
            st.session_state["chat_history_enabled"] = user["chat_history"]

            st.session_state.buffer_memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question", 
                return_messages=True,
                output_key="answer", 
            )

            st.session_state["session_log"] = {
                "username": username,
                "login_time": str(datetime.datetime.now()),
                "experience_level": level,                            # <-- and here
                "textbook": assigned_subject,
                "interactions": []
            }

            st.success(f"✅ Login successful. Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")



# ------------------- Authentication Gate -------------------
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

# ------------------- Qdrant Setup -------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Map human textbook names -> Qdrant collection names
COLLECTION_MAP = {
    "Introductory Macroeconomics": "introductory_macroeconomics_collection",
    "Introductory Microeconomics": "introductory_microeconomics_collection",
    "Statistics For Economics": "statistics_for_economics_collection",
    "MATHEMATICS Textbook for Class IX": "mathematics_textbook_for_class_ix_collection",
    "MATHEMATICS Textbook for Class X": "mathematics_textbook_for_class_x_collection",
    "MATHEMATICS Textbook for Class XI": "mathematics_textbook_for_class_xi_collection",
    "MATHEMATICS Textbook for Class XII PART I": "mathematics_textbook_for_class_xii_part_i_collection",
    "MATHEMATICS Textbook for Class XII PART II": "mathematics_textbook_for_class_xii_part_ii_collection",
}

selected_textbook = st.session_state.get("textbook", "Introductory Macroeconomics")
COLLECTION_NAME = COLLECTION_MAP.get(selected_textbook)

if not COLLECTION_NAME:
    st.error(f"❌ No collection configured for textbook: {selected_textbook}")
    st.stop()

# Textbook vector store
db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

# User-specific memory vector store (shared across all subjects)
USER_MEMORY_COLLECTION = "user_memory"
memory_db = Qdrant(client=client, collection_name=USER_MEMORY_COLLECTION, embeddings=embeddings)

# (Optional) small debug
try:
    info = client.get_collection(COLLECTION_NAME)
    st.caption(f"🗂️ Using collection **{COLLECTION_NAME}** · status: Active")
except Exception:
    st.caption(f"🗂️ Using collection **{COLLECTION_NAME}**")


# ------------------- Prompt Templates -------------------
PROMPT_BEGINNER_DETAILED = """
You are a patient and friendly tutor helping someone completely new to the subject {textbook}.

Based only on the context below, explain the answer clearly in no more than 20 sentences. Avoid jargon and technical terms. Use simple language and real-world analogies (like shopping, school, or weather) to help the student understand.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_BEGINNER_CONCISE = """
You are helping a beginner understand subject {textbook}. Give a short, friendly answer using very simple words — no jargon or equations and no more than 4 sentences.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_INTERMEDIATE_DETAILED = """
You are an experienced tutor helping a student with some background in subject {textbook}.

Using only the context below, provide a clear and informative answer in no more than 20 sentences. Use standard terms and concepts, but keep explanations digestible and well-structured. Include examples if helpful.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_INTERMEDIATE_CONCISE = """
You are a tutor providing a concise but clear explanation to a student with intermediate knowledge in subject {textbook}.

Using the context below, answer the question in no more than 4 sentences. Focus on clarity, not detail.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_ADVANCED_DETAILED = """
You are an expert academic tutor working with an advanced student who understands subkect {textbook} theory.

Using only the context provided, write a focused and rigorous answer in no more than 20 sentences. Feel free to include more complex concepts such as math if relevant.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_ADVANCED_CONCISE = """
You're providing a concise response to a student with advanced knowledge in {textbook}.

Using the context below, answer the question in no more than 4 sentences, assuming the reader is familiar with key terms and theories.

Context:
{context}

Question:
{question}

Answer:
"""



# ------------------- Streamlit UI -------------------
selected_textbook = st.session_state.get("textbook", "Textbook")

# Initialize toggle state
if "show_chat_history" not in st.session_state:
    st.session_state["show_chat_history"] = False

# Only show button if user has permission
if st.session_state.get("chat_history_enabled", False):
    if st.button("📜 Show/Hide Chat History", key="toggle_chat_history"):
        st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

# Format dynamic title and input prompt
st.title(f"📄 {selected_textbook} Q&A App")

# ------------------- Chat history showed when clicked the button -------------------
if st.session_state["show_chat_history"]:
    st.sidebar.title("📜 Chat History")

    history = get_user_chat_history(
        SESSION_LOGS,
        st.session_state.get("username"),
        st.session_state.get("textbook")
    )

    if not history:
        st.sidebar.info("No previous interactions found.")
    else:
        for item in reversed(history[-10:]):  # show more recent 10 questions
            timestamp = item['timestamp']
            dt = datetime.datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%m/%d/%Y %H:%M")
            
            st.sidebar.markdown(f"**Q:** {item['question']}  \n🕒 {formatted_time}")
            st.sidebar.markdown(f"**A ({item['option']}):** {item['answer']}")
            st.sidebar.markdown("---")


raw_query = st.text_input(f"Ask a question about {selected_textbook}:")
query = normalize_text(raw_query)



col1, col2, col3 = st.columns(3)
with col1:
    detailed_clicked = st.button("📖 Detailed Answer")
with col2:
    concise_clicked = st.button("✂️ Concise Answer")
with col3:
    voice_clicked = st.button("🔊 Voice Answer")

option = None
if detailed_clicked:
    option = "Detailed Answer"
elif concise_clicked:
    option = "Concise Answer"
elif voice_clicked:
    option = "Concise Answer + Voice"

# ------------------- MAIN Q&A BLOCK -------------------


def _to_text(x):
    """Coerce LangChain outputs (dict/AIMessage/str) into a clean string."""
    if isinstance(x, BaseMessage):
        return x.content or ""
    if isinstance(x, dict):
        return x.get("answer") or x.get("result") or x.get("output_text") or str(x)
    return str(x)

if query and option:
    level = st.session_state.get("experience_level", "Intermediate")
    selected_textbook = st.session_state.get("textbook", "Introductory Macroeconomics")
    username = st.session_state.get("username")

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    with st.spinner("💬 Generating answer..."):
        answer_type = "Concise" if ("Concise" in option or "Voice" in option) else "Detailed"

        # 1) Try reuse from logs (TF-IDF)
        existing_answer = find_similar_answer(
            SESSION_LOGS, query, level, selected_textbook, answer_type
        )

        source_docs = []
        if existing_answer:
            response_text = existing_answer
            st.info("🔁 Reused answer from previous session.")


        else:
            # 2) Build two retrievers: user_memory first, then textbook
            # --- memory retriever (on user_memory), STRICT filter ---
            user_filter = {
                "username": username,
                "textbook": selected_textbook,
                "experience_level": level,
                # "source": "user_memory",  # uncomment if you set this when uploading
            }
            memory_retriever = memory_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "filter": user_filter}
            )

            # Probe user_memory. If filter/index is missing or no hits, route to textbook.
            has_memory_match = False
            try:
                memory_hits = memory_db.similarity_search_with_score(query, k=1, filter=user_filter)
                has_memory_match = len(memory_hits) > 0
            except Exception as e:
                has_memory_match = False
                st.info("ℹ️ No personal memory available yet. Using textbook context.")
                # Optional: print debug
                # st.caption(f"[debug] memory probe exception: {e}")

            # --- textbook retriever (on subject collection) ---
            # Try filtered by textbook; if the collection has zero points with that payload yet,
            # gracefully retry without a filter.
            try:
                _ = db.similarity_search_with_score(query, k=1, filter={"textbook": selected_textbook})
                textbook_retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": {"textbook": selected_textbook}},
                )
            except Exception:
                st.info("ℹ️ Textbook filter not ready; falling back to unfiltered textbook search.")
                textbook_retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5},
                )

            # 3) Route: prefer user memory if there’s any hit; otherwise textbook
            chosen_retriever = memory_retriever if has_memory_match else textbook_retriever
            # --- choose prompt by level & answer type ---
            if answer_type == "Concise":
                if level == "Beginner":
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_BEGINNER_CONCISE)
                elif level == "Advanced":
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_ADVANCED_CONCISE)
                else:
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_INTERMEDIATE_CONCISE)
            else:  # Detailed
                if level == "Beginner":
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_BEGINNER_DETAILED)
                elif level == "Advanced":
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_ADVANCED_DETAILED)
                else:
                    selected_prompt = ChatPromptTemplate.from_template(PROMPT_INTERMEDIATE_DETAILED)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=model,
                retriever=chosen_retriever,
                memory=st.session_state.buffer_memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": selected_prompt}
            )

            # Use invoke to get sources for UI
            result = qa_chain.invoke({"question": query})
            response_text = result["answer"]
            source_docs = result.get("source_documents", []) or []
            if isinstance(result, dict) and "source_documents" in result:
                source_docs = result["source_documents"] or []

            # Keep buffer memory updated
            st.session_state.buffer_memory.chat_memory.add_user_message(raw_query)
            st.session_state.buffer_memory.chat_memory.add_ai_message(response_text)
        # Pick the most relevant context (first source document, if any)
        top_context = ""
        if source_docs:
            top_doc = source_docs[0]
            # keep context reasonable for logs
            top_context = (top_doc.page_content or "").strip()
            if len(top_context) > 1200:
                top_context = top_context[:1200] + "..."
        # 4) Post-process for Concise/Voice option
        if "Concise" in option or "Voice" in option:
            compress_prompt = (
                f"You are a tutor for {selected_textbook} with a(n) {level} learner. "
                f"Rewrite the answer below in no more than 2 sentences, clear and direct, "
                f"no equations or jargon.\n\nAnswer:\n{response_text}"
            )
            compressed = model.invoke(compress_prompt)
            response_text = _to_text(compressed)

        # 5) Log interaction (use UTC for consistency)
        st.session_state["session_log"]["interactions"].append({
            "timestamp": str(datetime.datetime.utcnow()),
            "experience_level": level,
            "question": raw_query,
            "option": option,
            "answer": response_text,
            "context": top_context,  # we rely on source_docs now
            "textbook": selected_textbook,
            "score": None
        })

    # 6) Voice only
    if "Voice" in option:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"🎙️ Generating voice with '{voice_choice}'..."):
            speech_response = openai.audio.speech.create(
                model="tts-1",
                voice=voice_choice,
                input=response_text
            )
            audio_path = "output.mp3"
            with open(audio_path, "wb") as f:
                f.write(speech_response.read())
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
    else:
        st.markdown("### 📘 Answer")
        st.write(response_text)

    # 7) Show supporting context that the chain actually used
    with st.expander("📚 Show Supporting Context"):
        if not source_docs:
            st.write("No supporting documents returned.")
        else:
            top_doc = source_docs[0]
            meta = top_doc.metadata or {}
            origin = meta.get("source", "unknown")  # 'textbook' / 'user_memory' if you set it
            st.markdown(f"**Source:** `{origin}` — **Textbook:** {meta.get('textbook', 'N/A')}")
            snippet = (top_doc.page_content or "").strip()
            st.write(snippet[:1200] + ("..." if len(snippet) > 1200 else ""))


# ------------------- Exit Button -------------------
def embed_and_upload_logs_on_exit(session_log):
    """
    Send session Q&A to the user_memory collection with payload fields used by retrieval.
    """
    from langchain.schema import Document

    interactions = session_log.get("interactions", []) or []
    if not interactions:
        return 0  # nothing to upload

    if "uploaded_to_memory" in st.session_state and st.session_state["uploaded_to_memory"]:
        return 0  # already uploaded this session

    if "memory_db" not in globals():
        st.warning("⚠️ user_memory vector store is not initialized; skipping memory upload.")
        return 0

    new_docs = []
    for entry in interactions:
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        if not q or not a:
            continue

        # keep payloads small (helps cost & speed)
        MAX_CHARS = 3000
        content = f"Q: {q}\nA: {a}"
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + " …"

        metadata = {
            "username": session_log.get("username"),
            "textbook": session_log.get("textbook"),
            "experience_level": entry.get("experience_level"),
            "option": entry.get("option"),
            "timestamp": entry.get("timestamp"),  # UTC upstream
            "source": "user_memory",
        }
        new_docs.append(Document(page_content=content, metadata=metadata))

    uploaded = 0
    if new_docs:
        try:
            memory_db.add_documents(new_docs)
            uploaded = len(new_docs)
            st.session_state["uploaded_to_memory"] = True
        except Exception as e:
            st.warning(f"⚠️ Failed to embed session Q&A to Qdrant user_memory: {e}")

    return uploaded

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    custom_exit = """
    <style>
    div.stButton > button:first-child {
        font-size: 18px;
        padding: 0.6em 2em;
        width: 100%;
        border-radius: 10px;
        background-color: #ffb47a;
        color: white;
    }
    </style>
    """
    st.markdown(custom_exit, unsafe_allow_html=True)

    if st.button("🚪 Exit", key="exit_session"):
        uploaded_count = 0
        if "session_log" in st.session_state:
            # 1) Embed to user_memory
            uploaded_count = embed_and_upload_logs_on_exit(st.session_state["session_log"])

            # 2) Always push the JSON log (record-keeping)
            try:
                success = append_log_to_github(st.session_state["session_log"])
                if success:
                    st.success(f"📤 Session log uploaded to GitHub. (Memory docs: {uploaded_count})")
                else:
                    st.warning("⚠️ Failed to upload session log to GitHub.")
            except Exception as e:
                st.warning(f"⚠️ GitHub upload error: {e}")

        # Fresh start next login (buffer is rebuilt on login)
        st.session_state.clear()
        st.rerun()

