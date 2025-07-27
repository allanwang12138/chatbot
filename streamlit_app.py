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


# ------------------- Normalize Response -------------
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)  # handles full-width characters
    text = re.sub(r"[^\w\s]", "", text.lower()).strip()  # lowercasing + remove punctuation
    return text

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


# ------------------- Load credentials with voice assignment -------------------
@st.cache_data
def load_credentials():
    df = pd.read_csv("sample_credentials_with_levels.csv")  # updated filename
    return {
        row["username"]: {
            "password": row["password"],
            "voice": row["voice"],
            "macro_level": row["macro_level"],
            "micro_level": row["micro_level"],
            "stats_level": row["stats_level"]
        }
        for _, row in df.iterrows()
    }
CREDENTIALS = load_credentials()

def login():
    st.title("🔐 Login")
    st.write("Enter your username, password, and select a textbook.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    textbook = st.selectbox("Select a textbook:", ["Introductory Macroeconomics", "Introductory Microeconomics", "Statistics For Economics"])

    if st.button("Login"):
        user = CREDENTIALS.get(username)
        if user and user["password"] == password:
            # ✅ Correctly map textbook to experience level field
            subject_key_map = {
                "Macroeconomics": "macro_level",
                "Microeconomics": "micro_level",
                "Stats": "stats_level"
            }
            level_key = subject_key_map.get(textbook, "macro_level")
            level = user.get(level_key, "Intermediate")

            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["voice"] = user["voice"]
            st.session_state["textbook"] = textbook
            st.session_state["experience_level"] = level
            st.session_state["session_log"] = {
                "username": username,
                "login_time": str(datetime.datetime.now()),
                "experience_level": level,
                "textbook": textbook,
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
# Select collection based on textbook choice
selected_textbook = st.session_state.get("textbook", "Introductory Macroeconomics")
if selected_textbook == "Introductory Macroeconomics":
    COLLECTION_NAME = "intro_macro_collection"
elif selected_textbook == "Introductory Microeconomics":
    COLLECTION_NAME = "intro_micro_collection"
elif selected_textbook == "Statistics For Economics":
    COLLECTION_NAME = "stats_econ_collection"
else:
    st.error("❌ Invalid textbook selection.")
    st.stop()
db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

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

Using only the context below, provide a clear and informative answer in no more than 20 sentences. Use standard macroeconomic terms and concepts, but keep explanations digestible and well-structured. Include examples if helpful.

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

Using only the context provided, write a focused and rigorous answer in no more than 20 sentences. Feel free to include concepts like equilibrium, derivatives, IS-LM, inflation expectations, or models if relevant.

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

# Format dynamic title and input prompt
st.title(f"📄 {selected_textbook} Q&A App")

# ------------------- Always-visible Chat History in Sidebar -------------------
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
        st.sidebar.markdown(f"**Q:** {item['question']}")
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

if query and option:
    with st.spinner("🔍 Searching for relevant context..."):
        raw_docs = db.similarity_search_with_score(query, k=5)
        docs = [(doc, score) for doc, score in raw_docs if doc.page_content.strip()]
        
        # Define a minimum score threshold for relevance
        MIN_SCORE = 0.75
        relevant_docs = [(doc, score) for doc, score in docs if score >= MIN_SCORE]

        if relevant_docs:
            top_doc, top_score = sorted(relevant_docs, key=lambda x: x[1], reverse=True)[0]
            context_text = top_doc.page_content
        else:
            context_text = ""
            top_score = None

    if context_text:
        # Format prompt based on selected type
        level = st.session_state.get("experience_level", "Intermediate")
        
        if option == "Concise Answer" or "Voice" in option:
            if level == "Beginner":
                prompt_template = ChatPromptTemplate.from_template(PROMPT_BEGINNER_CONCISE)
            elif level == "Advanced":
                prompt_template = ChatPromptTemplate.from_template(PROMPT_ADVANCED_CONCISE)
            else:
                prompt_template = ChatPromptTemplate.from_template(PROMPT_INTERMEDIATE_CONCISE)
        else:
            if level == "Beginner":
                prompt_template = ChatPromptTemplate.from_template(PROMPT_BEGINNER_DETAILED)
            elif level == "Advanced":
                prompt_template = ChatPromptTemplate.from_template(PROMPT_ADVANCED_DETAILED)
            else:
                prompt_template = ChatPromptTemplate.from_template(PROMPT_INTERMEDIATE_DETAILED)
        prompt = prompt_template.format(
            context=context_text,
            question=query,
            textbook=selected_textbook
        )
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        with st.spinner("💬 Generating answer..."):
            answer_type = "Concise" if "Concise" in option or "Voice" in option else "Detailed"
            existing_answer = find_similar_answer(
                SESSION_LOGS, query, level, selected_textbook, answer_type
            )
            if existing_answer:
                response = existing_answer
                st.info("🔁 Reused answer from previous session.")
            else:
                response = model.predict(prompt)

            st.session_state["session_log"]["interactions"].append({
                "timestamp": str(datetime.datetime.now()),
                "experience_level": st.session_state.get("experience_level", "Intermediate"),
                "question": raw_query,
                "option": option,
                "answer": response,
                "context": context_text,
                "textbook": st.session_state.get("textbook", "Introductory Macroeconomics"),
                "score": top_score
            })

        # Voice only
        if "Voice" in option:
            voice_choice = st.session_state.get("voice", "alloy")
            with st.spinner(f"🎙️ Generating voice with '{voice_choice}'..."):
                speech_response = openai.audio.speech.create(
                    model="tts-1",
                    voice=voice_choice,
                    input=response
                )
                audio_path = "output.mp3"
                with open(audio_path, "wb") as f:
                    f.write(speech_response.read())
                audio_file = open(audio_path, "rb")
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.markdown("### 📘 Answer")
            st.write(response)

        with st.expander("📚 Show Supporting Context from Textbook"):
            st.markdown(f"**Most Relevant Chunk — Score: {top_score:.2f}**")
            st.write(context_text)

    else:
        st.warning("⚠️ This question appears to be outside the scope of the textbook.")
        # Log out-of-scope question
        st.session_state["session_log"]["interactions"].append({
            "timestamp": str(datetime.datetime.now()),
            "experience_level": st.session_state.get("experience_level", "Intermediate"),
            "question": raw_query,
            "option": option,
            "answer": "⚠️ This question appears to be outside the scope of the textbook.",
            "context": "",
            "textbook": st.session_state.get("textbook", "Introductory Macroeconomics"),
            "score": None
        })

# ------------------- Exit Button -------------------

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
    
    if st.button("🚪 Exit"):
        # Push session log to GitHub
        if "session_log" in st.session_state:
            success = append_log_to_github(st.session_state["session_log"])
            if success:
                st.success("📤 Session log uploaded to GitHub.")
            else:
                st.warning("⚠️ Failed to upload session log.")
        st.session_state.clear()
        st.rerun()


