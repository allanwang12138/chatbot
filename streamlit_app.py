import streamlit as st
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# ------------------- Load secrets -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

openai.api_key = OPENAI_API_KEY
COLLECTION_NAME = "macroecon_collection"

# ------------------- Load credentials with voice assignment -------------------
@st.cache_data
def load_credentials():
    df = pd.read_csv("sample_credentials_with_voices.csv")  # updated filename
    return {
        row["username"]: {"password": row["password"], "voice": row["voice"]}
        for _, row in df.iterrows()
    }

CREDENTIALS = load_credentials()

def login():
    st.title("🔐 Login")
    st.write("Enter your username and password to access the app.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = CREDENTIALS.get(username)
        if user and user["password"] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["voice"] = user["voice"]
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
db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

# ------------------- Prompt Templates -------------------
PROMPT_DETAILED = """
You are an expert economics tutor. Your job is to answer questions in a clear, friendly, and educational way that is easy for students to follow.

Based only on the context below, write a well-explained answer to the question in no more than 4 sentences. Use simple language, provide helpful examples where appropriate, and avoid excessive technical detail.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_CONCISE = """
You are an expert economics tutor. Based only on the context below, provide a very concise answer to the question in no more than 2 sentences.

Context:
{context}

Question:
{question}

Concise Answer:
"""

# ------------------- Streamlit UI -------------------
st.title("📄 Macro Economics Q&A App")

query = st.text_input("Ask a question about Macro Economics:")

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
        MIN_SCORE = 0.8
        relevant_docs = [(doc, score) for doc, score in docs if score >= MIN_SCORE]

        if relevant_docs:
            top_doc, top_score = sorted(relevant_docs, key=lambda x: x[1], reverse=True)[0]
            context_text = top_doc.page_content
        else:
            context_text = ""
            top_score = None

    if context_text:
        # Format prompt based on selected type
        prompt_template = ChatPromptTemplate.from_template(
            PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE
        )
        prompt = prompt_template.format(context=context_text, question=query)

        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        with st.spinner("💬 Generating answer..."):
            response = model.predict(prompt)

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
# ------------------- Exit Button -------------------
st.markdown("---")

# Create 3 columns and center the button in the middle one
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Custom style to make the button larger
    custom_exit = """
    <style>
    div.stButton > button:first-child {
        font-size: 18px;
        padding: 0.6em 2em;
        width: 100%;
        border-radius: 10px;
        background-color: #f44336;
        color: white;
    }
    </style>
    """
    st.markdown(custom_exit, unsafe_allow_html=True)
    if st.button("🚪 Exit"):
        st.session_state.clear()
        st.rerun()


