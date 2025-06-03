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
You are an expert economics tutor. Your job is to answer questions in a detailed, clear, and educational way.

Based only on the following context, write a comprehensive answer to the question in no more than 6 sentences. Include explanations, definitions, and examples where appropriate, but keep the response focused and digestible.

Context:
{context}

Question:
{question}

Detailed Answer:
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
    with st.spinner("Searching context..."):
        docs = db.similarity_search_with_score(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc, _ in docs])

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE
    )
    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    with st.spinner("Generating answer..."):
        response = model.predict(prompt)

    if "Voice" in option:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"Generating voice with '{voice_choice}'..."):
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
        st.markdown("### Answer")
        st.write(response)

    with st.expander("Show Retrieved Context"):
        st.write(context_text)
