import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.prompts import ChatPromptTemplate
from pydub import AudioSegment
from pydub.playback import play
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

CHROMA_PATH = "chatbot/chroma_db"
DOC_PATH = "macroeconomics_textbook.pdf"  # Your PDF file path

# Setup embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Function to build the Chroma DB if not present
def build_chroma_db():
    st.info("Chroma DB not found. Building DB from documents now. This may take a moment...")
    loader = PyPDFLoader(DOC_PATH)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    st.success("✅ Chroma DB successfully built and saved.")
    return db

# Check if DB exists (simple check: directory exists and not empty)
if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) == 0:
    db_chroma = build_chroma_db()
else:
    db_chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Prompt templates
PROMPT_DETAILED = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

PROMPT_CONCISE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a clear and concise summary in no more than 2 sentences.
"""

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
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _ in docs_chroma])

    if option == "Detailed Answer":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_DETAILED)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_CONCISE)

    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    with st.spinner("Generating answer..."):
        response = model.predict(prompt)

    if option in ["Detailed Answer", "Concise Answer"]:
        st.markdown("### Answer")
        st.write(response)
    elif option == "Concise Answer + Voice":
        with st.spinner("Generating voice..."):
            speech_response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response
            )
            audio_path = "output.mp3"
            with open(audio_path, "wb") as f:
                f.write(speech_response.read())
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")

    with st.expander("Show Retrieved Context"):
        st.write(context_text)
