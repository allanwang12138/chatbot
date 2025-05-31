import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

openai.api_key = OPENAI_API_KEY
DOC_PATH = "macroeconomics_textbook.pdf"
COLLECTION_NAME = "macroecon_collection"

# Setup embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Setup Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Helper to build collection
def build_qdrant_collection():
    st.info("No collection found. Building it from PDF...")

    # Load and split PDF
    loader = PyPDFLoader(DOC_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # (Re)create collection and upload embeddings
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    db = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )

    st.success("✅ Collection successfully created and uploaded to Qdrant.")
    return db

# Check for collection existence and build/load accordingly
existing_collections = [col.name for col in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    db = build_qdrant_collection()
else:
    db = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings  # 🔧 must be passed again
    )

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

# UI
st.title("📄 Macro Economics Q&A App")

# Optional: upload PDF (you only need this once if not hardcoded)
if not os.path.exists(DOC_PATH):
    uploaded_file = st.file_uploader("Upload your Macroeconomics textbook PDF", type="pdf")
    if uploaded_file is not None:
        with open(DOC_PATH, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ PDF uploaded. Please refresh to build the collection.")
        st.stop()

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
    else:
        st.markdown("### Answer")
        st.write(response)

    with st.expander("Show Retrieved Context"):
        st.write(context_text)
