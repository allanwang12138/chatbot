import streamlit as st
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DOC_PATH = "macroeconomics_textbook.pdf"

openai.api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def build_pinecone_index():
    st.info("Pinecone index is empty. Building from document...")
    loader = PyPDFLoader(DOC_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    st.success("✅ Pinecone index populated.")
    return LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

# Check if index exists and has vectors
index = pinecone.Index(PINECONE_INDEX_NAME)
index_stats = index.describe_index_stats()
if index_stats['total_vector_count'] == 0:
    db = build_pinecone_index()
else:
    db = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

# Prompts
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

# Streamlit UI
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
    docs = db.similarity_search_with_score(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _ in docs])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE)
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
            st.audio(audio_path, format="audio/mp3")

    with st.expander("Show Retrieved Context"):
        st.write(context_text)
