import streamlit as st
import openai
import os
import tempfile

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

import pinecone  # Using pinecone-client v2.2.4

# Load API keys and config from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "macro-econ-index")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone v2 client
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

# LangChain embedding setup
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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

# Streamlit UI
st.title("📄 Macro Economics Q&A App")

uploaded_pdf = st.file_uploader("Upload your Macro Economics PDF", type="pdf")
if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing and indexing document..."):
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        LangchainPinecone.from_documents(
            chunks,
            embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace="default"
        )
        st.success("✅ Document indexed into Pinecone.")

    query = st.text_input("Ask a question about your uploaded textbook:")

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
        vectorstore = LangchainPinecone(
            index_name=PINECONE_INDEX_NAME,
            embedding_function=embeddings,
            namespace="default"
        )
        docs = vectorstore.similarity_search_with_score(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc, _ in docs])

        prompt_template = ChatPromptTemplate.from_template(
            PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE
        )
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
else:
    st.info("📥 Upload a PDF to begin.")
