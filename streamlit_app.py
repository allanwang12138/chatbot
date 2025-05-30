import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize LangChain embedding model
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("🧠 LangChain + Pinecone RAG App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    # Store in Pinecone via LangChain
    vectorstore = LangchainPinecone.from_documents(
        chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME
    )

    st.success("✅ PDF embedded and stored in Pinecone!")

# Ask questions
query = st.text_input("Ask a question about the PDF:")

if query:
    # Load the vectorstore
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding
    )

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        "Answer the following based on the context:\n\n{context}\n\nQuestion: {question}"
    )

    messages = prompt.format_messages(context=context, question=query)
    response = llm(messages)

    st.subheader("Answer:")
    st.write(response.content)
