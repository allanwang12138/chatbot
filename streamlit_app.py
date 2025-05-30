import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone (legacy SDK)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Streamlit UI
st.title("Document QA with Pinecone & LangChain")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Load PDF
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Create embeddings
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Store in Pinecone
    vectorstore = LangchainPinecone.from_documents(
        chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
    )

    st.success("Documents indexed successfully in Pinecone.")

    # Ask questions
    query = st.text_input("Ask a question about your PDF:")
    if query:
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        prompt = ChatPromptTemplate.from_template("Answer the question based on the context: {context}\n\nQuestion: {question}")

        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        chain_input = prompt.format(context=context, question=query)
        response = llm.invoke(chain_input)

        st.write("Answer:")
        st.write(response.content)
