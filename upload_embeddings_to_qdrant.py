import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

DOC_PATH = "macroeconomics_textbook.pdf" # change this
COLLECTION_NAME = "macroecon_collection"

# --------------------- Load and Chunk PDF ---------------------
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(pages)
documents = [doc for doc in documents if doc.page_content.strip()]

# --------------------- Initialize Qdrant ---------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if client.collection_exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' exists. Deleting and recreating...")
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# --------------------- Embed and Upload ---------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
)

print("✅ Successfully uploaded textbook to Qdrant.")
