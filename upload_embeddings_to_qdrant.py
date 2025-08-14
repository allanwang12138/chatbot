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

# change this
BASE_FOLDER = "your directory"

textbook_name = [
    "Introductory Macroeconomics",
    "Introductory Microeconomics",
    "Statistics For Economics",
    "MATHEMATICS Textbook for Class IX",
    "MATHEMATICS Textbook for Class X",
    "MATHEMATICS Textbook for Class XI",
    "MATHEMATICS Textbook for Class XII PART I",
    "MATHEMATICS Textbook for Class XII PART II"
]

# ---------------- LOOP THROUGH TEXTBOOKS ----------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

for TEXTBOOK_LABEL in textbook_name:
    FOLDER_PATH = os.path.join(BASE_FOLDER, TEXTBOOK_LABEL)
    COLLECTION_NAME = TEXTBOOK_LABEL.lower().replace(" ", "_").replace("-", "_") + "_collection"

    print(f"\n📚 Processing textbook: {TEXTBOOK_LABEL}")
    print(f"📂 Folder: {FOLDER_PATH}")
    print(f"🗂 Collection: {COLLECTION_NAME}")

    # ---------- 1) LOAD + SPLIT ----------
    all_documents = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(FOLDER_PATH, filename)
            print(f"📄 Loading {filename}...")
            pages = PyPDFLoader(pdf_path).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)
            chunks = [c for c in chunks if c.page_content.strip()]
            all_documents.extend(chunks)

    print(f"✅ Loaded and split {len(all_documents)} chunks from PDFs.")

    # Add textbook-only metadata
    for doc in all_documents:
        doc.metadata["textbook"] = TEXTBOOK_LABEL
        doc.metadata["source"] = "textbook"

    # ---------- 2) CREATE / RECREATE COLLECTION ----------
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"⚠️ Collection '{COLLECTION_NAME}' exists. Deleting and recreating...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # ---------- 3) CREATE FILTER INDEXES ----------
    headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
    for field in ["textbook", "source"]:
        resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/index",
            headers=headers,
            json={"field_name": field, "field_schema": "keyword"},
            timeout=30,
        )
        if resp.status_code == 200:
            print(f"✅ Indexed field: {field}")
        else:
            print(f"❌ Failed to index field: {field} -> {resp.status_code} {resp.text}")

    # ---------- 4) EMBED + UPLOAD ----------
    Qdrant.from_documents(
        documents=all_documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    print(f"🎉 Uploaded textbook chunks to Qdrant. Created {COLLECTION_NAME}")
