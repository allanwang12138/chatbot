import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

DOC_PATH = "macroeconomics_textbook.pdf" # change this
COLLECTION_NAME = "macroecon_collection"

# Set up embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Qdrant Cloud client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Create collection if not exists
if not client.collection_exists(collection_name=COLLECTION_NAME):
    print("Creating collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
else:
    print("Collection already exists.")

# Load and split the PDF
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(pages)

# Prepare texts and metadata
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]

# Embed texts
print("Embedding texts...")
vectors = embedding_model.embed_documents(texts)

# Convert to PointStruct list
print("Uploading to Qdrant...")
# Convert to PointStruct list
# Convert to PointStruct list
points = [
    PointStruct.model_construct(
        id=i,  # ← USE PURE INTEGER HERE ✅
        vector=vectors[i],
        payload={
            "text": texts[i],
            **(metadatas[i] if metadatas else {})
        }
    )
    for i in range(len(vectors))
]


# Chunking helper
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Upload in batches
print("Uploading to Qdrant in batches...")
batch_size = 100  # adjust to 50 if error persists
for i, batch in enumerate(chunked(points, batch_size)):
    print(f"Uploading batch {i+1} of {len(points) // batch_size + 1}...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

print("✅ Successfully uploaded all chunks to Qdrant Cloud.")
