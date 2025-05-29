import os
from dotenv import load_dotenv  # ✅ Load environment variables

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# SQLite fix for LangChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ✅ Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Paths
DOC_PATH = "/workspaces/chatbot/macroeconomics_textbook.pdf"
CHROMA_PATH = "/workspaces/chatbot/chroma_db"

# Load, split, embed, and persist
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

print("✅ Chroma DB successfully built and saved to:", CHROMA_PATH)
