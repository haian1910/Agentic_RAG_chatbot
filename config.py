import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Vector store settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.0-flash-exp"

# Search settings
WEB_SEARCH_RESULTS = 5

# File paths
VECTORSTORE_PATH = "vectorstore"
DOCUMENTS_PATH = "documents"