# config.py
import os

# Paths
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"

# Derived paths (use these consistently across modules)
CHROMA_PATH = CHROMA_DB_DIR
PARENT_STORE_PATH = os.path.join(CHROMA_DB_DIR, "parent_store.json")
BM25_INDEX_PATH = os.path.join(CHROMA_DB_DIR, "bm25_index.json")
COLLECTIONS_PATH = os.path.join(CHROMA_DB_DIR, "collections.json")

# Models
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L6-v2
LLM_MODEL = "llama3"
VISION_MODEL = "llava"

# Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100