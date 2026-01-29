# config.py
import os

# Paths (can be overridden via environment variables)
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
CHROMA_DB_DIR = os.environ.get("RAG_CHROMA_DIR", "chroma_db")

# Derived paths (use these consistently across modules)
CHROMA_PATH = CHROMA_DB_DIR
PARENT_STORE_PATH = os.path.join(CHROMA_DB_DIR, "parent_store.json")
BM25_INDEX_PATH = os.path.join(CHROMA_DB_DIR, "bm25_index.json")
COLLECTIONS_PATH = os.path.join(CHROMA_DB_DIR, "collections.json")
SESSION_DB_PATH = os.path.join(CHROMA_DB_DIR, "sessions.db")

# Models (can be overridden via environment variables)
EMBEDDING_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-mpnet-base-v2")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "llama3")
VISION_MODEL = os.environ.get("RAG_VISION_MODEL", "llava")

# Settings
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))
