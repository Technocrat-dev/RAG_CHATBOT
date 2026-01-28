import chromadb
import uuid
import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Simple config constants
CHROMA_PATH = "chroma_db"
PARENT_STORE_PATH = os.path.join(CHROMA_PATH, "parent_store.json")
EMBED_MODEL = "all-MiniLM-L6-v2"

class VectorDB:
    def __init__(self, collection_name: str = "rag_master"):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Persistent Storage for Parents - loads from disk on init
        self.parent_store = self._load_parent_store()
    
    def _load_parent_store(self) -> dict:
        """Load parent chunks from persistent JSON storage"""
        if os.path.exists(PARENT_STORE_PATH):
            try:
                with open(PARENT_STORE_PATH, 'r', encoding='utf-8') as f:
                    print(f"📂 Loaded {os.path.getsize(PARENT_STORE_PATH)//1024}KB of parent chunks from disk")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load parent store: {e}")
        return {}
    
    def _save_parent_store(self):
        """Persist parent chunks to JSON file"""
        os.makedirs(CHROMA_PATH, exist_ok=True)
        with open(PARENT_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.parent_store, f, ensure_ascii=False)
        print(f"💾 Saved {len(self.parent_store)} parent chunks to disk") 

    def add_documents(self, parent_chunks: List[Dict[str, Any]]):
        """
        Takes Parent Chunks -> Generates Small Children -> Indexes Children
        """
        print(f"💾 Indexing {len(parent_chunks)} parent chunks...")
        
        child_texts = []
        child_metadatas = []
        child_ids = []

        for p_chunk in parent_chunks:
            # 1. Store the Big Parent Chunk
            parent_id = str(uuid.uuid4())
            self.parent_store[parent_id] = p_chunk["text"]
            
            # 2. Create Small Children (Sliding Window)
            children = self._split_into_children(p_chunk["text"])
            
            # 3. Prepare Children for Chroma
            for i, child_text in enumerate(children):
                child_texts.append(child_text)
                # Link Child -> Parent ID
                meta = p_chunk.get("metadata", {}).copy()
                meta["parent_id"] = parent_id
                child_metadatas.append(meta)
                child_ids.append(f"{parent_id}_{i}")

        # 4. Batch Upload
        if child_texts:
            embeddings = self.embedder.encode(child_texts)
            self.collection.add(
                documents=child_texts,
                embeddings=embeddings.tolist(),
                metadatas=child_metadatas,
                ids=child_ids
            )
        # 5. Persist parent store to disk
        self._save_parent_store()
        print("✅ Indexing Complete.")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        # 1. Search for Children
        query_vec = self.embedder.encode([query])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k
        )

        # 2. Retrieve their Parents
        retrieved_parents = []
        seen_ids = set()
        
        if results["metadatas"]:
            for meta in results["metadatas"][0]:
                pid = meta["parent_id"]
                if pid not in seen_ids and pid in self.parent_store:
                    retrieved_parents.append(self.parent_store[pid])
                    seen_ids.add(pid)
        
        return retrieved_parents

    def _split_into_children(self, text: str, window_size: int = 200) -> List[str]:
        """
        Sliding window splitter with improved sizing for better semantic embeddings.
        
        Args:
            text: Parent chunk text to split
            window_size: Number of words per child chunk (default 200)
        """
        words = text.split()
        if not words: 
            return []
        
        children = []
        step_size = 50  # 75% overlap for continuity
        
        for i in range(0, len(words), step_size):
            chunk = " ".join(words[i : i + window_size])
            if len(chunk) > 50:  # Minimum viable chunk
                children.append(chunk)
        
        return children