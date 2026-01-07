import chromadb
import uuid
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Simple config constants
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

class VectorDB:
    def __init__(self, collection_name: str = "rag_master"):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # In-Memory Storage for Parents (For production, use Redis or SQL)
        self.parent_store = {} 

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

    def _split_into_children(self, text: str, window_size=100) -> List[str]:
        """Simple sliding window splitter"""
        words = text.split()
        if not words: return []
        children = []
        for i in range(0, len(words), 25): # Step 25, Window 50
            chunk = " ".join(words[i : i + window_size])
            if len(chunk) > 20: 
                children.append(chunk)
        return children