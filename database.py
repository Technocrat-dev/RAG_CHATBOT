import chromadb
import uuid
import os
import json
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import config

# Use centralized config
CHROMA_PATH = config.CHROMA_PATH
PARENT_STORE_PATH = config.PARENT_STORE_PATH
BM25_INDEX_PATH = config.BM25_INDEX_PATH
EMBED_MODEL = config.EMBEDDING_MODEL

class VectorDB:
    def __init__(self, collection_name: str = "rag_master"):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Persistent Storage for Parents - loads from disk on init
        self.parent_store = self._load_parent_store()
        
        # BM25 index for hybrid search
        self.bm25_corpus: List[List[str]] = []  # Tokenized documents
        self.bm25_parent_ids: List[str] = []    # Corresponding parent IDs
        self.bm25_index: BM25Okapi = None
        self._load_bm25_index()
    
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
    
    def _load_bm25_index(self):
        """Load BM25 corpus from disk and rebuild index"""
        if os.path.exists(BM25_INDEX_PATH):
            try:
                with open(BM25_INDEX_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.bm25_corpus = data.get("corpus", [])
                    self.bm25_parent_ids = data.get("parent_ids", [])
                    if self.bm25_corpus:
                        self.bm25_index = BM25Okapi(self.bm25_corpus)
                        print(f"📂 Loaded BM25 index with {len(self.bm25_corpus)} documents")
            except Exception as e:
                print(f"⚠️ Failed to load BM25 index: {e}")
    
    def _save_bm25_index(self):
        """Persist BM25 corpus to JSON file"""
        with open(BM25_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "corpus": self.bm25_corpus,
                "parent_ids": self.bm25_parent_ids
            }, f, ensure_ascii=False)
        print(f"💾 Saved BM25 index with {len(self.bm25_corpus)} documents") 

    def add_documents(self, parent_chunks: List[Dict[str, Any]], collection_id: str = "default"):
        """
        Takes Parent Chunks -> Generates Small Children -> Indexes Children
        Also builds BM25 index for hybrid search.
        
        Args:
            parent_chunks: List of parent chunk dictionaries with 'text' and 'metadata'
            collection_id: ID to group documents together (for isolated chat threads)
        """
        print(f"💾 Indexing {len(parent_chunks)} parent chunks for collection: {collection_id}")
        
        child_texts = []
        child_metadatas = []
        child_ids = []

        for p_chunk in parent_chunks:
            # 1. Store the Big Parent Chunk with collection_id
            parent_id = str(uuid.uuid4())
            self.parent_store[parent_id] = {
                "text": p_chunk["text"],
                "collection_id": collection_id
            }
            
            # 2. Add to BM25 corpus (tokenized)
            tokens = p_chunk["text"].lower().split()
            self.bm25_corpus.append(tokens)
            self.bm25_parent_ids.append(parent_id)
            
            # 3. Create Small Children (Sliding Window)
            children = self._split_into_children(p_chunk["text"])
            
            # 4. Prepare Children for Chroma
            for i, child_text in enumerate(children):
                child_texts.append(child_text)
                # Link Child -> Parent ID with collection_id
                meta = p_chunk.get("metadata", {}).copy()
                meta["parent_id"] = parent_id
                meta["collection_id"] = collection_id
                child_metadatas.append(meta)
                child_ids.append(f"{parent_id}_{i}")

        # 5. Batch Upload to ChromaDB
        if child_texts:
            embeddings = self.embedder.encode(child_texts)
            self.collection.add(
                documents=child_texts,
                embeddings=embeddings.tolist(),
                metadatas=child_metadatas,
                ids=child_ids
            )
        
        # 6. Rebuild BM25 index
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # 7. Persist both stores
        self._save_parent_store()
        self._save_bm25_index()
        print("✅ Indexing Complete.")

    def retrieve(self, query: str, top_k: int = 3, collection_id: str = None) -> List[str]:
        """
        Hybrid retrieval combining dense vector search (semantic) 
        and BM25 sparse search (keyword matching).
        
        Uses Reciprocal Rank Fusion (RRF) to combine scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            collection_id: If provided, only return documents from this collection
        """
        # Get more candidates initially for fusion
        n_candidates = top_k * 3
        
        # --- Dense Vector Search ---
        query_vec = self.embedder.encode([query])
        
        # Add collection filter if specified
        where_filter = {"collection_id": collection_id} if collection_id else None
        
        dense_results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=n_candidates,
            where=where_filter
        )
        
        # Extract parent IDs with rank scores
        dense_scores = {}  # parent_id -> RRF score
        if dense_results["metadatas"] and dense_results["metadatas"][0]:
            for rank, meta in enumerate(dense_results["metadatas"][0]):
                pid = meta.get("parent_id")
                if pid:
                    # Only include if collection matches (double-check)
                    if collection_id is None or meta.get("collection_id") == collection_id:
                        dense_scores[pid] = 1.0 / (rank + 60)  # RRF constant = 60
        
        # --- BM25 Sparse Search ---
        bm25_scores = {}  # parent_id -> RRF score
        if self.bm25_index and self.bm25_corpus:
            query_tokens = query.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top N by score
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_candidates]
            
            for rank, idx in enumerate(ranked_indices):
                if scores[idx] > 0:  # Only include if there's a match
                    pid = self.bm25_parent_ids[idx]
                    # Check collection filter for BM25 results
                    if collection_id is not None:
                        parent_data = self.parent_store.get(pid)
                        if isinstance(parent_data, dict) and parent_data.get("collection_id") != collection_id:
                            continue
                    bm25_scores[pid] = 1.0 / (rank + 60)  # RRF constant = 60
        
        # --- Reciprocal Rank Fusion ---
        all_parent_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        fused_scores = {}
        
        for pid in all_parent_ids:
            fused_scores[pid] = dense_scores.get(pid, 0) + bm25_scores.get(pid, 0)
        
        # Sort by fused score and get top_k
        sorted_parents = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Retrieve parent texts
        retrieved_parents = []
        for pid, score in sorted_parents:
            if pid in self.parent_store:
                parent_data = self.parent_store[pid]
                # Handle both old format (string) and new format (dict)
                if isinstance(parent_data, dict):
                    retrieved_parents.append(parent_data["text"])
                else:
                    retrieved_parents.append(parent_data)
        
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
    
    def delete_all(self):
        """
        Delete all documents from the database.
        Clears ChromaDB collection, parent store, and BM25 index.
        """
        print("🗑️ Deleting all documents...")
        
        # Clear ChromaDB
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.get_or_create_collection(name=self.collection.name)
        except Exception as e:
            print(f"⚠️ ChromaDB clear error: {e}")
        
        # Clear parent store
        self.parent_store = {}
        self._save_parent_store()
        
        # Clear BM25 index
        self.bm25_corpus = []
        self.bm25_parent_ids = []
        self.bm25_index = None
        self._save_bm25_index()
        
        print("✅ All documents deleted")
        return {"status": "deleted", "message": "All documents cleared"}
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        return {
            "parent_chunks": len(self.parent_store),
            "child_chunks": self.collection.count(),
            "bm25_documents": len(self.bm25_corpus),
            "embedding_model": EMBED_MODEL
        }