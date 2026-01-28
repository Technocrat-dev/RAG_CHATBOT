"""
Reranker - Cross-encoder re-ranking for improved retrieval precision

Re-ranks retrieved chunks using a cross-encoder model that directly 
compares query-document pairs, providing more accurate relevance scores
than bi-encoder similarity.
"""
from sentence_transformers import CrossEncoder
from typing import List, Tuple


class Reranker:
    """
    Cross-encoder based re-ranker for improving retrieval precision.
    
    Retrieves more candidates initially, then uses cross-encoder to 
    score and select the most relevant ones.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder.
                       Default is ms-marco-MiniLM-L-6-v2 (fast, good quality)
        """
        print(f"🔄 Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        """
        Re-rank documents based on relevance to query.
        
        Args:
            query: The user's search query
            documents: List of document texts to re-rank
            top_k: Number of top documents to return
            
        Returns:
            List of re-ranked documents (top_k most relevant)
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        # Create query-document pairs for scoring
        pairs = [(query, doc) for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending) and return top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
    
    def rerank_with_scores(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Re-rank documents and return with scores.
        
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
