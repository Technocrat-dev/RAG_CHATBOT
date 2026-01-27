"""
Self-Correction Pipeline Package

Provides self-correcting RAG capabilities:
- QueryRewriter: Expands and refines queries for better retrieval
- AnswerValidator: Validates answers against source context
- SelfCorrectingRAG: Complete pipeline with iterative refinement
"""

from .query_rewriter import QueryRewriter
from .validator import AnswerValidator
from .pipeline import SelfCorrectingRAG

__all__ = [
    'QueryRewriter',
    'AnswerValidator', 
    'SelfCorrectingRAG'
]
