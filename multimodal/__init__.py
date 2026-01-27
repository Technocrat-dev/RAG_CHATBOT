"""
Multi-Modal Support Package

Provides vision capabilities for the RAG system:
- VisionPipeline: Process images through vision model (LLaVA)
- Image description and question-answering about images
"""

from .vision import VisionPipeline

__all__ = [
    'VisionPipeline'
]
