"""
DocumentRouter - Automatically routes documents to appropriate handlers

Strategy:
- Uses file extension and content analysis to detect document type
- Falls back to TechnicalHandler for unknown types
"""
import os
import re
from typing import Optional, Tuple
from .base import BaseHandler
from .technical import TechnicalHandler
from .legal import LegalHandler
from .financial import FinancialHandler
from .image import ImageHandler


class DocumentRouter:
    """Automatically routes documents to the appropriate specialized handler"""
    
    def __init__(self):
        self.handlers = {
            'technical': TechnicalHandler(),
            'legal': LegalHandler(),
            'financial': FinancialHandler(),
            'image': ImageHandler()
        }
        self.default_handler = self.handlers['technical']
    
    def route(self, file_path: str) -> Tuple[BaseHandler, str]:
        """
        Determine document type and return appropriate handler.
        
        Returns:
            Tuple of (handler, detected_type)
        """
        doc_type = self.classify(file_path)
        handler = self.handlers.get(doc_type, self.default_handler)
        return handler, doc_type
    
    def classify(self, file_path: str) -> str:
        """Classify document type based on extension and content"""
        
        ext = os.path.splitext(file_path)[1].lower()
        
        # Image files
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
            return 'image'
        
        # For PDFs and other text documents, analyze content
        if ext in ['.pdf', '.txt', '.md', '.doc', '.docx']:
            return self._analyze_content(file_path)
        
        # Default to technical for unknown
        return 'technical'
    
    def _analyze_content(self, file_path: str) -> str:
        """Analyze document content to determine type"""
        
        try:
            # Read first portion of the file for analysis
            import pymupdf4llm
            
            # Get first ~2000 chars for analysis
            text = pymupdf4llm.to_markdown(file_path)[:3000].lower()
            
            # Score each document type
            scores = {
                'legal': self._score_legal(text),
                'financial': self._score_financial(text),
                'technical': self._score_technical(text)
            }
            
            # Return highest scoring type
            best_type = max(scores, key=scores.get)
            print(f"📋 [Router] Detected type: {best_type} (scores: {scores})")
            
            return best_type
            
        except Exception as e:
            print(f"⚠️ [Router] Content analysis failed: {e}")
            return 'technical'
    
    def _score_legal(self, text: str) -> int:
        """Count legal document indicators"""
        legal_terms = [
            'whereas', 'hereby', 'herein', 'thereof', 'pursuant',
            'agreement', 'contract', 'parties', 'clause', 'article',
            'section', 'provision', 'obligation', 'liability', 'indemnify',
            'terminate', 'breach', 'covenant', 'warrant', 'represent',
            'jurisdiction', 'governing law', 'arbitration', 'confidential'
        ]
        return sum(1 for term in legal_terms if term in text)
    
    def _score_financial(self, text: str) -> int:
        """Count financial document indicators"""
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'ebitda', 'earnings',
            'balance sheet', 'cash flow', 'assets', 'liabilities', 'equity',
            'quarterly', 'annual report', 'fiscal', 'q1', 'q2', 'q3', 'q4',
            'shareholders', 'dividend', 'stock', 'investment', 'financial',
            'gross margin', 'net income', 'operating', 'capital expenditure'
        ]
        
        # Also check for numeric patterns (tables, financials have many numbers)
        number_count = len(re.findall(r'\$[\d,]+|\d+%|\d{1,3}(?:,\d{3})+', text))
        
        term_score = sum(1 for term in financial_terms if term in text)
        return term_score + (number_count // 5)  # Bonus for lots of numbers
    
    def _score_technical(self, text: str) -> int:
        """Count technical document indicators"""
        technical_terms = [
            'installation', 'configuration', 'setup', 'manual', 'guide',
            'procedure', 'step', 'instruction', 'specification', 'requirement',
            'system', 'software', 'hardware', 'api', 'interface',
            'troubleshoot', 'error', 'warning', 'note', 'caution',
            'diagram', 'figure', 'table', 'appendix', 'reference'
        ]
        
        # Check for code patterns
        code_patterns = len(re.findall(r'```|def\s+|class\s+|function\s+|\(\)|->|==', text))
        
        term_score = sum(1 for term in technical_terms if term in text)
        return term_score + code_patterns


# Convenience function for simple usage
def get_handler(file_path: str) -> Tuple[BaseHandler, str]:
    """Get the appropriate handler for a file"""
    router = DocumentRouter()
    return router.route(file_path)
