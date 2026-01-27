"""
Document Handlers Package

Provides specialized handlers for different document types:
- TechnicalHandler: Technical manuals, documentation (markdown/header-based)
- LegalHandler: Legal documents, contracts (clause-based)
- FinancialHandler: Financial reports, statements (table/KPI focus)
- ImageHandler: Images and visual content (vision model)

The DocumentRouter automatically selects the appropriate handler based on content.
"""

from .base import BaseHandler
from .technical import TechnicalHandler
from .legal import LegalHandler
from .financial import FinancialHandler
from .image import ImageHandler, PDFImageExtractor
from .router import DocumentRouter, get_handler

__all__ = [
    'BaseHandler',
    'TechnicalHandler',
    'LegalHandler',
    'FinancialHandler',
    'ImageHandler',
    'PDFImageExtractor',
    'DocumentRouter',
    'get_handler'
]
