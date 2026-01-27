"""
FinancialHandler - Specialized handler for financial documents (reports, statements)

Strategy:
- Extract and preserve table structures
- Identify key financial metrics (KPIs)
- Tag chunks with financial period metadata
"""
import pymupdf4llm
import pdfplumber
import re
from typing import List, Dict, Any
from .base import BaseHandler


class FinancialHandler(BaseHandler):
    """Handler for financial documents (reports, statements, analyses)"""
    
    # Common financial terms to detect in headers
    FINANCIAL_KEYWORDS = [
        'revenue', 'income', 'profit', 'loss', 'ebitda', 'earnings',
        'balance sheet', 'cash flow', 'assets', 'liabilities', 'equity',
        'quarterly', 'annual', 'fiscal', 'fy', 'q1', 'q2', 'q3', 'q4'
    ]
    
    def get_type_name(self) -> str:
        return "Financial Report (Table/KPI Focus)"
    
    def ingest(self, file_path: str) -> str:
        """Extract text and tables from financial PDF"""
        print(f"💰 [FinancialHandler] Processing {file_path}...")
        
        # First, get markdown text (good for narrative sections)
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Also extract tables using pdfplumber (more accurate for tables)
        tables_text = self._extract_tables(file_path)
        
        # Combine both
        combined = md_text + "\n\n--- EXTRACTED TABLES ---\n\n" + tables_text
        
        # Clean up
        cleaned = combined.replace("<br>", " ")
        cleaned = cleaned.replace("_", " ")
        cleaned = re.sub(r'- \n', '', cleaned)
        
        print(f"✅ [FinancialHandler] Ingested {len(cleaned)} chars with tables")
        return cleaned
    
    def _extract_tables(self, file_path: str) -> str:
        """Extract tables from PDF using pdfplumber"""
        tables_text = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:
                            # Convert table to markdown format
                            table_md = self._table_to_markdown(table)
                            if table_md:
                                tables_text.append(f"**Table (Page {page_num + 1})**\n{table_md}")
        except Exception as e:
            print(f"⚠️ [FinancialHandler] Table extraction error: {e}")
        
        return "\n\n".join(tables_text)
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert a table (list of rows) to markdown format"""
        if not table or len(table) < 2:
            return ""
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                cell_text = str(cell) if cell else ""
                cell_text = cell_text.replace("\n", " ").strip()
                cleaned_row.append(cell_text)
            cleaned_table.append(cleaned_row)
        
        # Build markdown table
        lines = []
        
        # Header
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in cleaned_table[1:]:
            # Pad row if needed
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split financial document, preserving table boundaries"""
        print("✂️ [FinancialHandler] Chunking with table preservation...")
        
        chunks = []
        
        # Split on major section boundaries
        sections = re.split(r'\n(?=#+\s|---|\*\*Table)', text)
        
        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue
            
            # Detect if this is a table section
            is_table = section.startswith("**Table") or "|" in section[:200]
            
            # Try to detect financial period
            period = self._detect_period(section)
            
            # If section is too large, sub-chunk it
            if len(section) > 2000 and not is_table:
                sub_chunks = self._size_chunk(section, max_size=1500)
                for i, sub in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub,
                        "metadata": {
                            "type": "financial",
                            "is_table": False,
                            "period": period,
                            "subsection": i + 1
                        }
                    })
            else:
                chunks.append({
                    "text": section,
                    "metadata": {
                        "type": "financial",
                        "is_table": is_table,
                        "period": period
                    }
                })
        
        print(f"✅ [FinancialHandler] Created {len(chunks)} financial chunks")
        return chunks
    
    def _detect_period(self, text: str) -> str:
        """Try to detect financial period mentioned in text"""
        text_lower = text.lower()
        
        # Look for year patterns
        year_match = re.search(r'(fy|fiscal year|annual)\s*(20\d{2})', text_lower)
        if year_match:
            return f"FY{year_match.group(2)}"
        
        # Quarter patterns
        quarter_match = re.search(r'(q[1-4])\s*(20\d{2})', text_lower)
        if quarter_match:
            return f"{quarter_match.group(1).upper()} {quarter_match.group(2)}"
        
        # Plain year
        year_only = re.search(r'20\d{2}', text)
        if year_only:
            return year_only.group(0)
        
        return "unknown"
    
    def _size_chunk(self, text: str, max_size: int = 1500) -> List[str]:
        """Simple size-based chunking"""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) > max_size and current:
                chunks.append(' '.join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
