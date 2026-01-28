"""
LegalHandler - Specialized handler for legal documents (contracts, agreements, regulations)

Strategy:
- Extract numbered clauses and sections
- Preserve cross-references between sections
- Handle dense legal terminology with clause-based chunking
"""
import pymupdf4llm
import re
from typing import List, Dict, Any
from .base import BaseHandler


class LegalHandler(BaseHandler):
    """Handler optimized for legal documents (contracts, agreements, regulations)"""
    
    # Patterns for detecting legal structure
    CLAUSE_PATTERNS = [
        r'^(\d+\.)+\d*\s+',           # 1.1, 2.3.4
        r'^(Article|ARTICLE)\s+\d+',   # Article 5
        r'^(Section|SECTION)\s+\d+',   # Section 3
        r'^(Clause|CLAUSE)\s+\d+',     # Clause 7
        r'^\([a-z]\)',                 # (a), (b), (c)
        r'^\([ivxlcdm]+\)',            # (i), (ii), (iii) - roman numerals
    ]
    
    def get_type_name(self) -> str:
        return "Legal Document (Clause-Based)"
    
    def ingest(self, file_path: str) -> str:
        """Convert legal PDF to clean text, preserving structure"""
        print(f"⚖️ [LegalHandler] Converting {file_path} to text...")
        
        # Convert to markdown (preserves some structure)
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Clean up common PDF artifacts
        cleaned = md_text.replace("<br>", "\n")
        cleaned = cleaned.replace("_", " ")
        cleaned = re.sub(r'- \n', '', cleaned)  # Fix hyphenation
        
        # Normalize multiple newlines but preserve paragraph breaks
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        print(f"✅ [LegalHandler] Ingested {len(cleaned)} characters")
        return cleaned
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split legal document by clauses and sections"""
        print("✂️ [LegalHandler] Splitting by legal clauses/sections...")
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_metadata = {"type": "legal", "clause": "preamble"}
        
        for line in lines:
            # Check if this line starts a new clause/section
            clause_match = self._detect_clause(line)
            
            if clause_match and current_chunk:
                # Save the current chunk with clause context
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) > 50:  # Minimum viable chunk
                    clause_prefix = f"[Legal Clause: {current_metadata['clause']}]\n\n"
                    chunks.append({
                        "text": clause_prefix + chunk_text,
                        "metadata": current_metadata.copy()
                    })
                
                # Start new chunk
                current_chunk = [line]
                current_metadata = {"type": "legal", "clause": clause_match}
            else:
                current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) > 50:
                clause_prefix = f"[Legal Clause: {current_metadata['clause']}]\n\n"
                chunks.append({
                    "text": clause_prefix + chunk_text,
                    "metadata": current_metadata.copy()
                })
        
        # If we got very few chunks, fall back to size-based chunking
        if len(chunks) < 3:
            print("⚠️ [LegalHandler] Few clauses found, using size-based fallback...")
            chunks = self._fallback_chunk(text)
        
        print(f"✅ [LegalHandler] Created {len(chunks)} legal chunks")
        return chunks
    
    def _detect_clause(self, line: str) -> str | None:
        """Check if a line starts a new legal clause/section"""
        line = line.strip()
        for pattern in self.CLAUSE_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # Also check for all-caps headers (common in legal docs)
        if line.isupper() and 3 < len(line) < 100:
            return line[:50]  # Truncate long headers
        
        return None
    
    def _fallback_chunk(self, text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Fallback to simple size-based chunking"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 5):  # Rough word count
            chunk_words = words[i:i + chunk_size // 5]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text) > 100:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"type": "legal", "clause": f"section_{len(chunks) + 1}"}
                })
        
        return chunks
