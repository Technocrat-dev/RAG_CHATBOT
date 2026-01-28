import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List, Dict, Any
from .base import BaseHandler

class TechnicalHandler(BaseHandler):
    def get_type_name(self) -> str:
        return "Technical Manual (Markdown/Header Based)"

    def ingest(self, file_path: str) -> str:
        print(f"📄 [TechnicalHandler] Converting {file_path} to Markdown...")
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # --- CORRECTED CLEANING ---
        # 1. Replace breaks with SPACES (Fixes "Chemito<br>Tech" -> "Chemito Tech")
        cleaned_text = md_text.replace("<br>", " ") 
        
        # 2. Fix underscores acting as spaces
        cleaned_text = cleaned_text.replace("_", " ")
        
        # 3. Fix hyphenation
        cleaned_text = cleaned_text.replace("- \n", "")
        
        # REMOVED THE LINE THAT DELETED NEWLINES!
        
        return cleaned_text

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        print("✂️ [TechnicalHandler] Splitting by Headers...")
        
        # Define the hierarchy we want to respect
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = splitter.split_text(text)
        
        # Convert to the standard dictionary format our DB expects
        clean_chunks = []
        for split in splits:
            # Build header context string from metadata
            header_parts = []
            for key in ["Header 1", "Header 2", "Header 3"]:
                if key in split.metadata:
                    header_parts.append(split.metadata[key])
            
            header_context = " > ".join(header_parts)
            
            # Prepend headers to content for better LLM context
            if header_context:
                text_with_context = f"[Section: {header_context}]\n\n{split.page_content}"
            else:
                text_with_context = split.page_content
            
            clean_chunks.append({
                "text": text_with_context,
                "metadata": split.metadata
            })
            
        print(f"✅ Created {len(clean_chunks)} parent chunks.")
        return clean_chunks