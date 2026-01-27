"""
ImageHandler - Handler for images and visual content

Strategy:
- Use vision model (LLaVA) to describe images
- Generate text descriptions for embedding
- Store image path in metadata for display
"""
import os
import ollama
import base64
from typing import List, Dict, Any
from .base import BaseHandler


class ImageHandler(BaseHandler):
    """Handler for images and visual content"""
    
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
    
    def __init__(self, vision_model: str = "llava"):
        self.vision_model = vision_model
    
    def get_type_name(self) -> str:
        return "Image/Visual Content"
    
    def ingest(self, file_path: str) -> str:
        """Use vision model to describe the image"""
        print(f"🖼️ [ImageHandler] Analyzing image: {file_path}")
        
        # Verify file exists and is supported
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        try:
            # Call vision model to describe the image
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': '''Describe this image in detail. Include:
1. What the image shows (main subject, objects, people)
2. Any text visible in the image
3. Colors, layout, and visual style
4. If it's a chart/graph/diagram, explain what data it represents
5. Any relevant context that would help someone searching for this image

Be thorough but concise.''',
                    'images': [file_path]
                }]
            )
            
            description = response['message']['content']
            print(f"✅ [ImageHandler] Generated description ({len(description)} chars)")
            
            # Prepend metadata to description
            filename = os.path.basename(file_path)
            return f"[IMAGE: {filename}]\n\n{description}"
            
        except Exception as e:
            print(f"❌ [ImageHandler] Vision model error: {e}")
            # Fallback to basic metadata
            return f"[IMAGE: {os.path.basename(file_path)}]\n\nImage file - vision model unavailable"
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Images don't need chunking - return as single chunk"""
        print("📦 [ImageHandler] Creating single chunk for image...")
        
        # Extract filename from the text if present
        filename = "unknown"
        if text.startswith("[IMAGE:"):
            end_bracket = text.find("]")
            if end_bracket > 0:
                filename = text[8:end_bracket].strip()
        
        return [{
            "text": text,
            "metadata": {
                "type": "image",
                "filename": filename,
                "is_visual": True
            }
        }]


class PDFImageExtractor:
    """Extract images embedded in PDF documents"""
    
    def __init__(self, output_dir: str = "data/extracted_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract all images from a PDF, return list of saved image paths"""
        import fitz  # PyMuPDF
        
        print(f"🔍 [PDFImageExtractor] Extracting images from {pdf_path}")
        
        extracted_paths = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image
                        image_name = f"{pdf_name}_page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                        image_path = os.path.join(self.output_dir, image_name)
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        extracted_paths.append(image_path)
                        
                    except Exception as e:
                        print(f"⚠️ Could not extract image {xref}: {e}")
            
            doc.close()
            print(f"✅ [PDFImageExtractor] Extracted {len(extracted_paths)} images")
            
        except Exception as e:
            print(f"❌ [PDFImageExtractor] Error: {e}")
        
        return extracted_paths
