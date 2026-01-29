"""
ImageHandler - Handler for images and visual content

Strategy:
- Use VisionPipeline (advanced vision model wrapper) to describe images
- Generate text descriptions for embedding
- Store image path in metadata for display
"""
import os
from typing import List, Dict, Any
from .base import BaseHandler
from multimodal import VisionPipeline


class ImageHandler(BaseHandler):
    """Handler for images and visual content using VisionPipeline"""
    
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
    
    def __init__(self, vision_model: str = "llava"):
        self.vision_pipeline = VisionPipeline(model=vision_model)
    
    def get_type_name(self) -> str:
        return "Image/Visual Content"
    
    def ingest(self, file_path: str) -> str:
        """Use VisionPipeline to analyze the image"""
        print(f"🖼️ [ImageHandler] Analyzing image: {file_path}")
        
        # Verify file exists and is supported
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        # Use VisionPipeline for comprehensive analysis
        analysis = self.vision_pipeline.analyze(file_path)
        
        # Build a comprehensive text representation
        parts = [
            f"IMAGE DESCRIPTION:\n{analysis.description}",
        ]
        
        # Add extracted text if available
        if analysis.extracted_text:
            parts.append(f"\nEXTRACTED TEXT:\n{analysis.extracted_text}")
        
        # Add chart data if available
        if analysis.chart_data:
            parts.append(f"\nCHART/GRAPH DATA:\n{analysis.chart_data}")
        
        # Add metadata
        parts.append(f"\n[Image Type: {analysis.image_type}]")
        parts.append(f"[File: {os.path.basename(file_path)}]")
        
        full_description = "\n".join(parts)
        
        print(f"✅ [ImageHandler] Image analyzed ({analysis.image_type})")
        print(f"   - Has text: {bool(analysis.extracted_text)}")
        print(f"   - Has chart data: {bool(analysis.chart_data)}")
        
        return full_description
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Images don't need chunking - return as single chunk"""
        print("📦 [ImageHandler] Creating single chunk for image...")
        
        # Extract filename from the text if present
        filename = "unknown"
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('[File:'):
                filename = line.split('[File: ')[1].rstrip(']')
                break
        
        return [{
            "text": text,
            "metadata": {
                "type": "image",
                "filename": filename,
                "is_visual": True
            }
        }]


class PDFImageExtractor:
    """Extract images from PDF documents"""
    
    @staticmethod
    def extract_images(pdf_path: str, output_dir: str = "extracted_images") -> List[str]:
        """
        Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        import fitz  # PyMuPDF
        
        os.makedirs(output_dir, exist_ok=True)
        extracted_paths = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images()
                
                for img_idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image
                    image_name = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_name)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    extracted_paths.append(image_path)
            
            doc.close()
            print(f"✅ Extracted {len(extracted_paths)} images from {pdf_path}")
            
        except Exception as e:
            print(f"⚠️ Error extracting images from PDF: {e}")
        
        return extracted_paths
