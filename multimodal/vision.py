"""
VisionPipeline - Process images through vision model

Uses LLaVA (or similar vision-language model) to:
- Generate detailed image descriptions
- Answer questions about images
- Extract text/data from charts and diagrams
"""
import os
import ollama
from typing import Optional
from dataclasses import dataclass


@dataclass
class ImageAnalysis:
    """Result of image analysis"""
    description: str
    extracted_text: Optional[str]  # OCR-like text extraction
    chart_data: Optional[str]  # Data from charts/graphs
    image_type: str  # 'photo', 'chart', 'diagram', 'document', 'unknown'


class VisionPipeline:
    """Process images using vision-language models"""
    
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
    
    def __init__(self, model: str = "llava"):
        self.model = model
    
    def describe(self, image_path: str, detail_level: str = "detailed") -> str:
        """
        Generate a text description of an image.
        
        Args:
            image_path: Path to the image file
            detail_level: 'brief', 'detailed', or 'comprehensive'
        
        Returns:
            Text description of the image
        """
        if not self._validate_image(image_path):
            return f"[Invalid or missing image: {image_path}]"
        
        prompts = {
            "brief": "Describe this image in 1-2 sentences.",
            "detailed": """Describe this image in detail. Include:
1. Main subject and objects
2. Any visible text
3. Colors and visual style
4. Context and setting""",
            "comprehensive": """Provide a comprehensive description of this image:
1. What does the image show? (main subject, people, objects)
2. What text is visible? (transcribe all readable text)
3. If it's a chart/graph/diagram, what data does it represent?
4. What are the colors, layout, and visual style?
5. What is the setting or context?
6. Any other notable details that would help someone understand this image?

Be thorough but concise."""
        }
        
        prompt = prompts.get(detail_level, prompts["detailed"])
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ [VisionPipeline] Error: {e}")
            return f"[Vision model error: {e}]"
    
    def analyze(self, image_path: str) -> ImageAnalysis:
        """
        Comprehensive image analysis including type detection and data extraction.
        """
        if not self._validate_image(image_path):
            return ImageAnalysis(
                description="Invalid or missing image",
                extracted_text=None,
                chart_data=None,
                image_type="unknown"
            )
        
        # First, get a detailed description
        description = self.describe(image_path, "comprehensive")
        
        # Detect image type
        image_type = self._detect_type(image_path, description)
        
        # Extract specialized content based on type
        extracted_text = None
        chart_data = None
        
        if image_type in ['document', 'diagram']:
            extracted_text = self._extract_text(image_path)
        
        if image_type == 'chart':
            chart_data = self._extract_chart_data(image_path)
        
        return ImageAnalysis(
            description=description,
            extracted_text=extracted_text,
            chart_data=chart_data,
            image_type=image_type
        )
    
    def answer_question(self, image_path: str, question: str) -> str:
        """Answer a specific question about an image"""
        if not self._validate_image(image_path):
            return "Cannot analyze - invalid or missing image."
        
        prompt = f"""Look at this image and answer the following question:

Question: {question}

Provide a clear, accurate answer based only on what you can see in the image."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            return response['message']['content']
            
        except Exception as e:
            return f"Error: {e}"
    
    def _validate_image(self, image_path: str) -> bool:
        """Check if image exists and is supported format"""
        if not os.path.exists(image_path):
            return False
        ext = os.path.splitext(image_path)[1].lower()
        return ext in self.SUPPORTED_FORMATS
    
    def _detect_type(self, image_path: str, description: str) -> str:
        """Detect the type of image based on content"""
        description_lower = description.lower()
        
        # Simple keyword-based detection
        if any(word in description_lower for word in ['chart', 'graph', 'bar', 'pie', 'line graph', 'axis']):
            return 'chart'
        elif any(word in description_lower for word in ['diagram', 'flowchart', 'schematic', 'architecture']):
            return 'diagram'
        elif any(word in description_lower for word in ['document', 'text', 'page', 'paragraph', 'printed']):
            return 'document'
        elif any(word in description_lower for word in ['photo', 'photograph', 'picture', 'scene']):
            return 'photo'
        
        return 'unknown'
    
    def _extract_text(self, image_path: str) -> str:
        """Extract readable text from document/diagram images"""
        prompt = """Extract and transcribe ALL readable text from this image.
Format the text to preserve its original structure as much as possible.
If there's no readable text, say "No text found"."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            return response['message']['content']
        except:
            return None
    
    def _extract_chart_data(self, image_path: str) -> str:
        """Extract data representation from charts/graphs"""
        prompt = """Analyze this chart/graph and extract the data it represents.
Describe:
1. Type of chart (bar, line, pie, etc.)
2. What the axes represent (if applicable)
3. Key data points and values
4. Any trends or insights visible

Format the data in a clear, structured way."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            return response['message']['content']
        except:
            return None
