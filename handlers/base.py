from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseHandler(ABC):
    """
    The Template that ALL specialized handlers must follow.
    """
    
    @abstractmethod
    def ingest(self, file_path: str) -> str:
        """
        Step 1: Read the file and convert to text/markdown.
        """
        pass

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Step 2: Split the text into meaningful parent chunks.
        """
        pass
    
    @abstractmethod
    def get_type_name(self) -> str:
        """Returns the name of this handler (e.g., 'Technical Manual')"""
        pass