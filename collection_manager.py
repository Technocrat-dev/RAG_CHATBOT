"""
Collections Manager - Manages document collections for multi-document RAG.

Each collection is an isolated namespace for documents, allowing separate
chat contexts for different document sets.
"""
import json
import os
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import config

COLLECTIONS_PATH = config.COLLECTIONS_PATH


@dataclass
class Collection:
    """Represents a document collection"""
    id: str
    name: str
    description: str
    created_at: str
    document_count: int = 0
    files: List[str] = None  # List of uploaded filenames
    
    def __post_init__(self):
        if self.files is None:
            self.files = []
    
    def to_dict(self) -> dict:
        return asdict(self)


class CollectionsManager:
    """
    Manages document collections stored in JSON.
    For production, replace with SQLite or PostgreSQL.
    """
    
    def __init__(self):
        self.collections: Dict[str, Collection] = {}
        self._load()
    
    def _load(self):
        """Load collections from disk"""
        if os.path.exists(COLLECTIONS_PATH):
            try:
                with open(COLLECTIONS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for coll_data in data.get("collections", []):
                        coll = Collection(**coll_data)
                        self.collections[coll.id] = coll
                print(f"📂 Loaded {len(self.collections)} collections")
            except Exception as e:
                print(f"⚠️ Failed to load collections: {e}")
    
    def _save(self):
        """Persist collections to disk"""
        os.makedirs(os.path.dirname(COLLECTIONS_PATH), exist_ok=True)
        with open(COLLECTIONS_PATH, 'w', encoding='utf-8') as f:
            data = {"collections": [c.to_dict() for c in self.collections.values()]}
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create(self, name: str, description: str = "") -> Collection:
        """Create a new collection"""
        coll = Collection(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            document_count=0
        )
        self.collections[coll.id] = coll
        self._save()
        print(f"✅ Created collection: {name} ({coll.id})")
        return coll
    
    def get(self, collection_id: str) -> Optional[Collection]:
        """Get a collection by ID"""
        return self.collections.get(collection_id)
    
    def list_all(self) -> List[Collection]:
        """List all collections"""
        return list(self.collections.values())
    
    def update(self, collection_id: str, name: str = None, description: str = None) -> Optional[Collection]:
        """Update collection metadata"""
        coll = self.collections.get(collection_id)
        if not coll:
            return None
        
        if name:
            coll.name = name
        if description:
            coll.description = description
        
        self._save()
        return coll
    
    def delete(self, collection_id: str) -> bool:
        """Delete a collection"""
        if collection_id in self.collections:
            del self.collections[collection_id]
            self._save()
            print(f"🗑️ Deleted collection: {collection_id}")
            return True
        return False
    
    def increment_doc_count(self, collection_id: str, count: int = 1):
        """Increment document count for a collection"""
        if collection_id in self.collections:
            self.collections[collection_id].document_count += count
            self._save()
    
    def add_file(self, collection_id: str, filename: str):
        """Track an uploaded file for a collection"""
        if collection_id in self.collections:
            if filename not in self.collections[collection_id].files:
                self.collections[collection_id].files.append(filename)
                self._save()
    
    def get_files(self, collection_id: str) -> List[str]:
        """Get list of files for a collection"""
        if collection_id in self.collections:
            return self.collections[collection_id].files
        return []
    
    def get_or_create_default(self) -> Collection:
        """Get the default collection, creating it if needed"""
        default_id = "default"
        if default_id not in self.collections:
            coll = Collection(
                id=default_id,
                name="All Documents",
                description="Default collection for all documents",
                created_at=datetime.now().isoformat(),
                document_count=0
            )
            self.collections[default_id] = coll
            self._save()
        return self.collections[default_id]
