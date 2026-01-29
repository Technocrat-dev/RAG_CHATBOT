"""
Session Store - SQLite-based persistent chat session storage.

Manages chat sessions and message history for multi-tab chat interface.
"""
import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import config

DB_PATH = config.SESSION_DB_PATH



@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None


@dataclass
class ChatSession:
    """A chat session with metadata"""
    id: str
    name: str
    collection_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class SessionStore:
    """
    SQLite-based session storage for persistent chat history.
    """
    
    def __init__(self):
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database and tables"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                collection_id TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                sources TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
        print("📂 Session store initialized")
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection"""
        return sqlite3.connect(DB_PATH)
    
    def create_session(self, name: str, collection_id: str = "default") -> ChatSession:
        """Create a new chat session"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO sessions (id, name, collection_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, name, collection_id, now, now)
        )
        
        conn.commit()
        conn.close()
        
        return ChatSession(
            id=session_id,
            name=name,
            collection_id=collection_id,
            created_at=now,
            updated_at=now,
            message_count=0
        )
    
    def list_sessions(self, collection_id: Optional[str] = None) -> List[ChatSession]:
        """List all sessions, optionally filtered by collection"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if collection_id:
            cursor.execute(
                "SELECT id, name, collection_id, created_at, updated_at FROM sessions WHERE collection_id = ? ORDER BY updated_at DESC",
                (collection_id,)
            )
        else:
            cursor.execute(
                "SELECT id, name, collection_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
        
        rows = cursor.fetchall()
        sessions = []
        
        for row in rows:
            # Get message count
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (row[0],))
            msg_count = cursor.fetchone()[0]
            
            sessions.append(ChatSession(
                id=row[0],
                name=row[1],
                collection_id=row[2],
                created_at=row[3],
                updated_at=row[4],
                message_count=msg_count
            ))
        
        conn.close()
        return sessions
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, collection_id, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        msg_count = cursor.fetchone()[0]
        
        conn.close()
        
        return ChatSession(
            id=row[0],
            name=row[1],
            collection_id=row[2],
            created_at=row[3],
            updated_at=row[4],
            message_count=msg_count
        )
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted
    
    def add_message(self, session_id: str, role: str, content: str, 
                    confidence: Optional[float] = None, sources: Optional[List[str]] = None):
        """Add a message to a session"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        sources_json = json.dumps(sources) if sources else None
        
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, confidence, sources) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, now, confidence, sources_json)
        )
        
        # Update session timestamp
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        
        conn.commit()
        conn.close()
    
    def get_messages(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages for a session"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT role, content, timestamp, confidence, sources FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit)
        )
        
        messages = []
        for row in cursor.fetchall():
            sources = json.loads(row[4]) if row[4] else None
            messages.append(ChatMessage(
                role=row[0],
                content=row[1],
                timestamp=row[2],
                confidence=row[3],
                sources=sources
            ))
        
        conn.close()
        return messages
    
    def clear_messages(self, session_id: str):
        """Clear all messages in a session"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
