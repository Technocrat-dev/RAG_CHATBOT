# main.py
import os
import shutil
import ollama
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

# Import our Factory Modules
from handlers import DocumentRouter
from database import VectorDB
from self_correction import SelfCorrectingRAG
from collection_manager import CollectionsManager
from memory import SessionStore
import config

app = FastAPI(title="NeuralRAG API", version="2.0")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Workers
document_router = DocumentRouter()
db = VectorDB(collection_name="rag_master")
rag_pipeline = SelfCorrectingRAG(db=db, llm_model=config.LLM_MODEL)
collections_manager = CollectionsManager()
session_store = SessionStore()

# Ensure default collection exists
collections_manager.get_or_create_default()


# --- DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    model: str = "llama3"
    use_self_correction: bool = True
    session_id: Optional[str] = None
    collection_id: Optional[str] = None

class CollectionCreate(BaseModel):
    name: str
    description: str = ""

class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    document_count: int

class SessionUpdate(BaseModel):
    name: Optional[str] = None

class MessageCreate(BaseModel):
    role: str
    content: str
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None


# --- ENDPOINTS ---

@app.post("/upload")
def upload_document(file: UploadFile = File(...), collection_id: str = "default"):
    """
    Upload and index a document.
    
    Args:
        file: The document file to upload
        collection_id: ID of collection to add document to (for isolated chat threads)
    """
    file_location = f"{config.DATA_DIR}/{file.filename}"
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📥 Received file: {file.filename} for collection: {collection_id}")

    try:
        handler, doc_type = document_router.route(file_location)
        print(f"🔀 Using handler: {handler.get_type_name()}")
        
        raw_text = handler.ingest(file_location)
        chunks = handler.chunk(raw_text)
        db.add_documents(chunks, collection_id=collection_id)
        
        # Update collection document count (1 document, not chunk count)
        collections_manager.increment_doc_count(collection_id, 1)
        
        # Track the uploaded file
        collections_manager.add_file(collection_id, file.filename)
        
        return {
            "status": "success", 
            "message": f"Indexed {len(chunks)} chunks from {file.filename}",
            "document_type": doc_type,
            "handler": handler.get_type_name(),
            "collection_id": collection_id,
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """Standard chat endpoint (non-streaming)"""
    print(f"❓ Query: {request.query}")
    
    if request.use_self_correction:
        result = rag_pipeline.query(request.query, collection_id=request.collection_id)
        
        # Store in chat history if session provided
        if request.session_id:
            session_store.add_message(request.session_id, "user", request.query)
            session_store.add_message(request.session_id, "assistant", result.answer, result.confidence, result.sources)
        
        return {
            "response": result.answer,
            "context_used": result.sources,
            "confidence": result.confidence,
            "iterations": result.iterations,
            "was_corrected": result.was_corrected
        }
    else:
        # Simple RAG fallback
        context_docs = db.retrieve(request.query, top_k=3)
        
        if not context_docs:
            return {"response": "I couldn't find any relevant info."}

        formatted_context = "\n\n---\n\n".join(context_docs)
        
        prompt = f"""Use the context to answer the question.
CONTEXT:
{formatted_context}
QUESTION:
{request.query}
"""
        try:
            response = ollama.chat(model=request.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            answer = response['message']['content']
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")
        
        return {
            "response": answer, 
            "context_used": context_docs,
            "confidence": None,
            "iterations": 1,
            "was_corrected": False
        }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with self-correction.
    
    Streams NDJSON events:
    - {"type": "log", "step": "...", "message": "..."} - Processing steps
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "done", "confidence": 0.87, "sources": [...]} - Final result
    """
    print(f"❓ [Streaming] Query: {request.query}")
    
    def generate():
        final_answer = ""
        final_confidence = 0.0
        final_sources = []
        was_corrected = False
        
        try:
            # Run self-correction pipeline with streaming logs
            for event in rag_pipeline.query_streaming(request.query, collection_id=request.collection_id):
                if event["type"] == "log":
                    # Stream log events to frontend
                    yield json.dumps({
                        "type": "log",
                        "step": event.get("step", ""),
                        "message": event["message"],
                        "warning": event.get("warning", False)
                    }) + "\n"
                
                elif event["type"] == "done":
                    # Got final answer, now stream it token by token for effect
                    final_answer = event["answer"]
                    final_confidence = event["confidence"]
                    final_sources = event.get("sources", [])
                    was_corrected = event.get("was_corrected", False)
                    
                    # Stream answer in chunks for typing effect
                    words = final_answer.split(" ")
                    for i in range(0, len(words), 3):  # 3 words at a time
                        chunk = " ".join(words[i:i+3]) + " "
                        yield json.dumps({
                            "type": "token",
                            "content": chunk
                        }) + "\n"
            
            # Store in history
            if request.session_id:
                session_store.add_message(request.session_id, "user", request.query)
                session_store.add_message(request.session_id, "assistant", final_answer, final_confidence, final_sources)
            
            # Final done event with metadata
            yield json.dumps({
                "type": "done",
                "confidence": final_confidence,
                "sources": final_sources[:2],
                "was_corrected": was_corrected
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """Get chat history for a session (uses persistent session_store)"""
    messages = session_store.get_messages(session_id)
    return {"history": [{"role": m.role, "content": m.content} for m in messages]}


@app.delete("/chat/history/{session_id}")
def clear_chat_history(session_id: str):
    """Clear chat history for a session (deletes the session)"""
    session_store.delete_session(session_id)
    return {"status": "cleared"}


@app.delete("/documents")
def delete_all_documents():
    """Delete all documents from the database (requires confirmation)"""
    result = db.delete_all()
    return result


# --- COLLECTION ENDPOINTS ---

@app.get("/collections")
def list_collections():
    """List all document collections"""
    collections = collections_manager.list_all()
    return {"collections": [c.to_dict() for c in collections]}


@app.post("/collections")
def create_collection(data: CollectionCreate):
    """Create a new document collection"""
    coll = collections_manager.create(name=data.name, description=data.description)
    return {"collection": coll.to_dict()}


@app.delete("/collections/{collection_id}")
def delete_collection(collection_id: str):
    """Delete a collection and its documents"""
    if collections_manager.delete(collection_id):
        return {"status": "deleted", "collection_id": collection_id}
    raise HTTPException(status_code=404, detail="Collection not found")


# --- SESSION ENDPOINTS ---

@app.get("/sessions")
def list_sessions(collection_id: Optional[str] = None):
    """List all chat sessions, optionally filtered by collection"""
    sessions = session_store.list_sessions(collection_id)
    return {"sessions": [s.to_dict() for s in sessions]}


@app.post("/sessions")
def create_session(name: str = "New Chat", collection_id: str = "default"):
    """Create a new chat session"""
    session = session_store.create_session(name=name, collection_id=collection_id)
    return {"session": session.to_dict()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session details and messages"""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session_store.get_messages(session_id)
    return {
        "session": session.to_dict(),
        "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp, 
                      "confidence": m.confidence, "sources": m.sources} for m in messages]
    }


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, data: SessionUpdate):
    """Update session name"""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if data.name:
        # Update name in database
        conn = session_store._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET name = ? WHERE id = ?", (data.name, session_id))
        conn.commit()
        conn.close()
    
    return {"status": "updated", "session_id": session_id, "name": data.name}


@app.post("/sessions/{session_id}/messages")
def add_message(session_id: str, data: MessageCreate):
    """Add a message to a session"""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_store.add_message(session_id, data.role, data.content, data.confidence, data.sources)
    return {"status": "added", "session_id": session_id}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """
    Delete a chat session and all associated resources:
    - Documents from vector database and BM25 index
    - Uploaded files from data folder  
    - Session from SQLite database
    """
    # Get session to find its collection_id
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    collection_id = session.collection_id
    
    # 1. Delete documents from vector DB and BM25 index
    print(f"🗑️ Deleting documents for session {session_id} (collection: {collection_id})")
    db_result = db.delete_by_collection(collection_id)
    
    # 2. Delete uploaded files from data folder
    # Note: Multiple sessions might share the same collection, so we only delete
    # files if this is the last session using this collection
    remaining_sessions = [s for s in session_store.list_sessions(collection_id) if s.id != session_id]
    
    if not remaining_sessions:
        # This was the last session using this collection - safe to delete files
        tracked_files = collections_manager.get_files(collection_id)
        files_deleted = 0
        
        for filename in tracked_files:
            file_path = os.path.join(config.DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    files_deleted += 1
                    print(f"   Deleted file: {filename}")
                except Exception as e:
                    print(f"   Error deleting {filename}: {e}")
        
        print(f"✅ Deleted {files_deleted} files from data folder")
        
        # Also delete the collection from collections_manager
        collections_manager.delete(collection_id)
    else:
        print(f"ℹ️  Other sessions still using collection {collection_id}, keeping files")
    
    # 3. Delete the session itself
    session_store.delete_session(session_id)
    
    return {
        "status": "deleted",
        "session_id": session_id,
        "collection_id": collection_id,
        "documents_deleted": db_result.get("deleted_chunks", 0),
        "shared_collection": len(remaining_sessions) > 0
    }


@app.get("/stats")
def get_stats():
    """Get database and system statistics"""
    return {
        "database": db.get_stats(),
        "sessions": len(session_store.list_sessions()),
        "collections": len(collections_manager.list_all()),
        "system": "NeuralRAG v2.0"
    }


@app.get("/")
def read_root():
    return {"status": "running", "system": "NeuralRAG v2.0 - Self-Correcting Multi-Modal RAG"}


@app.get("/health")
def health_check():
    """
    Health check endpoint that verifies all dependencies are available.
    Returns status of Ollama connection, database, and system.
    """
    health = {
        "status": "healthy",
        "ollama": "unknown",
        "database": "unknown",
        "system": "NeuralRAG v2.0"
    }
    
    # Check Ollama
    try:
        ollama.list()
        health["ollama"] = "connected"
    except Exception as e:
        health["ollama"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    # Check database
    try:
        db_stats = db.get_stats()
        health["database"] = f"connected ({db_stats['parent_chunks']} docs)"
    except Exception as e:
        health["database"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    return health