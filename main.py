# main.py
import os
import shutil
import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORT THIS
from pydantic import BaseModel
from typing import List, Optional

# Import our Factory Modules
from handlers import DocumentRouter
from database import VectorDB
# Ensure self_correction/__init__.py is created (see Step 2)
from self_correction import SelfCorrectingRAG 
import config

app = FastAPI()

# --- FIX 1: ADD CORS MIDDLEWARE ---
# This allows your frontend (React/Streamlit) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Workers
document_router = DocumentRouter()
db = VectorDB(collection_name="rag_master")
rag_pipeline = SelfCorrectingRAG(db=db, llm_model=config.LLM_MODEL)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    model: str = "llama3"
    use_self_correction: bool = True 

# --- ENDPOINTS ---

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):  # Removed 'async' to be safe
    file_location = f"{config.DATA_DIR}/{file.filename}"
    
    # Create data dir if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📥 Received file: {file.filename}")

    try:
        handler, doc_type = document_router.route(file_location)
        print(f"🔀 Using handler: {handler.get_type_name()}")
        
        raw_text = handler.ingest(file_location)
        chunks = handler.chunk(raw_text)
        db.add_documents(chunks)
        
        return {
            "status": "success", 
            "message": f"Indexed {len(chunks)} chunks from {file.filename}",
            "document_type": doc_type,
            "handler": handler.get_type_name()
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FIX 2: REMOVE 'async' ---
# Removing 'async' makes this a synchronous route, which prevents
# the server from freezing while waiting for Ollama.
@app.post("/chat")
def chat(request: ChatRequest):
    print(f"❓ Query: {request.query}")
    
    if request.use_self_correction:
        result = rag_pipeline.query(request.query)
        
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
        response = ollama.chat(model=request.model, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        return {
            "response": response['message']['content'], 
            "context_used": context_docs,
            "confidence": None,
            "iterations": 1,
            "was_corrected": False
        }

@app.get("/")
def read_root():
    return {"status": "running", "system": "Self-Correcting Multi-Modal RAG Factory"}