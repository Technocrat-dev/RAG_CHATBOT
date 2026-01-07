import os
import shutil
import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# Import our Factory Modules
from handlers.technical import TechnicalHandler
from database import VectorDB
import config

app = FastAPI()

# Initialize the Workers
# (In the future, a Router will pick the right handler here)
current_handler = TechnicalHandler() 
db = VectorDB(collection_name="rag_master")

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    model: str = "llama3"  # Default to Llama-3

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    1. Saves the file locally.
    2. Ingests & Chunks it (using TechnicalHandler).
    3. Indexes it into ChromaDB.
    """
    file_location = f"{config.DATA_DIR}/{file.filename}"
    
    # Save the file to disk
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📥 Received file: {file.filename}")

    try:
        # 1. Ingest (Convert to Markdown)
        raw_text = current_handler.ingest(file_location)
        
        # 2. Chunk (Split by Headers)
        chunks = current_handler.chunk(raw_text)
        
        # 3. Store (Vector Database)
        db.add_documents(chunks)
        
        return {"status": "success", "message": f"Indexed {len(chunks)} chunks from {file.filename}"}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    1. Retrieves context from ChromaDB.
    2. Sends context + query to Ollama (Llama-3).
    3. Returns the answer.
    """
    print(f"❓ Query: {request.query}")
    
    # 1. Retrieve Context
    context_docs = db.retrieve(request.query, top_k=3)
    
    if not context_docs:
        return {"response": "I couldn't find any relevant info in the documents."}

    # 2. Construct Prompt
    # We join the retrieved parents into one big context block
    formatted_context = "\n\n---\n\n".join(context_docs)
    
    prompt = f"""You are a helpful assistant. Use the following context documents to answer the user's question.
    If the answer isn't in the context, say "I don't know".
    
    CONTEXT:
    {formatted_context}
    
    QUESTION:
    {request.query}
    """

    # 3. Generate Answer (using Ollama)
    print("🤖 Generating answer with Llama-3...")
    response = ollama.chat(model=request.model, messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    answer = response['message']['content']
    print("✅ Answer sent.")
    
    return {
        "response": answer, 
        "context_used": context_docs  # Sending this back helps the UI show "Sources"
    }

# --- ROOT ---
@app.get("/")
def read_root():
    return {"status": "running", "system": "Universal RAG Factory"}