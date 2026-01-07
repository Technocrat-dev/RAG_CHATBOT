import pdfplumber
import chromadb
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import ChatOllama

# Configuration
PDF_PATH = "data/sample.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

# ==========================================
# 1. TEXT CLEANING
# ==========================================
def clean_text(text: str) -> str:
    # Remove page numbers and excessive whitespace
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_pdf(path: str) -> str:
    print(f"📄 Reading {path}...")
    full_text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=True)
            if page_text:
                full_text += page_text + "\n"
    return full_text

# ==========================================
# 2. PARENT CHUNKING STRATEGIES
# ==========================================
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        pass

class HierarchicalChunker(ChunkingStrategy):
    """
    Creates PARENT chunks based on Document Structure (Headers).
    """
    def __init__(self, chunk_size=1000, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap 

    def is_header(self, line: str) -> bool:
        line = line.strip()
        if len(line) > 50 or len(line) < 3: return False
        
        # Resume/Doc Keywords
        keywords = ["education", "experience", "projects", "skills", 
                    "summary", "introduction", "conclusion", "references"]
        if any(k in line.lower() for k in keywords) and len(line) < 30:
            return True
        
        # All Caps Heuristic
        if line.isupper() and not line.endswith(('.', ':')):
            return True
        return False

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        print("🧱 Creating PARENT chunks (Hierarchical)...")
        lines = text.split('\n')
        chunks = []
        current_header = "General"
        buffer = []
        current_len = 0

        for line in lines:
            line = line.strip()
            if not line: continue

            if self.is_header(line):
                if current_len > 50: # Save previous section
                    chunks.append({
                        "text": f"CONTEXT: {current_header}\n" + "\n".join(buffer),
                        "metadata": {"section": current_header, "type": "parent"}
                    })
                    buffer = []
                    current_len = 0
                current_header = line
                continue

            buffer.append(line)
            current_len += len(line)

            if current_len >= self.chunk_size:
                chunks.append({
                    "text": f"CONTEXT: {current_header}\n" + "\n".join(buffer),
                    "metadata": {"section": current_header, "type": "parent"}
                })
                buffer = buffer[-5:] # Small overlap for continuity
                current_len = sum(len(w) for w in buffer)

        if buffer:
            chunks.append({
                "text": f"CONTEXT: {current_header}\n" + "\n".join(buffer),
                "metadata": {"section": current_header, "type": "parent"}
            })
        return chunks

class SemanticChunker(ChunkingStrategy):
    """
    Creates PARENT chunks based on Semantic Similarity (Topics).
    """
    def __init__(self, embedding_model, threshold=0.4):
        self.embedder = embedding_model
        self.threshold = threshold

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        print("🧠 Creating PARENT chunks (Semantic)...")
        sentences = re.split(r'(?<=[.?!])\s+', text)
        embeddings = self.embedder.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            score = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            if score >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": {"type": "parent"}
                })
                current_chunk = [sentences[i]]
        
        if current_chunk:
            chunks.append({"text": " ".join(current_chunk), "metadata": {"type": "parent"}})
        return chunks

# ==========================================
# 3. PARENT-CHILD INDEXER (The New Layer)
# ==========================================
class ParentChildIndexer:
    def __init__(self, embedder):
        self.embedder = embedder
        self.parent_store = {} # In-memory Key-Value store for Parents
        self.chroma_client = chromadb.Client()
        try:
            self.chroma_client.delete_collection("child_nodes")
        except: pass
        self.collection = self.chroma_client.create_collection("child_nodes")

    def split_into_children(self, text: str, chunk_size=300) -> List[str]:
        """Splits a big parent chunk into tiny overlapping windows."""
        words = text.split()
        children = []
        for i in range(0, len(words), 50): # Step of 50 words
            chunk = " ".join(words[i : i + 80]) # Window of 80 words
            if len(chunk) > 10:
                children.append(chunk)
            if i + 80 >= len(words): break
        return children

    def add_parents(self, parent_chunks: List[Dict[str, Any]]):
        print("👶 Generating CHILDREN and Indexing...")
        
        child_texts = []
        child_metadatas = []
        child_ids = []

        for p_chunk in parent_chunks:
            # 1. Create a Unique ID for the Parent
            parent_id = str(uuid.uuid4())
            parent_text = p_chunk["text"]
            
            # 2. Store Parent in 'KV Store' (Memory)
            self.parent_store[parent_id] = parent_text

            # 3. Generate Children
            children = self.split_into_children(parent_text)

            # 4. Prepare Children for Vector DB
            for i, child_text in enumerate(children):
                child_texts.append(child_text)
                child_metadatas.append({"parent_id": parent_id, "child_index": i})
                child_ids.append(f"{parent_id}_{i}")

        # 5. Batch Add to Chroma
        if child_texts:
            embeddings = self.embedder.encode(child_texts)
            self.collection.add(
                documents=child_texts,
                embeddings=embeddings.tolist(),
                metadatas=child_metadatas,
                ids=child_ids
            )
        print(f"✅ Indexed {len(child_texts)} child nodes pointing to {len(self.parent_store)} parent nodes.")

    def retrieve(self, query: str, top_k=3):
        # 1. Search for Children (The Needle)
        query_vec = self.embedder.encode([query])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k * 2 # Fetch more children to ensure we get diverse parents
        )

        # 2. Map to Parents (The Haystack)
        retrieved_parents = {}
        print("\n🔎 Internal Search Steps:")
        
        for i, meta in enumerate(results["metadatas"][0]):
            parent_id = meta["parent_id"]
            child_text = results["documents"][0][i]
            
            # Avoid duplicates (if 3 children point to same parent, only get parent once)
            if parent_id not in retrieved_parents:
                print(f"   • Hit Child: '{child_text[:50]}...' -> Found Parent {parent_id[:8]}")
                retrieved_parents[parent_id] = self.parent_store[parent_id]
            
            if len(retrieved_parents) >= top_k:
                break
        
        return list(retrieved_parents.values())

# ==========================================
# 4. MAIN FLOW
# ==========================================
def main():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print("--- 🧠 ADVANCED RAG: PARENT-CHILD ARCHITECTURE ---")
    print("1. Structured (Resumes, Manuals) - Uses Header Hierarchy")
    print("2. Unstructured (Essays, Articles) - Uses Semantic Topics")
    choice = input("Select Doc Type (1/2): ")

    # Step 1: Strategy Selection (How we define the 'Parent')
    if choice == "2":
        strategy = SemanticChunker(embedder)
    else:
        strategy = HierarchicalChunker()

    # Step 2: Processing
    raw_text = read_pdf(PDF_PATH)
    clean = clean_text(raw_text)
    
    # Generate PARENTS
    parent_chunks = strategy.chunk(clean)
    
    # Step 3: Indexing (Small-to-Big)
    indexer = ParentChildIndexer(embedder)
    indexer.add_parents(parent_chunks)

    # Step 4: Chat
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    print("\n✅ System Ready.\n")

    while True:
        query = input("🧑 You: ")
        if query.lower() in ["exit", "quit"]: break

        # Retrieve PARENTS via CHILDREN
        parents = indexer.retrieve(query)
        
        # Visualize for the user what's happening
        print(f"\n📦 Feeding {len(parents)} Parent Contexts to LLM (Full Sections)...\n")

        context_block = "\n\n".join(parents)
        
        prompt = f"""
        Answer based ONLY on the context below.
        
        Context:
        {context_block}
        
        Question: {query}
        """
        
        print("🤖 AI Response:")
        response = llm.invoke(prompt)
        print(response.content)
        print("-" * 60)

if __name__ == "__main__":
    main()