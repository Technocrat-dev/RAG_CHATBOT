import pymupdf4llm
import chromadb
import uuid
import fitz  # PyMuPDF
import ollama
import re  # <--- ADDED THIS
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import ChatOllama

# ==========================================
# CONFIGURATION
# ==========================================
PDF_PATH = "data/sample.pdf" # Make sure this matches your file path
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"        
VISION_MODEL = "llava"      

# ==========================================
# 1. MULTI-MODAL INGESTION
# ==========================================
class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def _extract_images_and_caption(self) -> str:
        """
        Scans PDF for images/charts, sends them to a Vision LLM.
        """
        # (Keeping this logic same as before to save space, assuming it works)
        # For this test, we can focus on the text fix.
        return "" 

    def process(self) -> str:
        # 1. Convert Text/Tables to Markdown
        print(f"📄 Converting PDF Layout to Markdown...")
        md_text = pymupdf4llm.to_markdown(self.pdf_path)
        
        # --- 🛠️ FIX START: HEADER PROMOTION ---
        print("🧹 Cleaning and Promoting Headers...")
        
        # Rule 1: Fix broken lines inside words (e.g. "Chemito<br>Tech" -> "Chemito Tech")
        md_text = md_text.replace("<br>", " ")
        md_text = md_text.replace("- \n", "")
        
        # Rule 2: Promote Bold Numbered Sections to Real Headers
        # Finds: "**2.2 Data Used for Analysis:**"
        # Turns into: "## 2.2 Data Used for Analysis"
        # Regex explanation: Look for line start (^), optional space, bold (**), number, text, close bold
        pattern = r"^\s*\*\*\s*(\d+(\.\d+)+)\s+(.*?)\s*\*\*.*$"
        md_text = re.sub(pattern, r"## \1 \3", md_text, flags=re.MULTILINE)
        
        # --- FIX END ---
        
        # 2. Merge (Visuals disabled for speed in this test)
        full_document = md_text
        return full_document

# ==========================================
# 2. STRUCTURED CHUNKING (Markdown)
# ==========================================
class MarkdownStructChunker:
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        print("🧱 Chunking by Document Structure (Headers)...")
        
        # Added "Header 2" (##) to catch our newly promoted headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False 
        )
        
        docs = splitter.split_text(text)
        
        chunks = []
        for doc in docs:
            # Create a clean breadcrumb path (e.g., "Evaluation > 2.2 Data Used")
            header_path = " > ".join(doc.metadata.values())
            
            chunks.append({
                "text": f"SECTION: {header_path}\n\n{doc.page_content}",
                "metadata": {
                    "type": "parent", 
                    "headers": str(doc.metadata)
                }
            })
            
        print(f"   • Created {len(chunks)} logical parent chunks.")
        return chunks

# ==========================================
# 3. PARENT-CHILD INDEXER
# ==========================================
class ParentChildIndexer:
    def __init__(self, embedder):
        self.embedder = embedder
        self.parent_store = {} 
        self.chroma_client = chromadb.PersistentClient(path="chroma_db_practice") # Using disk storage
        self.collection = self.chroma_client.get_or_create_collection("child_nodes_practice")

    def split_into_children(self, text: str) -> List[str]:
        words = text.split()
        children = []
        # Increased window slightly to catch more context
        for i in range(0, len(words), 50): 
            chunk = " ".join(words[i : i + 120]) 
            if len(chunk) > 20:
                children.append(chunk)
        return children

    def add_parents(self, parent_chunks: List[Dict[str, Any]]):
        print(f"👶 Generating Children from {len(parent_chunks)} parents...")
        
        # Clear old data for this test run to ensure we use the clean version
        try:
            current_count = self.collection.count()
            if current_count > 0:
                print("   (Clearing old database data...)")
                self.chroma_client.delete_collection("child_nodes_practice")
                self.collection = self.chroma_client.create_collection("child_nodes_practice")
        except: pass

        child_texts = []
        child_metadatas = []
        child_ids = []

        for p_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            self.parent_store[parent_id] = p_chunk["text"]

            children = self.split_into_children(p_chunk["text"])

            for i, child_text in enumerate(children):
                child_texts.append(child_text)
                child_metadatas.append({"parent_id": parent_id, "child_index": i})
                child_ids.append(f"{parent_id}_{i}")

        # Batch Add
        batch_size = 100
        for i in range(0, len(child_texts), batch_size):
            self.collection.add(
                documents=child_texts[i:i+batch_size],
                embeddings=self.embedder.encode(child_texts[i:i+batch_size]).tolist(),
                metadatas=child_metadatas[i:i+batch_size],
                ids=child_ids[i:i+batch_size]
            )
        print(f"✅ Indexed {len(child_texts)} child nodes.")

    def retrieve(self, query: str, top_k=3):
        query_vec = self.embedder.encode([query])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k * 2 
        )

        retrieved_parents = {}
        for i, meta in enumerate(results["metadatas"][0]):
            parent_id = meta["parent_id"]
            if parent_id not in retrieved_parents and parent_id in self.parent_store:
                retrieved_parents[parent_id] = self.parent_store[parent_id]
            if len(retrieved_parents) >= top_k:
                break
        
        return list(retrieved_parents.values())

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("--- 🛠️  Practice Bot (Header Fixing Edition) ---")
    
    # 1. Processing
    processor = DocumentProcessor(PDF_PATH)
    full_markdown = processor.process()
    
    # 2. Chunking
    chunker = MarkdownStructChunker()
    parent_chunks = chunker.chunk(full_markdown)

    # 3. Indexing
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    indexer = ParentChildIndexer(embedder)
    indexer.add_parents(parent_chunks)

    # 4. Chat Loop
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    print("\n✅ System Ready. Try asking: 'What is 2.2 Data Used for Analysis?'\n")

    while True:
        query = input("🧑 You: ")
        if query.lower() in ["exit", "quit"]: break

        parents = indexer.retrieve(query, top_k=3)
        context_block = "\n---\n".join(parents)
        
        prompt = f"""
        You are a technical analyst assistant. Answer based ONLY on the context.
        
        CONTEXT:
        {context_block}
        
        QUESTION: {query}
        """
        
        print("\n🤖 AI Response:")
        try:
            response_stream = llm.stream(prompt)
            for chunk in response_stream:
                print(chunk.content, end="", flush=True)
            print("\n" + "-" * 60)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()