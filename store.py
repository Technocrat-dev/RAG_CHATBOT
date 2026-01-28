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






































import pymupdf4llm
import chromadb
import uuid
import fitz  # PyMuPDF
import ollama
import io
from PIL import Image
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import ChatOllama

# ==========================================
# CONFIGURATION
# ==========================================
PDF_PATH = "sample.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"        # For text generation
VISION_MODEL = "llava"      # For image captioning (Make sure to 'ollama pull llava')

# ==========================================
# 1. MULTI-MODAL INGESTION
# ==========================================
class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def _extract_images_and_caption(self) -> str:
        """
        Scans PDF for images/charts, sends them to a Vision LLM, 
        and returns a Markdown formatted list of descriptions.
        """
        print(f"👁️  Scanning {self.pdf_path} for visual data (Charts/Diagrams)...")
        doc = fitz.open(self.pdf_path)
        captions = []
        
        total_images = 0
        for page_index, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                total_images += 1
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Filter small icons/logos to save time
                if len(image_bytes) < 5000: continue 

                print(f"   • Processing Image {img_index+1} on Page {page_index+1}...")
                
                try:
                    # Send image bytes directly to Ollama Vision Model
                    response = ollama.chat(
                        model=VISION_MODEL,
                        messages=[{
                            'role': 'user',
                            'content': 'Analyze this image. If it is a chart or graph, detail the data values, percentages, and labels. If it is a diagram, explain the flow.',
                            'images': [image_bytes]
                        }]
                    )
                    description = response['message']['content']
                    captions.append(f"### Figure on Page {page_index+1}\n**Visual Description:** {description}\n")
                except Exception as e:
                    print(f"   ⚠️ Failed to caption image: {e}")

        print(f"✅ Processed {total_images} images. Generated {len(captions)} detailed captions.")
        return "\n\n".join(captions)

    def process(self) -> str:
        # 1. Convert Text/Tables to Markdown (Preserves Structure)
        print(f"📄 Converting PDF Layout to Markdown...")
        md_text = pymupdf4llm.to_markdown(self.pdf_path)
        
        # 2. Generate Image Captions
        visual_context = ""
        
        # 3. Merge: We append visual context at the end so it's retrievable
        full_document = f"{md_text}\n\n# APPENDIX: VISUAL DATA & CHARTS\n{visual_context}"
        return full_document

# ==========================================
# 2. STRUCTURED CHUNKING (Markdown)
# ==========================================
class MarkdownStructChunker:
    """
    Splits content based on Document Headers (#, ##) rather than character count.
    This keeps tables and related sections intact.
    """
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        print("🧱 Chunking by Document Structure (Headers)...")
        
        # Define the hierarchy to split on
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
            # Metadata tracking allows us to cite the section in the answer
            header_path = " > ".join(doc.metadata.values())
            
            # We treat these semantic sections as "Parents"
            chunks.append({
                "text": f"CONTEXT SOURCE: {header_path}\n\n{doc.page_content}",
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
        self.chroma_client = chromadb.Client()
        
        # Reset DB for fresh run
        try: self.chroma_client.delete_collection("child_nodes")
        except: pass
        
        self.collection = self.chroma_client.create_collection("child_nodes")

    def split_into_children(self, text: str) -> List[str]:
        """Sliding window for child chunks"""
        words = text.split()
        children = []
        # Window size 100 words, step 50 words
        for i in range(0, len(words), 50): 
            chunk = " ".join(words[i : i + 100])
            if len(chunk) > 20:
                children.append(chunk)
            if i + 100 >= len(words): break
        return children

    def add_parents(self, parent_chunks: List[Dict[str, Any]]):
        print("👶 Generating Children & Indexing...")
        
        child_texts = []
        child_metadatas = []
        child_ids = []

        for p_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent_text = p_chunk["text"]
            
            # Store Parent in Memory
            self.parent_store[parent_id] = parent_text

            # Create Children
            children = self.split_into_children(parent_text)

            for i, child_text in enumerate(children):
                child_texts.append(child_text)
                child_metadatas.append({"parent_id": parent_id, "child_index": i})
                child_ids.append(f"{parent_id}_{i}")

        # Batch Add
        if child_texts:
            # Process in batches of 100 to avoid hitting API limits/Timeouts
            batch_size = 100
            for i in range(0, len(child_texts), batch_size):
                batch_texts = child_texts[i:i+batch_size]
                batch_metas = child_metadatas[i:i+batch_size]
                batch_ids = child_ids[i:i+batch_size]
                
                embeddings = self.embedder.encode(batch_texts)
                self.collection.add(
                    documents=batch_texts,
                    embeddings=embeddings.tolist(),
                    metadatas=batch_metas,
                    ids=batch_ids
                )
        print(f"✅ Indexed {len(child_texts)} child nodes pointing to {len(self.parent_store)} parent contexts.")

    def retrieve(self, query: str, top_k=3):
        query_vec = self.embedder.encode([query])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k * 3 # Fetch more children to ensure unique parents
        )

        retrieved_parents = {}
        print("\n🔎 Retrieval Path:")
        
        for i, meta in enumerate(results["metadatas"][0]):
            parent_id = meta["parent_id"]
            if parent_id not in retrieved_parents:
                # Preview the match
                match_preview = results["documents"][0][i][:60].replace('\n', ' ')
                print(f"   • Match: '...{match_preview}...' -> Parent ID: {parent_id[:8]}")
                retrieved_parents[parent_id] = self.parent_store[parent_id]
            
            if len(retrieved_parents) >= top_k:
                break
        
        return list(retrieved_parents.values())

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("--- 🛠️  Advanced Technical RAG (Markdown + Vision) ---")
    
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
    print("\n✅ System Ready. Ask about tables, charts, or technical specs.\n")

    while True:
        query = input("🧑 You: ")
        if query.lower() in ["exit", "quit"]: break

        parents = indexer.retrieve(query, top_k=3)
        
        context_block = "\n---\n".join(parents)
        
        prompt = f"""
        You are a technical analyst assistant. Answer the question based ONLY on the provided context.
        The context includes Markdown tables and Image Descriptions.
        
        Rules:
        1. If reading a table, look at column headers carefully.
        2. If the answer is in a chart, refer to the "Visual Description".
        3. If you don't know, say "Data not found in document".
        
        Context:
        {context_block}
        
        Question: {query}
        """
        
        print("\n🤖 AI Response:")
        response_stream = llm.stream(prompt)
        for chunk in response_stream:
            print(chunk.content, end="", flush=True)
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()