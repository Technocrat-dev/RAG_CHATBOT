import os
from handlers.technical import TechnicalHandler
from database import VectorDB

def run_test():
    # --- CONFIGURATION ---
    # Change this to match your actual file name inside the 'data' folder
    TEST_FILE = "data/sample.pdf" 
    
    print("🚀 Starting Factory Test...")

    # 1. Check if file exists
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: File '{TEST_FILE}' not found!")
        print("Please move your PDF into the 'data' folder.")
        return

    # 2. Initialize the Workers
    # The Librarian (Database)
    print("📚 Initializing Database...")
    db = VectorDB(collection_name="test_factory_run")
    
    # The Chef (Technical Handler)
    print("👨‍🍳 Initializing Technical Handler...")
    handler = TechnicalHandler()
    print(f"   (Type: {handler.get_type_name()})")

    # 3. Run the Process (Ingest -> Chunk)
    print("\n--- PHASE 1: PROCESSING ---")
    markdown_text = handler.ingest(TEST_FILE)
    parent_chunks = handler.chunk(markdown_text)
    
    print(f"✅ Generated {len(parent_chunks)} Parent Chunks.")
    print(f"   Sample Header: {parent_chunks[0].get('metadata')}")

    # 4. Store in Database
    print("\n--- PHASE 2: STORAGE ---")
    db.add_documents(parent_chunks)

    # 5. Test Retrieval
    print("\n--- PHASE 3: RETRIEVAL TEST ---")
    query = "What is this document about?" 
    # You can change this query to something specific to your PDF
    
    print(f"❓ Asking: '{query}'")
    results = db.retrieve(query, top_k=2)

    print("\n📝 RESULTS:")
    if not results:
        print("❌ No results found. Something is wrong.")
    else:
        for i, text in enumerate(results):
            print(f"\n[Result {i+1}]")
            # Print just the first 300 characters to keep it clean
            print(text[:300].replace("\n", " ") + "...") 

if __name__ == "__main__":
    run_test()