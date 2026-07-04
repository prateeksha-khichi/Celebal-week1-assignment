import time
import json
import os

from document_loaders.loaders import load_directory
from chunking.splitter import TextSplitter
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore

def main():
    print("Starting Database Creation Pipeline with LangChain...")
    
    # 1. Load Documents
    print("Loading documents from data/...")
    documents = load_directory("data")
    if not documents:
        print("No documents found in data/ directory. Please add some .txt files.")
        return
        
    print(f"Loaded {len(documents)} documents.")
    
    # 2. Chunking
    print("Chunking documents...")
    start_chunk = time.time()
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(documents)
    chunk_time = time.time() - start_chunk
    print(f"Created {len(chunked_docs)} chunks.")
    
    # 3. Embeddings
    print("Generating embeddings...")
    start_embed = time.time()
    embedder = Embedder()
    # Notice we don't need to manually map [doc['text']] because FAISS takes documents
    # But for the log, we just measure setup time
    embed_time = time.time() - start_embed
    print(f"Embeddings setup in {embed_time:.2f} seconds.")
    
    # 4. Vector Store
    print("Saving to FAISS Vector Database...")
    start_db = time.time()
    store = FAISSStore(dimension=embedder.dimension)
    store.add_documents(chunked_docs, embedder.embeddings)
    store.save("faiss_index")
    db_time = time.time() - start_db
    print("Database saved successfully to faiss_index directory")
    
    # Save some metrics for the report generation later
    metrics = {
        "chunk_metrics": splitter.get_metrics(),
        "embed_metrics": embedder.get_info(),
        "db_metrics": {"type": "FAISS", "metric": "L2 Distance", "top_k": 5},
        "perf_metrics": {"ingestion_time": chunk_time + db_time}
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/ingestion_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
