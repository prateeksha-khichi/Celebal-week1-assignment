import time
import json
import os
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from retrievers.hybrid import HybridRetriever
from generation.llm import LLMGenerator

from evaluation.logger import setup_logger, log_query_evaluation
from reports.metrics import generate_system_metrics_report

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

def main():
    print("Starting RAG Query Pipeline with LangChain...")
    
    # Setup Logger
    logger = setup_logger()
    
    # Load Components
    print("Loading components...")
    
    # Embeddings
    embedder = Embedder()
    
    # Load FAISS index
    store = FAISSStore(dimension=embedder.dimension)
    try:
        store.load(embedder.embeddings, path="faiss_index")
    except Exception as e:
        print("Could not load FAISS index. Did you run create_database.py first?")
        return
        
    print(f"Loaded FAISS database with {store.current_id} chunks.")
    
    # Setup Retrievers
    from document_loaders.loaders import load_directory
    from chunking.splitter import TextSplitter
    
    hybrid_retriever = HybridRetriever(store, embedder)
    
    try:
        # Load documents to fit BM25
        documents = load_directory("data")
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunked_docs = splitter.split_documents(documents)
        hybrid_retriever.fit_bm25(chunked_docs)
    except Exception as e:
        print(f"Warning: BM25 not fitted. Reason: {e}")
        
    retriever = hybrid_retriever.get_retriever(top_k=3)
    
    # Setup Generator
    try:
        generator = LLMGenerator(model_name="llama3-8b-8192", temperature=0.0)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # Create Chains
    rag_chain = generator.get_chain(retriever)

    # Track metrics for report
    retrieval_latencies = []
    generation_latencies = []
    
    # Sample Questions (User can change these interactively)
    questions = [
        "What is Retrieval-Augmented Generation?",
        "What are the benefits of using a vector database?"
    ]
    
    print("\n--- Running Queries ---")
    
    for q in questions:
        print(f"\nQ: {q}")
        start_time = time.time()
        
        # 1. Retrieval
        start_ret = time.time()
        retrieved_contexts_docs = retriever.invoke(q)
        ret_time = time.time() - start_ret
        retrieval_latencies.append(ret_time)
        
        # Convert Document objects to text for logging compatibility
        retrieved_contexts = [doc.page_content for doc in retrieved_contexts_docs]
        
        # 2. Generation (End-to-End actually)
        start_gen = time.time()
        answer = rag_chain.invoke(q)
        # Approximate gen time (total time - retrieval time)
        gen_time = (time.time() - start_gen) - ret_time
        if gen_time < 0: gen_time = time.time() - start_gen # fallback if invoke does retrieval internally again
        generation_latencies.append(gen_time)
        
        total_time = time.time() - start_time
        
        print(f"A: {answer}")
        print(f"[Retrieved in {ret_time:.2f}s, Generated in {gen_time:.2f}s]")
        
        # 3. Logging Evaluation
        log_query_evaluation(logger, q, retrieved_contexts, answer, total_time)
        
    # Generate System Metrics Report
    print("\nGenerating System Metrics Report...")
    try:
        with open("reports/ingestion_metrics.json", "r") as f:
            metrics = json.load(f)
            
        perf_metrics = metrics["perf_metrics"]
        perf_metrics["avg_retrieval_latency"] = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0
        perf_metrics["avg_generation_latency"] = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0
        perf_metrics["avg_e2e_latency"] = (sum(retrieval_latencies) + sum(generation_latencies)) / len(retrieval_latencies) if retrieval_latencies else 0
        
        llm_metrics = {"model_name": llm.model_name, "temperature": llm.temperature}
        
        generate_system_metrics_report(
            chunk_metrics=metrics["chunk_metrics"],
            embed_metrics=metrics["embed_metrics"],
            db_metrics=metrics["db_metrics"],
            llm_metrics=llm_metrics,
            perf_metrics=perf_metrics
        )
    except FileNotFoundError:
        print("reports/ingestion_metrics.json not found. Run create_database.py first to generate full report.")

if __name__ == "__main__":
    main()
