# Document Question Answering System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built for academic demonstration. This project emphasizes modularity, explainability, logging, and evaluation without relying on opaque monolithic frameworks.

## 1. Project Overview
This project implements a system capable of answering user queries based on local documents (PDFs and Text files). It grounds Language Model generation in retrieved facts to mitigate hallucinations.

## 2. Objectives
- Ingest and chunk custom documents cleanly.
- Convert chunks into vector embeddings.
- Perform similarity search using a Vector Database (FAISS).
- Experiment with Hybrid Search (BM25 Keyword + Vector Search).
- Generate answers using an LLM (Groq API).
- Provide detailed logging and evaluation metrics.

## 3. System Architecture
```text
[Documents] -> [Loaders] -> [Chunker] -> [Embedder] -> [Vector DB]
                                                             |
[User Query] -> [Embedder] -> [Hybrid Retriever (BM25 + Vector)] -> [Context]
                                                                        |
                                         [LLM Generator] <---------------+
                                                |
                                         [Final Answer]
```

## 4. Components Used
- **Document Loaders**: `PyPDF2`, standard Python I/O
- **Chunking**: Custom Sliding Window
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: `faiss-cpu`
- **Keyword Search**: `rank_bm25`
- **LLM API**: `groq` (Llama3-8b)

## 5. Workflow
1. **Ingestion (`create_database.py`)**: Reads data, chunks text, generates embeddings, and saves to a local FAISS index.
2. **Querying (`main.py`)**: Loads index, receives user query, retrieves top chunks using Reciprocal Rank Fusion, constructs a prompt, and streams to LLM.

## 6. Installation Steps
1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Run `pip install -r requirements.txt`.
4. Open `.env` and add your Groq API Key: `GROQ_API_KEY=your_key_here`.

## 7. Example Outputs & Evaluation
When you run `main.py`, the system logs the exact query, the chunks retrieved (with similarity scores), the generated answer, and latency to `logs/evaluation.log`.

## 8. Improvement Experiments
This project implements **Hybrid Retrieval** as its primary experiment. By combining FAISS (semantic search) and BM25 (keyword search), the retriever is robust against both conceptual queries and specific keyword lookups.

## 9. Key Learnings
- **Chunking matters**: Overlapping chunks prevents cutting off context mid-sentence.
- **RRF (Reciprocal Rank Fusion)** is an effective way to combine scores from different retrieval algorithms without complex normalization.

## 10. Future Work
- Implementing a Cross-Encoder for the re-ranking stage.
- Adding a simple Streamlit UI.
- Experimenting with different embedding models and dimension sizes.

## 11. Glossary of Advanced RAG Terms
- **Advanced RAG**: Techniques that go beyond simple "retrieve and generate" pipelines, incorporating methods like query transformations, hybrid search, re-ranking, and dynamic chunking to improve the accuracy and relevance of the system.
- **Latency**: The time taken from when a user submits a query to when the system returns the final response. In a RAG system, this includes retrieval latency (searching the database) and generation latency (the LLM computing the answer).
- **Quantization**: A model optimization technique that reduces the precision of the numbers used to represent a model's parameters (e.g., from 32-bit floating point to 8-bit integers). This significantly reduces memory usage and improves inference speed (lower latency) while maintaining most of the model's performance.
- **Hybrid Search**: A retrieval approach that combines traditional keyword-based search (like BM25) with semantic vector search (like FAISS) to ensure robust retrieval for both exact matches and conceptually similar text.
- **Re-ranking / Cross-Encoders**: An advanced RAG step where an initial broad set of retrieved documents is scored and re-ordered by a more accurate (but slower) model to ensure the most relevant context is fed to the LLM.
- **Reciprocal Rank Fusion (RRF)**: An algorithm used to combine ranked lists from different retrieval methods (like BM25 and vector search) without needing to normalize their individual scores.
- **Embeddings**: High-dimensional numerical vectors that represent the semantic meaning of text, allowing the system to find conceptually similar chunks of text.
