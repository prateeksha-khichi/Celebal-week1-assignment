import logging
import os


def setup_logger(log_file="logs/evaluation.log"):
    """
    Configures a logger to record query evaluations and system events.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("RAG_Evaluator")
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        
    return logger

def log_query_evaluation(logger, question, retrieved_chunks, answer, response_time):
    """
    Logs the evaluation of a single query as required by the project specifications.
    """
    log_msg = f"\n{'='*50}\n"
    log_msg += f"Question:\n{question}\n\n"
    
    log_msg += "Retrieved Chunks:\n"
    for i, chunk in enumerate(retrieved_chunks):
        # Handle different return formats from vector (tuple) vs hybrid (dict)
        if isinstance(chunk, tuple):
            doc, score = chunk
        else:
            doc = chunk
            score = "N/A (Hybrid/BM25)"
            
        chunk_id = doc['metadata'].get('chunk_id', 'Unknown')
        source = doc['metadata'].get('source', 'Unknown')
        
        log_msg += f"  - Chunk ID: {chunk_id}\n"
        log_msg += f"    Similarity Score: {score}\n"
        log_msg += f"    Source File: {source}\n"
        log_msg += f"    Content Snippet: {doc['text'][:100]}...\n"
        
    log_msg += f"\nGenerated Answer:\n{answer}\n\n"
    log_msg += f"Response Time:\n{response_time:.4f} seconds\n"
    log_msg += f"{'='*50}\n"
    
    logger.info(log_msg)
