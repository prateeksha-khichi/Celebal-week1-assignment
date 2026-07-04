import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.processing_time = 0
        
    def split_documents(self, documents):
        start_time = time.time()
        chunked_docs = self.splitter.split_documents(documents)
        self.processing_time = time.time() - start_time
        return chunked_docs
        
    def get_metrics(self):
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
