from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder
        self.bm25_retriever = None
        self.faiss_retriever = None
        
    def fit_bm25(self, documents):
        """
        Initializes the BM25 retriever with the documents.
        """
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        
    def _hybrid_search(self, query: str):
        if not self.store.store:
            raise ValueError("FAISS store not loaded.")
        if not self.faiss_retriever:
            self.faiss_retriever = self.store.store.as_retriever(search_kwargs={"k": 3})
            
        # Get FAISS results
        faiss_docs = self.faiss_retriever.invoke(query)
        
        # Get BM25 results
        if self.bm25_retriever:
            self.bm25_retriever.k = 3
            bm25_docs = self.bm25_retriever.invoke(query)
        else:
            bm25_docs = []
            
        # Combine and deduplicate
        combined_docs = faiss_docs + bm25_docs
        unique_docs = {}
        for doc in combined_docs:
            # use page_content as key to deduplicate
            unique_docs[doc.page_content] = doc
            
        # Return top 4 unique documents (a simple hybrid merge strategy)
        return list(unique_docs.values())[:4]
        
    def get_retriever(self, top_k=3):
        """
        Returns a LCEL-compatible RunnableLambda for hybrid search.
        """
        return RunnableLambda(self._hybrid_search)
