from langchain_community.vectorstores import FAISS

class FAISSStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.store = None
        
    def add_documents(self, chunked_docs, embeddings):
        """
        Create a new FAISS vector store from documents and embeddings
        """
        self.store = FAISS.from_documents(chunked_docs, embeddings)
        
    def save(self, path="faiss_index"):
        if self.store:
            self.store.save_local(path)
            
    def load(self, embeddings, path="faiss_index"):
        self.store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        
    @property
    def current_id(self):
        if self.store:
            return len(self.store.index_to_docstore_id)
        return 0
