from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        # 384 is the dimension for all-MiniLM-L6-v2
        self.dimension = 384 
        
    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)
        
    def get_info(self):
        return {
            "model": self.model_name,
            "dimension": self.dimension
        }
