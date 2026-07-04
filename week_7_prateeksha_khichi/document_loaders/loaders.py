from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

def load_directory(path: str):
    """
    Load all text and PDF documents from the given directory using LangChain.
    """
    # Load TXT files
    txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, loader_kwargs={'encoding': 'utf-8'})
    txt_documents = txt_loader.load()
    
    # Load PDF files
    pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
    pdf_documents = pdf_loader.load()
    
    return txt_documents + pdf_documents
