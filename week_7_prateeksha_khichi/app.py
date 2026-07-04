import streamlit as st
import time
import os
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from retrievers.hybrid import HybridRetriever
from generation.llm import LLMGenerator

# Load environment variables
load_dotenv()

# Basic premium styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #BB86FC;
        font-weight: 600;
    }
    
    /* Input Box */
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: #FFF;
        border-radius: 8px;
        border: 1px solid #333;
    }
    
    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #BB86FC 0%, #3700B3 100%);
        color: white;
        padding: 15px;
        border-radius: 12px 12px 0 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-bot {
        background-color: #1E1E1E;
        color: #E0E0E0;
        padding: 15px;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 20px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #000;
        border-left: 4px solid #BB86FC;
        padding: 10px;
        font-size: 0.85em;
        color: #AAA;
        margin-top: 10px;
        overflow-y: auto;
        max-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ RAG Assistant")
st.markdown("Ask anything based on the *Attention Is All You Need* paper.")

@st.cache_resource
def load_rag_pipeline():
    with st.spinner("Initializing AI Models and Vector DB... (This might take a moment)"):
        # Embeddings
        embedder = Embedder()
        
        # Vector Store
        store = FAISSStore(dimension=embedder.dimension)
        try:
            store.load(embedder.embeddings, path="faiss_index")
        except Exception:
            return None, "FAISS index not found. Please run fetch_data.py and create_database.py first."
            
        # Retriever
        hybrid_retriever = HybridRetriever(store, embedder)
        retriever = hybrid_retriever.get_retriever(top_k=3)
        
        # Generator
        try:
            generator = LLMGenerator(model_name="llama-3.1-8b-instant", temperature=0.0)
            rag_chain = generator.get_chain(retriever)
        except Exception as e:
            return None, f"Error initializing LLM: {e}"
            
        return (retriever, rag_chain), "Success"

pipeline, status = load_rag_pipeline()

if pipeline is None:
    st.error(status)
    st.stop()

retriever, rag_chain = pipeline

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-user"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        html_content = f'<div class="chat-bot"><strong>RAG Assistant:</strong><br>{message["content"]}'
        if "context" in message and message["context"]:
            context_str = "\n\n".join([doc.page_content for doc in message["context"]])
            html_content += f'<div class="context-box"><strong>Retrieved Context:</strong><br>{context_str}</div>'
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    st.markdown(f'<div class="chat-user"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process RAG
    with st.spinner("Thinking..."):
        start_time = time.time()
        
        # Get context
        retrieved_contexts_docs = retriever.invoke(prompt)
        
        # Get Answer
        answer = rag_chain.invoke(prompt)
        
        # Display assistant response
        html_content = f'<div class="chat-bot"><strong>RAG Assistant:</strong><br>{answer}'
        if retrieved_contexts_docs:
            context_str = "\n\n".join([doc.page_content for doc in retrieved_contexts_docs])
            html_content += f'<div class="context-box"><strong>Retrieved Context:</strong><br>{context_str}</div>'
        html_content += '</div>'
        
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer, "context": retrieved_contexts_docs})
