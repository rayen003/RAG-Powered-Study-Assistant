import streamlit as st
import os
import time
from utils import load_and_split_documents, create_memory
from rag import create_vector_store, create_rag_chain
from config import (
    OPENAI_API_KEY,
    PERSIST_DIRECTORY,
    MODEL_NAME,
    EMBEDDING_MODEL
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Initialize session state
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = create_memory()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

def format_chat_history(chat_history):
    """Format chat history for the LLM."""
    if not chat_history:
        return ""
    formatted = []
    for msg in chat_history:
        role = "Assistant" if msg["role"] == "assistant" else "Human"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def initialize_chain():
    """Initialize the smart chain with or without documents."""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )
    
    if st.session_state.documents_loaded:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        vectorstore = create_vector_store(
            st.session_state.document_chunks,
            embeddings
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    else:
        retriever = None
    
    st.session_state.chat_chain = create_rag_chain(
        retriever,
        llm,
        st.session_state.memory
    )

def main():
    st.title("AI Study Assistant")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("Study Materials")
        uploaded_file = st.file_uploader(
            "Upload a document (optional)",
            type="pdf",
            help="Upload a PDF to get context-aware responses"
        )
        
        if st.button("Clear Memory & Documents"):
            st.session_state.memory = create_memory()
            st.session_state.chat_history = []
            st.session_state.chat_chain = None
            st.session_state.documents_loaded = False
            if "document_chunks" in st.session_state:
                del st.session_state.document_chunks
            st.rerun()

        if uploaded_file:
            with st.spinner("Processing document..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.document_chunks = load_and_split_documents("temp.pdf")
                st.session_state.documents_loaded = True
                initialize_chain()
                st.success("✅ Document processed successfully!")

    # Initialize chain if not exists
    if st.session_state.chat_chain is None:
        initialize_chain()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your studies..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Show typing indicator
            message_placeholder.markdown("Typing...")
            time.sleep(0.5)

            # Format history and prepare inputs
            history = format_chat_history(st.session_state.chat_history[:-1])  # Exclude current prompt
            
            # Stream the response
            for chunk in st.session_state.chat_chain.stream({
                "input": prompt,
                "history": history
            }):
                if "text" in chunk:
                    full_response += chunk["text"]
                else:
                    full_response += chunk.get("response", "")
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)

            # Show final response
            message_placeholder.markdown(full_response)

        # Add assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()