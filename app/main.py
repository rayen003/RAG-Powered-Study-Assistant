import streamlit as st
import os
from utils import load_and_split_documents
from rag import create_vector_store, create_rag_chain
from config import (
    OPENAI_API_KEY,
    PERSIST_DIRECTORY,
    MODEL_NAME,
    EMBEDDING_MODEL
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

def create_simple_chain():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, openai_api_key=OPENAI_API_KEY)
    return lambda x: {"text": llm.predict(x["input"])}

# Streamlit app
def main():
    # Set up the Streamlit app
    st.title("Personalized Study Assistant")
    st.sidebar.header("📎 Attach")

    # Custom CSS to style the file uploader
    st.markdown("""
    <style>
    .stFileUploader {
        display: flex;
        align-items: center;
    }
    .stFileUploader label {
        background-color: transparent;
        color: grey;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stFileUploader label:hover {
        color: blue;
    }
    </style>
    """, unsafe_allow_html=True)

    # File uploader with attachment icon
    uploaded_file = st.sidebar.file_uploader(
        "Attach", 
        type="pdf", 
        label_visibility="collapsed",
        help="Upload a PDF to get context-aware responses"
    )

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG chain (optional)
    if "rag_chain" not in st.session_state:
        # Create a default LLM chain even without a document
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            temperature=0.7, 
            openai_api_key=OPENAI_API_KEY
        )
        st.session_state.rag_chain = create_rag_chain(llm=llm)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        # Immediately display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")

            # Generate response in the background
            try:
                # For the default chain, just use the input directly
                response = st.session_state.rag_chain({"input": user_input})
                response_text = response.get('text', str(response))
                
                # Update the placeholder with the actual response
                response_placeholder.markdown(response_text)

                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text
                })
            except Exception as e:
                response_placeholder.markdown(f"Sorry, I encountered an error: {str(e)}")

    # File processing
    if uploaded_file:
        with st.spinner("Processing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and split the document
            chunks = load_and_split_documents("temp.pdf")

            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL, 
                openai_api_key=OPENAI_API_KEY
            )

            # Create vector store
            vectorstore = create_vector_store(chunks, embeddings)

            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Reinitialize RAG chain with the new retriever
            llm = ChatOpenAI(
                model=MODEL_NAME, 
                temperature=0.7, 
                openai_api_key=OPENAI_API_KEY
            )
            st.session_state.rag_chain = create_rag_chain(retriever, llm)

            st.sidebar.success("File uploaded and processed successfully!")

# Run the Streamlit app
if __name__ == "__main__":
    main()
