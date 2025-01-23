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

# Streamlit app
def main():
    # Set up the Streamlit app
    st.title("Personalized Study Assistant")
    st.sidebar.header("Upload a PDF")

    # Step 1: Upload a PDF file
    uploaded_file = st.sidebar.file_uploader("Drag and drop a PDF file", type="pdf")

    # Initialize session state to store the RAG chain and chat history
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Step 2: Process the uploaded file
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split the document
        chunks = load_and_split_documents("temp.pdf")

        # Initialize embeddings and create the vector store
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        vectorstore = create_vector_store(chunks, embeddings)

        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize the LLM
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, openai_api_key=OPENAI_API_KEY)

        # Create the RAG chain and store it in session state
        st.session_state.rag_chain = create_rag_chain(retriever, llm)

        st.sidebar.success("File uploaded and processed successfully!")

    # Step 3: Chat interface
    st.header("Chat with the Study Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    if st.session_state.rag_chain is not None:
        user_input = st.chat_input("Ask a question about the document...")
        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get the response from the RAG chain
            response = st.session_state.rag_chain.invoke(user_input)

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})

            # Rerun the app to update the chat interface
            st.rerun()
    else:
        st.warning("Please upload a PDF file to get started.")

# Run the Streamlit app
if __name__ == "__main__":
    main()