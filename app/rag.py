from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from config import OPENAI_API_KEY, PERSIST_DIRECTORY, MODEL_NAME, EMBEDDING_MODEL

# Similarity threshold for considering document context
SIMILARITY_THRESHOLD = 0.7

def create_vector_store(chunks, embeddings, persist_directory=PERSIST_DIRECTORY):
    """Create a vector store from the document chunks."""
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vectorstore

def create_rag_chain(retriever=None, llm=None):
    """Create a conversational retrieval chain."""
    # If no LLM is provided, create a default one
    if llm is None:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
    
    # Create a conversational prompt template
    prompt_template = """You are a helpful study assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # If no retriever is provided, create a chain without retrieval
    if retriever is None:
        # Simple chain without document retrieval
        def simple_predict(input):
            response = llm.predict(input)
            return {"text": response}
        
        simple_predict.predict = simple_predict
        return simple_predict
    
    # Create conversational retrieval chain with the retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    return chain