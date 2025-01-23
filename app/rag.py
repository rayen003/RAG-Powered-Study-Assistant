from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_KEY, PERSIST_DIRECTORY

def create_vector_store(chunks, embeddings, persist_directory=PERSIST_DIRECTORY):
    """
    Create a vector store from the document chunks.
    """
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vectorstore

def create_rag_chain(retriever, llm):
    """
    Create a RAG chain using the retriever and LLM.
    """
    # Define the prompt template
    template = """
    You are a helpful study assistant. Given the following context, answer the user's question in a clear and concise manner.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the RAG chain
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm)
    return rag_chain