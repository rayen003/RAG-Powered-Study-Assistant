from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_KEY, PERSIST_DIRECTORY
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

def create_vector_store(chunks, embeddings, persist_directory=PERSIST_DIRECTORY):
    """
    Create a vector store from the document chunks.
    """
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)


def create_memory():
    """
    Create a conversation summary memory.
    """
    return ConversationSummaryMemory(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )º

def create_rag_chain(retriever, memory, llm):
    """
    Create a RAG chain using the retriever and LLM.
    """
    template = """
    You are a helpful study assistant. Given the following context and chat history, answer the user's question in a clear and concise manner.

    Previous conversation:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

    # Create the RAG chain with proper memory integration
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join(doc.page_content for doc in docs)),
            "chat_history": memory.load_memory_variables | (lambda x: x["chat_history"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return rag_chain