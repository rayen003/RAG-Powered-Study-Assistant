from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_KEY, PERSIST_DIRECTORY
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import List
from langchain_core.output_parsers.string import StrOutputParser





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
    )

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

def input_classifier(model, initial_query: str, categories: List[str]):
    # Input validation
    if not isinstance(initial_query, str):
        raise ValueError("initial_query must be a string.")
    if not isinstance(categories, list) or not all(isinstance(cat, str) for cat in categories):
        raise ValueError("categories must be a list of strings.")

    # Define the prompt template
    prompt = """
    You are a query classifier. Given the following query and a list of categories, assign the query to the most appropriate category.

    Query: {query}
    Categories: {categories}

    Return only the name of the category that best matches the query. Do not include any additional text.
    """
    prompt_template = PromptTemplate(template=prompt, input_variables=["query", "categories"])

    # Create the classification chain
    classification_chain = (
        {
            "query": RunnablePassthrough(),  # Pass the query directly
            "categories": lambda x: ", ".join(x["categories"])  # Format categories as a string
        }
        | prompt_template
        | model
    )

    # Run the chain
    selected_category = classification_chain.invoke({"query": initial_query, "categories": categories})

    # Validate the output
    if selected_category not in categories:
        raise ValueError(f"LLM returned an invalid category: {selected_category}")

    return selected_category

# Example usage
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
categories = ["General Processing", "Advanced Processing", "Basic Processing", "Data Analysis", "Content Generation", "Multimodal Processing"]
query = "How do I summarize this document?"
selected_category = input_classifier(model, query, categories)
print(f"Selected category: {selected_category}")