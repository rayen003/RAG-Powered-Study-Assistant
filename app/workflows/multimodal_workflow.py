from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from app.templates import TEMPLATES
from ..config import MODEL_NAME  # Import MODEL_NAME instead of API key

def multimodal_workflow(memory, question, file_input=None):
    llm = ChatOpenAI(model=MODEL_NAME)  # Use model name from config

    if file_input:
        try:
            # For text files, decode the content
            if file_input.mime_type.startswith('text/'):
                file_content = file_input.content.decode('utf-8')
                context = f"File content:\n{file_content}"
            else:
                context = "File type not fully supported yet."
            
            chain = (
                {
                    "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
                    "question": RunnablePassthrough(),
                    "context": lambda _: context
                }
                | TEMPLATES["multimodal"]
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(question)
            
            # Save to memory
            memory.save_context(
                {"question": question},
                {"answer": response}
            )
            
            return {
                "answer": response,
                "chat_history": memory.load_memory_variables({}).get("chat_history", "")
            }
                
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {
                "answer": f"Error processing file: {str(e)}",
                "chat_history": memory.load_memory_variables({}).get("chat_history", "")
            }
    
    # If no file input, use standard conversation
    chain = (
        {
            "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
            "question": RunnablePassthrough()
        }
        | TEMPLATES["general"]
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    memory.save_context(
        {"question": question},
        {"answer": response}
    )
    
    return {
        "answer": response,
        "chat_history": memory.load_memory_variables({}).get("chat_history", "")
    } 