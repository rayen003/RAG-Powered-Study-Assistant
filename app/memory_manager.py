from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

def create_memory():
    """
    Create a conversation summary memory.
    """
    return ConversationSummaryMemory(
        llm=ChatOpenAI(),
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

class MemoryManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.memory = create_memory()
        return cls._instance
    
    @classmethod
    def get_memory(cls):
        return cls().memory 