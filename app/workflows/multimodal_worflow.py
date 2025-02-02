from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from app.templates import TEMPLATES
from ..config import OPENAI_API_KEY

# OpenAI API Configuration
OPENAI_API_KEY = "sk-g_X87viKVTK3qt5IIkTmoT0ScIXdsSGFcSVgZXsRsbT3BlbkFJprfRpFvLrPA2Ve0JUIf7FHe_Yt5LblGXsdy2vWBIUA"




def multimodal_workflow(memory, question, docs=None):
    retriever = BM25Retriever.from_documents(docs, k=3) if docs else None
    llm = ChatOpenAI(model="gpt-4-mini")  # No need to pass api_key here

    chain = (
        {
            "context": retriever if retriever else (lambda x: "No additional context available"),
            "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
            "question": RunnablePassthrough()
        }
        | TEMPLATES["multimodal"]
        | llm
        | StrOutputParser()
    )