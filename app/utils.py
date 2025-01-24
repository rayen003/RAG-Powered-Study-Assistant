from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def initialize_flan_t5():
    """Initialize FLAN-T5 model and tokenizer"""
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    return model, tokenizer

def load_and_split_documents(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Load a PDF document and split it into chunks.
    """
    # Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_memory():
    """
    Create a conversation buffer memory.
    """
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        return_messages=True
    )
    return memory