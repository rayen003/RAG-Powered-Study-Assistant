import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Text Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Configuration
PERSIST_DIRECTORY = "db"