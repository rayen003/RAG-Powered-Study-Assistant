import os
from dotenv import load_dotenv
from pathlib import Path

# Get the absolute path to the .env file
env_path = Path(__file__).parent.parent / '.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other configurations
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "db"