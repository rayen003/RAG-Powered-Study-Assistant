import os
from dotenv import load_dotenv
from pathlib import Path

# Get the absolute path to the .env file
env_path = Path(__file__).parent.parent / '.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = "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"

# Set the API key in environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Other configurations
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "db"