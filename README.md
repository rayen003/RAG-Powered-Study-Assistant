# Study Assistant

A smart study assistant powered by RAG (Retrieval Augmented Generation) that helps students better understand their study materials. This application uses OpenAI's GPT-4 model and embeddings to provide intelligent responses to questions about uploaded PDF documents.

## Features

- PDF document upload and processing
- Intelligent question answering using RAG (Retrieval Augmented Generation)
- Interactive chat interface using Streamlit
- Context-aware responses based on document content
- Modern and user-friendly interface

## Prerequisites

- Python 3.8+
- OpenAI API key

## Setup

1. Clone the repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app/main.py
   ```
2. Upload a PDF document using the sidebar
3. Ask questions about the document in the chat interface
4. Get AI-powered responses based on the document content

## Project Structure

```
study-assistant/
├── app/
│   ├── main.py          # Streamlit app entry point
│   ├── rag.py           # RAG pipeline implementation
│   ├── utils.py         # Document processing utilities
│   └── config.py        # Configuration settings
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── .env                # Environment variables
```

## Configuration

The application can be configured through the following environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: OpenAI model to use (default: "gpt-4")
- `EMBEDDING_MODEL`: Model for embeddings (default: "text-embedding-ada-002")

## Contributing

Feel free to submit issues and enhancement requests!
