# AI Study Assistant

A smart study assistant powered by LangChain and OpenAI that helps students understand their study materials through interactive conversations. The assistant uses RAG (Retrieval Augmented Generation) to provide context-aware responses while seamlessly blending in general knowledge when needed.

## Features

- **Smart Context Understanding**: Automatically decides when to use document context and when to supplement with general knowledge
- **Document Processing**: Upload and process PDF study materials
- **Interactive Chat**: Natural conversation interface with memory of previous interactions
- **Dynamic Response Generation**: Uses cosine similarity and LLM evaluation to ensure relevant and comprehensive answers
- **Conversation Memory**: Maintains context across multiple questions for more coherent discussions

## Technical Details

### Architecture

- **Frontend**: Streamlit for the web interface
- **Backend**: Python with LangChain for RAG implementation
- **Embedding**: OpenAI's text embedding model for document vectorization
- **Vector Store**: Chroma DB for efficient similarity search
- **LLM**: OpenAI's GPT models for response generation

### Smart RAG Implementation

The assistant uses a sophisticated RAG system that:
1. Retrieves relevant document chunks using similarity search
2. Evaluates context relevance using cosine similarity
3. Determines if the context is sufficient or needs supplementation
4. Dynamically combines document knowledge with general knowledge when appropriate

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd study-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app/main.py
```

## Usage

1. **Start the Application**: Launch the web interface using Streamlit
2. **Upload Documents** (Optional): Use the sidebar to upload PDF study materials
3. **Ask Questions**: Type your questions in the chat interface
4. **View Responses**: The assistant will provide answers using:
   - Document context when relevant
   - General knowledge when needed
   - A combination of both when appropriate

## Dependencies

- Python 3.8+
- LangChain
- OpenAI
- Streamlit
- ChromaDB
- PyPDF2
- python-dotenv

## Configuration

Key settings can be adjusted in `config.py`:
- Model selection
- Similarity thresholds
- Chunk sizes for document processing
- Vector store settings

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.

## License

[Your chosen license]
