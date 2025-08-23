# QA RAG System

A simple Question-Answering system using Retrieval-Augmented Generation (RAG) with FastAPI backend and Streamlit frontend.

## Update
- 23/8/2025: Adding memory for chatbot.

## Features

- 📄 PDF document upload and indexing
- 🔍 Vector search using FAISS
- 🌐 Web search fallback using Tavily
- 🤖 Agentic RAG with Google Gemini
- 🚀 FastAPI REST API
- 🎨 Streamlit web interface
- 🐳 Docker containerization

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Streamlit     │───▶│   FastAPI    │───▶│   RAG Agent     │
│   Frontend      │    │   Backend    │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                    │
                                           ┌────────┴────────┐
                                           │                 │
                                    ┌──────▼──────┐ ┌────────▼──────┐
                                    │ Vector Store│ │  Web Search   │
                                    │   (FAISS)   │ │   (Tavily)    │
                                    └─────────────┘ └───────────────┘
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd qa-rag-system
```

### 2. Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Getting API Keys:**
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

### 3. Installation Options

#### Option A: Docker (Recommended)

```bash
# Start both services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

#### Option B: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python main.py

# Start Streamlit (in another terminal)
streamlit run streamlit_app.py
```

## Usage

### 1. Access the Application

- **Streamlit Interface**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs

### 2. Upload Documents

1. Open the Streamlit interface
2. Use the sidebar to upload PDF documents
3. Click "Upload & Index" to process the document

### 3. Ask Questions

- Type questions in the chat interface
- The system will search your documents first
- If no relevant information is found, it will search the web

### 4. API Endpoints

#### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

#### Ask Question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## Project Structure

```
.
├── main.py              # FastAPI application
├── streamlit_app.py     # Streamlit frontend
├── agent.py            # RAG agent implementation
├── vector_store.py     # Vector store management
├── web_search.py       # Web search functionality
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Multi-service setup
├── .env.example        # Environment variables template
├── documents/          # Uploaded documents (auto-created)
├── vectorstore/        # Vector store data (auto-created)
└── README.md          # This file
```

## How It Works

1. **Document Upload**: PDFs are processed, chunked, and embedded using HuggingFace embeddings
2. **Vector Storage**: Document embeddings are stored in FAISS for fast similarity search
3. **Agent Decision**: The RAG agent decides whether to search documents or the web
4. **Response Generation**: Google Gemini generates responses based on retrieved context

## Example Questions

- "What are the key financial metrics in Q3 2024?"
- "What milestones did the Shanghai factory achieve?"
- "Latest news about Tesla stock performance?"

## Customization

### Modify Settings

Edit `config.py` to change:
- Chunk size and overlap
- Embedding model
- LLM model
- Search result count

### Add New Tools

Extend the `RAGAgent` class in `agent.py` to add new search tools or data sources.

## Troubleshooting

### Common Issues

1. **API Connection Error**: Make sure FastAPI is running on port 8000
2. **Missing API Keys**: Check your `.env` file has the correct API keys
3. **Docker Build Issues**: Ensure Docker and docker-compose are installed

### Logs

```bash
# View container logs
docker-compose logs fastapi
docker-compose logs streamlit

# Follow logs
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

<<<<<<< HEAD
This project is licensed under the MIT License.
=======
This project is licensed under the MIT License.
>>>>>>> 3cd84c173e5a093788b4355ef9737a43f6a164ee
