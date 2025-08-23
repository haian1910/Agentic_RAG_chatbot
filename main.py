from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
from typing import Dict, List, Optional
from agent import RAGAgent
import config

app = FastAPI(title="QA RAG API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store RAG agents by session ID
rag_agents: Dict[str, RAGAgent] = {}

# Load existing vector store on startup
@app.on_event("startup")
async def startup_event():
    print("Starting QA RAG API...")
    if os.path.exists(config.VECTORSTORE_PATH):
        print("Vector store found, will be loaded for new sessions")

def get_or_create_agent(session_id: str) -> RAGAgent:
    """Get existing agent or create new one for session"""
    if session_id not in rag_agents:
        rag_agents[session_id] = RAGAgent(session_id=session_id)
        # Load existing vector store if available
        if os.path.exists(config.VECTORSTORE_PATH):
            rag_agents[session_id].load_existing_vectorstore()
            print(f"Loaded existing vector store for session {session_id}")
    return rag_agents[session_id]

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str

class SessionRequest(BaseModel):
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

class UploadRequest(BaseModel):
    session_id: Optional[str] = None

@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    get_or_create_agent(session_id)  # This creates the agent
    return SessionResponse(
        session_id=session_id,
        message="New session created successfully"
    )

@app.post("/session/clear", response_model=SessionResponse)
async def clear_session(request: SessionRequest):
    """Clear memory for a specific session"""
    session_id = request.session_id or str(uuid.uuid4())
    agent = get_or_create_agent(session_id)
    agent.clear_memory()
    return SessionResponse(
        session_id=session_id,
        message="Session memory cleared successfully"
    )

@app.get("/session/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in rag_agents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = rag_agents[session_id]
    messages = agent.get_memory_summary()
    return ChatHistoryResponse(session_id=session_id, messages=messages)

@app.post("/upload", summary="Upload PDF document")
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Upload a PDF document to be indexed"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create documents directory if it doesn't exist
    os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(config.DOCUMENTS_PATH, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # If session_id provided, load for that specific session
        # Otherwise, load for all existing sessions and future ones
        if session_id:
            agent = get_or_create_agent(session_id)
            success = agent.load_documents(file_path)
        else:
            # Load for a default agent (this will save to disk for all sessions)
            default_agent = get_or_create_agent("default")
            success = default_agent.load_documents(file_path)
            
            # Reload vector store for all existing sessions
            for agent in rag_agents.values():
                agent.load_existing_vectorstore()
        
        if success:
            return {"message": f"Document {file.filename} uploaded and indexed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_documents(request: QueryRequest):
    """Ask a question about uploaded documents or general knowledge"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = get_or_create_agent(session_id)
        answer = agent.query(request.question)
        return QueryResponse(answer=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/sessions", summary="List active sessions")
async def list_sessions():
    """List all active session IDs"""
    return {"sessions": list(rag_agents.keys()), "count": len(rag_agents)}

@app.delete("/session/{session_id}", summary="Delete a session")
async def delete_session(session_id: str):
    """Delete a specific session and its memory"""
    if session_id in rag_agents:
        del rag_agents[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(rag_agents),
        "vectorstore_available": os.path.exists(config.VECTORSTORE_PATH)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "QA RAG API with Memory is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)