from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
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

# Initialize RAG agent
rag_agent = RAGAgent()

# Load existing vector store on startup
@app.on_event("startup")
async def startup_event():
    if os.path.exists(config.VECTORSTORE_PATH):
        rag_agent.load_existing_vectorstore()
        print("Loaded existing vector store")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/upload", summary="Upload PDF document")
async def upload_document(file: UploadFile = File(...)):
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
        
        # Load documents into vector store
        success = rag_agent.load_documents(file_path)
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
        answer = rag_agent.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "QA RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)