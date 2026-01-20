"""FastAPI server for PocketRAG."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pocketrag.pocketrag import PocketRAG


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    return_diagnostics: bool = False


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    citations: List[dict]
    sources: List[str]
    citation_valid: bool
    entity_verified: bool
    diagnostics: Optional[dict] = None


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    message: str
    num_documents: int
    num_chunks: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    num_documents: int
    num_chunks: int
    indexed: bool
    avg_chunk_size: float


# Initialize FastAPI app
app = FastAPI(
    title="PocketRAG API",
    description="Fully offline document QA system with citation-backed answers",
    version="0.1.0"
)

# Initialize PocketRAG system (global instance)
# Note: In production, consider using dependency injection
pocket_rag: Optional[PocketRAG] = None


@app.on_event("startup")
async def startup_event():
    """Initialize PocketRAG on startup."""
    global pocket_rag
    
    print("Initializing PocketRAG system...")
    pocket_rag = PocketRAG(
        data_dir=Path("data"),
        use_llm=False  # Set to True to use LLM (slower but better answers)
    )
    print("PocketRAG ready!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PocketRAG API",
        "version": "0.1.0",
        "endpoints": {
            "POST /ingest/file": "Upload and ingest a document",
            "POST /ingest/directory": "Ingest documents from a directory",
            "POST /query": "Query the system",
            "GET /stats": "Get system statistics",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pocket_rag is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "healthy",
        "indexed": pocket_rag.indexed
    }


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a single document file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Ingestion result
    """
    if pocket_rag is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check file type
    allowed_extensions = ['.pdf', '.txt', '.md']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest the file
        documents = pocket_rag.ingest_documents(file_paths=[temp_file_path])
        
        stats = pocket_rag.get_statistics()
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            temp_file_path.unlink()


@app.post("/ingest/directory")
async def ingest_directory(directory_path: str):
    """Ingest all documents from a directory.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Ingestion result
    """
    if pocket_rag is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")
    
    try:
        documents = pocket_rag.ingest_documents(directory=dir_path)
        
        stats = pocket_rag.get_statistics()
        
        return IngestResponse(
            message=f"Successfully ingested documents from {directory_path}",
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting directory: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the document QA system.
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and citations
    """
    if pocket_rag is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = pocket_rag.query(
            request.query,
            return_diagnostics=request.return_diagnostics
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics.
    
    Returns:
        System statistics
    """
    if pocket_rag is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = pocket_rag.get_statistics()
    return StatsResponse(**stats)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
