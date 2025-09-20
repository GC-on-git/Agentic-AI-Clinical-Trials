#!/usr/bin/env python3
"""
FastAPI Backend for Clinical Trial AI
Main application file with all endpoints
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_trial_ai.backend.services.ai_layer.langgraph_flow import LangGraphFlow
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about clinical trials")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class QueryResponse(BaseModel):
    success: bool
    response: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    relevant_chunks: List[Dict[str, Any]]
    session_id: str


class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    file_size: int
    processing_status: str
    message: str
    processing_time: Optional[float] = None


class DocumentSummaryResponse(BaseModel):
    success: bool
    document_id: str
    summary: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_time: datetime
    processing_status: str


class SystemStatusResponse(BaseModel):
    success: bool
    system_state: Dict[str, Any]
    llm_status: Dict[str, Any]
    available_agents: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    timestamp: datetime


# Global variables for system components
app_state = {
    "flow": None,
    "documents": {},  # In-memory document storage for demo
    "processing_jobs": {},  # Track async processing jobs
    "initialized": False
}

# FastAPI app initialization
app = FastAPI(
    title="Clinical Trial AI API",
    description="API for processing clinical trial documents and answering queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the Clinical Trial AI system on startup"""
    try:
        print("Initializing Clinical Trial AI System...")

        # Initialize components
        text_processor = ClinicalTextProcessor()
        embedder = SBERTEmbedder()
        vector_store = ChromaVectorStore()
        storage_service = DocumentStorageService()

        # Initialize the enhanced flow with agentic capabilities
        app_state["flow"] = LangGraphFlow(
            storage_service=storage_service,
            text_processor=text_processor,
            embedder=embedder,
            vector_store=vector_store
        )

        app_state["initialized"] = True
        print("Clinical Trial AI System initialized successfully!")

    except Exception as e:
        print(f"Failed to initialize system: {e}")
        app_state["initialized"] = False


# Dependency to check system initialization
async def get_flow():
    if not app_state["initialized"] or not app_state["flow"]:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please try again later."
        )
    return app_state["flow"]


# Utility function for error handling
def handle_error(error: Exception, error_code: str) -> JSONResponse:
    """Handle errors and return standardized error response"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(error),
            error_code=error_code,
            timestamp=datetime.now()
        ).dict()
    )


# Background task for document processing
async def process_document_async(document_id: str, file_path: str, filename: str):
    """Background task to process uploaded documents"""
    try:
        start_time = datetime.now()

        # Update status
        app_state["documents"][document_id]["processing_status"] = "processing"

        # Process the document
        processed_doc_id = await app_state["flow"].upload_tool.process_file(file_path)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Update document info
        app_state["documents"][document_id].update({
            "processing_status": "completed",
            "processed_doc_id": processed_doc_id,
            "processing_time": processing_time,
            "completed_at": end_time
        })

        print(f"Document {filename} processed successfully in {processing_time:.2f}s")

    except Exception as e:
        print(f"Error processing document {filename}: {e}")
        app_state["documents"][document_id]["processing_status"] = "failed"
        app_state["documents"][document_id]["error"] = str(e)
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Clinical Trial AI API",
        "version": "1.0.0",
        "status": "running" if app_state["initialized"] else "initializing",
        "docs": "/docs"
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if app_state["initialized"] else "initializing",
        "timestamp": datetime.now(),
        "system_initialized": app_state["initialized"]
    }


@app.post("/api/v1/upload", response_model=DocumentUploadResponse)
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        flow=Depends(get_flow)
):
    """Upload and process a clinical trial document"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.csv'}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Check file size (150MB limit)
        max_size = 150 * 1024 * 1024  # 150MB in bytes
        file_size = 0

        # Create temporary file
        document_id = str(uuid.uuid4())
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            prefix=f"clinical_doc_{document_id}_"
        )

        try:
            # Read and write file in chunks to check size
            while chunk := await file.read(8192):  # 8KB chunks
                file_size += len(chunk)
                if file_size > max_size:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    raise HTTPException(
                        status_code=413,
                        detail="File too large. Maximum size allowed is 150MB."
                    )
                temp_file.write(chunk)

            temp_file.close()

            # Store document info
            app_state["documents"][document_id] = {
                "document_id": document_id,
                "filename": file.filename,
                "file_size": file_size,
                "upload_time": datetime.now(),
                "processing_status": "uploaded",
                "file_path": temp_file.name
            }

            # Add background task for processing
            background_tasks.add_task(
                process_document_async,
                document_id,
                temp_file.name,
                file.filename
            )

            return DocumentUploadResponse(
                success=True,
                document_id=document_id,
                filename=file.filename,
                file_size=file_size,
                processing_status="uploaded",
                message="File uploaded successfully. Processing in background."
            )

        except Exception as e:
            # Clean up on error
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "UPLOAD_ERROR")


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
        request: QueryRequest,
        flow=Depends(get_flow)
):
    """Process a user query about clinical trials"""
    try:
        start_time = datetime.now()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process the query
        result = await flow.process_query(request.query, top_k=request.top_k)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return QueryResponse(
            success=True,
            response=result['response'],
            confidence=result['insights']['confidence'],
            processing_time=processing_time,
            metadata=result['metadata'],
            relevant_chunks=result['relevant_chunks'],
            session_id=session_id
        )

    except Exception as e:
        return handle_error(e, "QUERY_ERROR")


@app.post("/api/v1/documents/{document_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
        document_id: str,
        flow=Depends(get_flow)
):
    """Generate a summary of a processed document"""
    try:
        # Check if document exists and is processed
        if document_id not in app_state["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_info = app_state["documents"][document_id]
        if doc_info["processing_status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Document not ready for summary. Status: {doc_info['processing_status']}"
            )

        start_time = datetime.now()

        # Generate summary using the AI system
        summary_query = "Please provide a comprehensive summary of this clinical trial document, including key findings, methodologies, and outcomes."
        result = await flow.process_query(summary_query, top_k=10)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return DocumentSummaryResponse(
            success=True,
            document_id=document_id,
            summary=result['response'],
            confidence=result['insights']['confidence'],
            processing_time=processing_time,
            metadata=result['metadata']
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "SUMMARY_ERROR")


@app.get("/api/v1/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = []
        for doc_id, doc_info in app_state["documents"].items():
            documents.append(DocumentInfo(
                document_id=doc_id,
                filename=doc_info["filename"],
                file_size=doc_info["file_size"],
                upload_time=doc_info["upload_time"],
                processing_status=doc_info["processing_status"]
            ))

        return documents

    except Exception as e:
        return handle_error(e, "LIST_DOCUMENTS_ERROR")


@app.get("/api/v1/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get information about a specific document"""
    try:
        if document_id not in app_state["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_info = app_state["documents"][document_id].copy()
        # Remove sensitive file path
        doc_info.pop("file_path", None)

        return {
            "success": True,
            "document": doc_info
        }

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "GET_DOCUMENT_ERROR")


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    try:
        if document_id not in app_state["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")

        # Remove from memory
        doc_info = app_state["documents"].pop(document_id)

        # Clean up temporary file if it still exists
        if "file_path" in doc_info and os.path.exists(doc_info["file_path"]):
            os.remove(doc_info["file_path"])

        # TODO: In production, also remove from vector store and object store

        return {
            "success": True,
            "message": f"Document {doc_info['filename']} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "DELETE_DOCUMENT_ERROR")


@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status(flow=Depends(get_flow)):
    """Get system status and agent information"""
    try:
        # Get system state
        system_state = flow.get_system_state()

        # Get LLM status
        llm_status = flow.llm_service.get_status()

        # Get available agents
        available_agents = await flow.discover_available_tools()

        return SystemStatusResponse(
            success=True,
            system_state=system_state,
            llm_status=llm_status,
            available_agents=available_agents
        )

    except Exception as e:
        return handle_error(e, "SYSTEM_STATUS_ERROR")


@app.get("/api/v1/system/agents")
async def get_available_agents(flow=Depends(get_flow)):
    """Get detailed information about available AI agents"""
    try:
        available_agents = await flow.discover_available_tools()

        return {
            "success": True,
            "agents": available_agents,
            "total_agents": len(available_agents)
        }

    except Exception as e:
        return handle_error(e, "GET_AGENTS_ERROR")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code="HTTP_ERROR",
            timestamp=datetime.now()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return handle_error(exc, "INTERNAL_ERROR")


# Main function for running the server
def main():
    """Main function to run the FastAPI server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()