#!/usr/bin/env python3
"""
Documents router for Clinical Trial AI API
Handles document upload, processing, and management
"""

import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ..schemas import (
    DocumentUploadResponse, DocumentSummaryResponse, DocumentInfo, ErrorResponse
)
from ..dependencies import get_flow, get_app_state

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def process_document_async(document_id: str, file_path: str, filename: str, app_state: dict):
    """Background task to process uploaded documents"""
    try:
        start_time = datetime.now()

        # Update status
        app_state["documents"][document_id]["processing_status"] = "processing"

        # Process the document
        flow = app_state["flow"]
        processed_doc_id = await flow.upload_tool.process_file(file_path)

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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    flow=Depends(get_flow),
    app_state=Depends(get_app_state)
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
                file.filename,
                app_state
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="UPLOAD_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )


@router.get("", response_model=List[DocumentInfo])
async def list_documents(app_state=Depends(get_app_state)):
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="LIST_DOCUMENTS_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )


@router.get("/{document_id}")
async def get_document_info(document_id: str, app_state=Depends(get_app_state)):
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="GET_DOCUMENT_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )


@router.post("/{document_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
    document_id: str,
    flow=Depends(get_flow),
    app_state=Depends(get_app_state)
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="SUMMARY_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str, app_state=Depends(get_app_state)):
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="DELETE_DOCUMENT_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )
