#!/usr/bin/env python3
"""
Pydantic schemas for Clinical Trial AI API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for query processing"""
    query: str = Field(..., description="User query about clinical trials")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class QueryResponse(BaseModel):
    """Response schema for query processing"""
    success: bool
    response: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    relevant_chunks: List[Dict[str, Any]]
    session_id: str


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    success: bool
    document_id: str
    filename: str
    file_size: int
    processing_status: str
    message: str
    processing_time: Optional[float] = None


class DocumentSummaryResponse(BaseModel):
    """Response schema for document summary"""
    success: bool
    document_id: str
    summary: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class DocumentInfo(BaseModel):
    """Schema for document information"""
    document_id: str
    filename: str
    file_size: int
    upload_time: datetime
    processing_status: str


class SystemStatusResponse(BaseModel):
    """Response schema for system status"""
    success: bool
    system_state: Dict[str, Any]
    llm_status: Dict[str, Any]
    available_agents: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    error_code: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    timestamp: datetime
    system_initialized: bool


class RootResponse(BaseModel):
    """Root endpoint response schema"""
    message: str
    version: str
    status: str
    docs: str


class AgentInfo(BaseModel):
    """Schema for agent information"""
    name: str
    description: str
    capabilities: List[str]
    status: str


class AgentsResponse(BaseModel):
    """Response schema for available agents"""
    success: bool
    agents: List[AgentInfo]
    total_agents: int
