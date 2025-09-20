#!/usr/bin/env python3
"""
Queries router for Clinical Trial AI API
Handles query processing and RAG operations
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..schemas import QueryRequest, QueryResponse, ErrorResponse
from ..dependencies import get_flow

router = APIRouter(prefix="/api/v1", tags=["queries"])


@router.post("/query", response_model=QueryResponse)
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="QUERY_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )
