#!/usr/bin/env python3
"""
System router for Clinical Trial AI API
Handles system status, health checks, and agent information
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..schemas import (
    SystemStatusResponse, HealthResponse, RootResponse,
    AgentsResponse, ErrorResponse
)
from ..dependencies import get_flow, get_app_state

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check(app_state=Depends(get_app_state)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if app_state["initialized"] else "initializing",
        timestamp=datetime.now(),
        system_initialized=app_state["initialized"]
    )


@router.get("/system/status", response_model=SystemStatusResponse)
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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="SYSTEM_STATUS_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )


@router.get("/system/agents", response_model=AgentsResponse)
async def get_available_agents(flow=Depends(get_flow)):
    """Get detailed information about available AI agents"""
    try:
        available_agents = await flow.discover_available_tools()

        return AgentsResponse(
            success=True,
            agents=available_agents,
            total_agents=len(available_agents)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                error_code="GET_AGENTS_ERROR",
                timestamp=datetime.now()
            ).model_dump()
        )
