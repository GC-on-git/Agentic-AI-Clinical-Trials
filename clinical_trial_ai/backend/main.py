#!/usr/bin/env python3
"""
Main FastAPI application for Clinical Trial AI
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from starlette.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles

from backend.config import config

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.dependencies import initialize_system, shutdown_system, get_app_state
from api.schemas import ErrorResponse
from api.routers import documents, queries, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup and shutdown"""
    # Startup
    try:
        await initialize_system()
        yield
    except Exception as e:
        print(f"Failed to start application: {e}")
        raise e
    finally:
        # Shutdown
        await shutdown_system()


# FastAPI app initialization
app = FastAPI(
    title="Clinical Trial AI API",
    description="API for processing clinical trial documents and answering queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent
# print(f"{BASE_DIR.parent}/frontend")
app.mount("/static", StaticFiles(directory=f"{BASE_DIR.parent}/frontend", html=True), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router)
app.include_router(queries.router)
app.include_router(documents.router)


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
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now()
        ).model_dump()
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return FileResponse(f"{BASE_DIR.parent}/frontend/index.html")


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
