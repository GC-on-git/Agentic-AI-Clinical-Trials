#!/usr/bin/env python3
"""
Dependencies for Clinical Trial AI API
Provides dependency injection for FastAPI endpoints
"""

from typing import Dict, Any
from fastapi import HTTPException, Depends

from clinical_trial_ai.backend.services.ai_layer.langgraph_flow import LangGraphFlow
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService
from clinical_trial_ai.backend.config import get_config


# Global app state
app_state: Dict[str, Any] = {
    "flow": None,
    "documents": {},  # In-memory document storage for demo
    "processing_jobs": {},  # Track async processing jobs
    "initialized": False
}


async def get_flow() -> LangGraphFlow:
    """Dependency to get the initialized LangGraphFlow instance"""
    if not app_state["initialized"] or not app_state["flow"]:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please try again later."
        )
    return app_state["flow"]


def get_app_state() -> Dict[str, Any]:
    """Dependency to get the application state"""
    return app_state


async def initialize_system():
    """Initialize the Clinical Trial AI system"""
    try:
        print("Initializing Clinical Trial AI System...")
        
        # Get configuration
        config = get_config()

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
        raise e


async def shutdown_system():
    """Shutdown the Clinical Trial AI system"""
    try:
        print("Shutting down Clinical Trial AI System...")
        
        # Clean up resources
        if app_state["flow"]:
            # Add any cleanup logic here if needed
            pass
        
        app_state["initialized"] = False
        print("Clinical Trial AI System shutdown complete!")
        
    except Exception as e:
        print(f"Error during shutdown: {e}")
