import asyncio
from typing import List, Dict, Any
from backend.services.vector_db.chroma_store import ChromaVectorStore

class SearchTool:
    """Search chunks and documents in the vector database"""

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    async def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k most similar chunks"""
        results = await self.vector_store.search(query, top_k=top_k)
        return results

    async def get_chunks_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch all chunks for a specific document"""
        chunks = await self.vector_store.get_chunks(document_id)
        return chunks
