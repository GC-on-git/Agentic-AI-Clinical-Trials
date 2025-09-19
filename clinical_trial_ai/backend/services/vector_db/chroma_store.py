from typing import List, Dict, Any, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """
    Simplified ChromaDB vector store for pipeline-compatible clinical document embeddings.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "clinical_embeddings"):
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_server_host=host,
            chroma_server_http_port=port,
            anonymized_telemetry=False
        ))
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Clinical documents embeddings"}
        )

    async def store_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """
        Store pipeline-compatible embeddings in ChromaDB.

        Args:
            embeddings: List of dictionaries with keys:
                        {"id": str, "embedding": List[float], "metadata": dict}

        Returns:
            Success status
        """
        try:
            ids = [e["id"] for e in embeddings]
            vectors = [e["embedding"] for e in embeddings]
            metadatas = [
                {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                 for k, v in e["metadata"].items()}
                for e in embeddings
            ]
            documents = [e["metadata"].get("content", "") for e in embeddings]

            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception:
            return False

    async def similarity_search(self, query_embedding: List[float], top_k: int = 5,
                                filters: Dict[str, Any] = None, similarity_threshold: float = 0.7
                                ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search against stored embeddings.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity to include

        Returns:
            List of tuples (chunk_metadata, similarity_score)
        """
        filters = {k: str(v) for k, v in (filters or {}).items()}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters if filters else None
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                if similarity < similarity_threshold:
                    continue
                metadata = results["metadatas"][0][i]
                search_results.append((metadata, similarity))

        return search_results

    async def delete_document(self, document_id: str) -> bool:
        """Delete all embeddings for a given document ID"""
        try:
            results = self.collection.get(where={"document_id": document_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            return True
        except Exception:
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic collection stats"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "last_updated": datetime.now().isoformat()
            }
        except Exception:
            return {}
