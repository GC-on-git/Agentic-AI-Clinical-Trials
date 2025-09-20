import asyncio
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class SBERTEmbedder:
    """Async-friendly embedding service for clinical text chunks using SBERT."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name,device="mps")

    async def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of pipeline-style chunks.

        Args:
            chunks: List of dictionaries, each containing at least:
                    {
                        "metadata": ...,
                        "content": ...
                    }

        Returns:
            List of dictionaries in format:
            [
                {"id": <chunk_id>, "embedding": <vector>, "metadata": <chunk_metadata>},
                ...
            ]
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await asyncio.to_thread(
            self.model.encode, texts, convert_to_numpy=True, show_progress_bar=False
        )
        embeddings = embeddings.tolist()

        # Return in pipeline-compatible format
        result = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("id", f"chunk_{i}")
            result.append({
                "id": chunk_id,
                "embedding": embeddings[i],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {})
            })
        return result

    def generate_embeddings_sync(self, chunks: List[Dict]) -> List[Dict]:
        """
        Synchronous wrapper for non-async code.
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

        result = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("id", f"chunk_{i}")
            result.append({
                "id": chunk_id,
                "embedding": embeddings[i],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {})
            })
        return result
