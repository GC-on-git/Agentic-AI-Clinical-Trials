import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any

from backend.services.text_processing.preprocessing import ClinicalTextProcessor
from backend.services.embeddings.sbert_embed import SBERTEmbedder
from backend.services.vector_db.chroma_store import ChromaVectorStore
from backend.services.storage.object_store import DocumentStorageService
from backend.services.data_collection.user_uploads import FileUploadCollector


class FileProcessingPipeline:
    """End-to-end pipeline for processing uploaded files"""

    def __init__(self,
                 storage_service: DocumentStorageService,
                 text_processor: ClinicalTextProcessor,
                 embedder: SBERTEmbedder,
                 vector_store: ChromaVectorStore):
        self.storage_service = storage_service
        self.text_processor = text_processor
        self.embedder = embedder
        self.vector_store = vector_store
        self.collector = FileUploadCollector()

    async def process_file(self, file_path: Path):
        """Full pipeline for a single file"""
        # Step 1: Collect file
        collected = self.collector.collect(file_path)
        content = collected["content"]
        metadata = collected["metadata"]

        # Step 2: Convert structured data to text if needed
        if isinstance(content, list):
            content_text = "\n".join([str(row) for row in content])
        elif isinstance(content, dict):
            content_text = "\n".join([str(v) for v in content.values()])
        else:
            content_text = str(content)

        # Step 3: Create document dict
        document = {
            "id": str(uuid.uuid4()),
            "title": metadata.get("file_name", "Untitled"),
            "source": "file_upload",
            "document_type": metadata.get("type", "unknown"),
            "content": content_text,
            "metadata": metadata,
            "processing_status": "RAW"
        }

        # Step 4: Store raw document
        await self.storage_service.store_document(document)
        await self.storage_service.update_document_status(document["id"], "PROCESSING")

        # Step 5: Chunk document
        processed = self.text_processor.process_text(document["content"], document_id=document["id"])
        chunks: List[Dict[str, Any]] = processed["chunks"]

        # Step 6: Store chunks in DB
        await self.storage_service.store_chunks(chunks)
        await self.storage_service.update_document_status(document["id"], "CHUNKED")

        # Step 7: Generate embeddings
        chunk_texts = [c["content"] for c in chunks]
        embeddings = await self.embedder.generate_embeddings(chunk_texts)

        # Step 8: Store embeddings in Chroma
        await self.vector_store.store_embeddings(embeddings)
        await self.storage_service.update_document_status(document["id"], "EMBEDDED")

        return document["id"]
