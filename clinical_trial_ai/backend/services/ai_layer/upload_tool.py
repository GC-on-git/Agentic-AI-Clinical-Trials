import uuid
from pathlib import Path
from typing import List, Dict, Any
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService
from clinical_trial_ai.backend.services.data_collection.user_uploads import FileUploadCollector

class UploadTool:
    """Handles end-to-end file upload and processing"""

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

    async def process_file(self, file_path: str) -> str:
        """Full async pipeline for one file"""
        file_path_obj = Path(file_path)
        collected = self.collector.collect(file_path_obj)
        content = collected["content"]
        metadata = collected["metadata"]

        # Convert structured data to text
        if isinstance(content, list):
            content_text = "\n".join([str(row) for row in content])
        elif isinstance(content, dict):
            content_text = "\n".join([str(v) for v in content.values()])
        else:
            content_text = str(content)

        # Create document
        document = {
            "id": str(uuid.uuid4()),
            "title": metadata.get("file_name", "Untitled"),
            "source": "file_upload",
            "document_type": metadata.get("type", "unknown"),
            "content": content_text,
            "metadata": metadata,
            "processing_status": "RAW"
        }

        # Store raw document
        await self.storage_service.store_document(document)
        await self.storage_service.update_document_status(document["id"], "PROCESSING")

        # Chunk document
        processed = self.text_processor.process_text(document["content"], document_id=document["id"])
        chunks: List[Dict[str, Any]] = processed["chunks"]

        # Store chunks
        await self.storage_service.store_chunks(chunks)
        await self.storage_service.update_document_status(document["id"], "CHUNKED")

        # Generate embeddings
        embeddings = await self.embedder.generate_embeddings(chunks)

        # Store embeddings
        await self.vector_store.store_embeddings(embeddings)
        await self.storage_service.update_document_status(document["id"], "EMBEDDED")

        return document["id"]
