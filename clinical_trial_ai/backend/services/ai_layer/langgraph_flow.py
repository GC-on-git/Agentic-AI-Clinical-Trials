import asyncio
from typing import Dict, Any
from backend.services.text_processing.preprocessing import ClinicalTextProcessor
from backend.services.embeddings.sbert_embed import SBERTEmbedder
from backend.services.vector_db.chroma_store import ChromaVectorStore
from backend.services.storage.object_store import DocumentStorageService
# from backend.services.data_collection.user_uploads import FileUploadCollector

from .upload_tool import UploadTool
from .predictive_models import PredictiveModels
from .reasoning_tool import ReasoningTool
from .search_tool import SearchTool
from .summary_tool import SummaryTool


class LangGraphFlow:
    """Orchestrates the complete AI-driven document pipeline"""

    def __init__(self,
                 storage_service: DocumentStorageService,
                 text_processor: ClinicalTextProcessor,
                 embedder: SBERTEmbedder,
                 vector_store: ChromaVectorStore):
        self.upload_tool = UploadTool(storage_service, text_processor, embedder, vector_store)
        self.predictive_models = PredictiveModels()
        self.reasoning_tool = ReasoningTool()
        self.search_tool = SearchTool(vector_store)
        self.summary_tool = SummaryTool()

    async def run_pipeline(self, file_path: str) -> Dict[str, Any]:
        """Run full pipeline: upload → chunk → embed → summarize"""
        document_id = await self.upload_tool.process_file(file_path)

        # Retrieve stored chunks
        chunks = await self.search_tool.get_chunks_by_document_id(document_id)

        # Generate predictions or insights
        predictions = await self.predictive_models.predict(chunks)

        # Reasoning
        reasoning_result = await self.reasoning_tool.analyze(predictions)

        # Summarize
        summary = await self.summary_tool.generate_summary(chunks)

        return {
            "document_id": document_id,
            "predictions": predictions,
            "reasoning": reasoning_result,
            "summary": summary
        }
