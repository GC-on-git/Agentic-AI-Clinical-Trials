# backend/services/text_processing/chunker.py
"""
Text chunking for clinical documents with pipeline-compatible output.
"""

from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime


class ClinicalTextChunker:
    """Splits text into overlapping chunks, returning pipeline-ready chunk objects."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: The document text
            document_id: ID of the source document

        Returns:
            List of chunks, each as a dict with minimal metadata
        """
        sentences = sent_tokenize(text)
        chunks: List[Dict[str, Any]] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(word_tokenize(sentence))

            # Start new chunk if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "id": f"{document_id}_chunk_{len(chunks)}",
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "content": chunk_text,
                    "word_count": len(word_tokenize(chunk_text)),
                    "chunk_type": "text",
                    "metadata": {"original_doc_id": document_id},
                    "created_at": datetime.now().isoformat()
                })

                # Keep overlap sentences
                overlap_sentences = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(word_tokenize(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "id": f"{document_id}_chunk_{len(chunks)}",
                "document_id": document_id,
                "chunk_index": len(chunks),
                "content": chunk_text,
                "word_count": len(word_tokenize(chunk_text)),
                "chunk_type": "text",
                "metadata": {"original_doc_id": document_id},
                "created_at": datetime.now().isoformat()
            })

        return chunks
