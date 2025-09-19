import aiosqlite
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class DocumentStorageService:
    """
    Async SQLite storage service for clinical documents and chunks.
    Uses SQLite for metadata and local filesystem for raw documents.
    """

    def __init__(self, db_path: str = "clinical_docs.db", storage_path: str = "./storage"):
        self.db_path = db_path
        self.db_conn: aiosqlite.Connection | None = None
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize SQLite connection and create tables"""
        self.db_conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()

    async def _create_tables(self):
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                document_type TEXT,
                file_path TEXT,
                file_hash TEXT,
                metadata TEXT DEFAULT '{}',
                processing_status TEXT DEFAULT 'RAW',
                created_at TEXT,
                updated_at TEXT
            );
            """)
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'text',
                word_count INTEGER,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                UNIQUE(document_id, chunk_index),
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """)
            await self.db_conn.commit()

    async def store_document(self, document: Dict[str, Any]) -> bool:
        """Store raw document content and metadata"""
        try:
            doc_id = document["id"]
            content = document["content"]
            metadata = document.get("metadata", {})

            # Compute content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Save content to file
            file_path = self.storage_path / f"{doc_id}.txt"
            file_path.write_text(content, encoding="utf-8")

            # Insert or replace in SQLite
            async with self.db_conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO documents (
                        id, title, source, document_type, file_path, file_hash, metadata,
                        processing_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    document.get("title", ""),
                    document.get("source", ""),
                    document.get("document_type", ""),
                    str(file_path),
                    content_hash,
                    json.dumps(metadata),
                    document.get("processing_status", "RAW"),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                await self.db_conn.commit()
            return True
        except Exception:
            return False

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store processed document chunks"""
        try:
            async with self.db_conn.cursor() as cursor:
                for i, c in enumerate(chunks):
                    chunk_id = c["metadata"].get("id", f"{c['metadata'].get('document_id')}_chunk_{i}")
                    await cursor.execute("""
                        INSERT OR REPLACE INTO document_chunks (
                            id, document_id, chunk_index, content, chunk_type,
                            word_count, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        c["metadata"].get("document_id", ""),
                        c["metadata"].get("chunk_index", i),
                        c["content"],
                        c.get("chunk_type", "text"),
                        c.get("word_count", len(c["content"].split())),
                        json.dumps(c.get("metadata", {})),
                        datetime.now().isoformat()
                    ))
                await self.db_conn.commit()
            return True
        except Exception:
            return False

    async def update_document_status(self, document_id: str, status: str) -> bool:
        """Update document processing status"""
        try:
            async with self.db_conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE documents
                    SET processing_status = ?, updated_at = ?
                    WHERE id = ?
                """, (status, datetime.now().isoformat(), document_id))
                await self.db_conn.commit()
            return True
        except Exception:
            return False

    async def cleanup(self):
        """Close SQLite connection"""
        if self.db_conn:
            await self.db_conn.close()
