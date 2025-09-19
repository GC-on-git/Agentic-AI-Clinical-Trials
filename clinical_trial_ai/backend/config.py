"""
Configuration management for Clinical Trial AI system
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for Large Language Model integration"""
    provider: str = "openai"  # openai, anthropic, local
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    base_url: Optional[str] = None  # For custom endpoints


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "clinical_docs.db"
    storage_path: str = "./storage"


@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "clinical_embeddings"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "./embedding_cache"


class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or ".env"
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and config file"""
        
        # LLM Configuration
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            base_url=os.getenv("LLM_BASE_URL")
        )
        
        # Database Configuration
        self.database = DatabaseConfig(
            db_path=os.getenv("DB_PATH", "clinical_docs.db"),
            storage_path=os.getenv("STORAGE_PATH", "./storage")
        )
        
        # Vector Database Configuration
        self.vectordb = VectorDBConfig(
            host=os.getenv("VECTORDB_HOST", "localhost"),
            port=int(os.getenv("VECTORDB_PORT", "8000")),
            collection_name=os.getenv("VECTORDB_COLLECTION", "clinical_embeddings")
        )
        
        # Embedding Configuration
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache")
        )
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "llm": {
                "configured": bool(self.llm.api_key),
                "provider": self.llm.provider,
                "model": self.llm.model_name
            },
            "database": {
                "db_path": self.database.db_path,
                "storage_path": self.database.storage_path
            },
            "vectordb": {
                "host": self.vectordb.host,
                "port": self.vectordb.port,
                "collection": self.vectordb.collection_name
            },
            "embedding": {
                "model": self.embedding.model_name,
                "cache_dir": self.embedding.cache_dir
            }
        }
        
        return validation_results
    
    def get_llm_api_key(self) -> Optional[str]:
        """Get LLM API key with fallback methods"""
        if self.llm.api_key:
            return self.llm.api_key
        
        # Try alternative environment variable names
        alternative_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", 
            "API_KEY"
        ]
        
        for key in alternative_keys:
            value = os.getenv(key)
            if value:
                return value
        
        return None


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config
