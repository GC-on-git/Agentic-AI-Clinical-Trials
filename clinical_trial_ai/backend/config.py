"""
Configuration management for Clinical Trial AI system
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Environment variables will only be loaded from system environment.")


@dataclass
class LLMConfig:
    """Configuration for Large Language Model integration"""
    provider: str = "openai"  # openai, anthropic, local
    api_key: Optional[str] = None
    model_name: str = "sonar-pro"
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
        self._load_dotenv()
        self._load_config()

    def _load_dotenv(self):
        """Load environment variables from .env file"""
        if not DOTENV_AVAILABLE:
            return

        config_path = Path(self.config_file)

        if config_path.exists():
            load_dotenv(config_path)
            print(f"Loaded environment variables from {config_path}")
        else:
            # Try to load from default .env file in current directory
            default_env = Path("clinical_trial_ai/backend/.env")
            if default_env.exists():
                load_dotenv(default_env)
                print(f"Loaded environment variables from {default_env}")
            else:
                print(f"No .env file found at {config_path} or ./env")

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

    def reload_config(self, config_file: Optional[str] = None):
        """Reload configuration from file"""
        if config_file:
            self.config_file = config_file
        self._load_dotenv()
        self._load_config()
        print("Configuration reloaded successfully")

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "dotenv_available": DOTENV_AVAILABLE,
            "config_file": self.config_file,
            "config_file_exists": Path(self.config_file).exists(),
            "llm": {
                "configured": bool(self.llm.api_key),
                "provider": self.llm.provider,
                "model": self.llm.model_name
            },
            "database": {
                "db_path": self.database.db_path,
                "storage_path": self.database.storage_path,
                "storage_exists": Path(self.database.storage_path).exists()
            },
            "vectordb": {
                "host": self.vectordb.host,
                "port": self.vectordb.port,
                "collection": self.vectordb.collection_name
            },
            "embedding": {
                "model": self.embedding.model_name,
                "cache_dir": self.embedding.cache_dir,
                "cache_exists": Path(self.embedding.cache_dir).exists()
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

    def create_sample_env_file(self, filepath: str = ".env.example"):
        """Create a sample .env file with all available configuration options"""
        sample_content = """# Clinical Trial AI Configuration

# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7
LLM_BASE_URL=

# Alternative API key names (fallback options)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database Configuration
DB_PATH=clinical_docs.db
STORAGE_PATH=./storage

# Vector Database Configuration
VECTORDB_HOST=localhost
VECTORDB_PORT=8000
VECTORDB_COLLECTION=clinical_embeddings

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_DIR=./embedding_cache
"""

        with open(filepath, 'w') as f:
            f.write(sample_content)

        print(f"Sample environment file created at {filepath}")
        print("Copy this file to .env and update with your actual values")


# Global config instance
config = Config(config_file="/Users/anuganch/Desktop/Mock3_final/clinical_trial_ai/backend/.env")


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def install_dotenv_instructions():
    """Print instructions for installing python-dotenv"""
    print("""
To enable .env file support, install python-dotenv:

    pip install python-dotenv

Or add it to your requirements.txt file:
    python-dotenv>=0.19.0
""")


# Print installation instructions if dotenv is not available
if not DOTENV_AVAILABLE:
    install_dotenv_instructions()
