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
    provider: str = "openrouter"  # openrouter, perplexity
    openrouter_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    perplexity_model: str = "llama-3.1-sonar-small-128k-online"
    max_tokens: int = 1000
    temperature: float = 0.7


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
            provider=os.getenv("LLM_PROVIDER", "openrouter"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            perplexity_model=os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
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
                "provider": self.llm.provider,
                "openrouter_configured": bool(self.llm.openrouter_api_key),
                "perplexity_configured": bool(self.llm.perplexity_api_key),
                "openrouter_model": self.llm.openrouter_model,
                "perplexity_model": self.llm.perplexity_model,
                "current_provider_configured": self._is_current_provider_configured()
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

    def _is_current_provider_configured(self) -> bool:
        """Check if the current provider is properly configured"""
        if self.llm.provider == "openrouter":
            return bool(self.llm.openrouter_api_key)
        elif self.llm.provider == "perplexity":
            return bool(self.llm.perplexity_api_key)
        return False

    def get_openrouter_api_key(self) -> Optional[str]:
        """Get OpenRouter API key"""
        return self.llm.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

    def get_perplexity_api_key(self) -> Optional[str]:
        """Get Perplexity API key"""
        return self.llm.perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")

    def get_current_provider_api_key(self) -> Optional[str]:
        """Get API key for the currently configured provider"""
        if self.llm.provider == "openrouter":
            return self.get_openrouter_api_key()
        elif self.llm.provider == "perplexity":
            return self.get_perplexity_api_key()
        return None

    def get_available_providers(self) -> Dict[str, bool]:
        """Get list of available providers and their configuration status"""
        return {
            "openrouter": bool(self.get_openrouter_api_key()),
            "perplexity": bool(self.get_perplexity_api_key())
        }

    def switch_provider(self, provider: str) -> bool:
        """Switch to a different provider if configured"""
        provider = provider.lower()
        available_providers = self.get_available_providers()

        if provider not in available_providers:
            print(f"Error: Unknown provider '{provider}'. Available: {list(available_providers.keys())}")
            return False

        if not available_providers[provider]:
            print(f"Error: Provider '{provider}' is not configured (no API key)")
            return False

        self.llm.provider = provider
        print(f"Switched to provider: {provider}")
        return True

    def get_provider_models(self) -> Dict[str, str]:
        """Get the configured models for each provider"""
        return {
            "openrouter": self.llm.openrouter_model,
            "perplexity": self.llm.perplexity_model
        }

    def set_provider_model(self, provider: str, model: str) -> bool:
        """Set the model for a specific provider"""
        provider = provider.lower()

        if provider == "openrouter":
            self.llm.openrouter_model = model
            print(f"Set OpenRouter model to: {model}")
            return True
        elif provider == "perplexity":
            self.llm.perplexity_model = model
            print(f"Set Perplexity model to: {model}")
            return True
        else:
            print(f"Error: Unknown provider '{provider}'")
            return False

    def get_provider_status(self) -> Dict[str, Any]:
        """Get detailed status of all providers"""
        return {
            "current_provider": self.llm.provider,
            "providers": {
                "openrouter": {
                    "configured": bool(self.get_openrouter_api_key()),
                    "model": self.llm.openrouter_model,
                    "api_key_set": bool(self.llm.openrouter_api_key)
                },
                "perplexity": {
                    "configured": bool(self.get_perplexity_api_key()),
                    "model": self.llm.perplexity_model,
                    "api_key_set": bool(self.llm.perplexity_api_key)
                }
            }
        }

    def create_sample_env_file(self, filepath: str = ".env.example"):
        """Create a sample .env file with all available configuration options"""
        sample_content = """# Clinical Trial AI Configuration

# LLM Provider Configuration
LLM_PROVIDER=openrouter  # openrouter or perplexity

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Perplexity Configuration  
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PERPLEXITY_MODEL=llama-3.1-sonar-small-128k-online

# LLM General Settings
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7

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

# Available OpenRouter Models:
# - anthropic/claude-3.5-sonnet
# - anthropic/claude-3-haiku
# - openai/gpt-4o
# - openai/gpt-4o-mini
# - meta-llama/llama-3.1-8b-instruct
# - meta-llama/llama-3.1-70b-instruct
# - google/gemini-pro-1.5

# Available Perplexity Models:
# - llama-3.1-sonar-small-128k-online
# - llama-3.1-sonar-large-128k-online
# - llama-3.1-sonar-huge-128k-online
"""

        with open(filepath, 'w') as f:
            f.write(sample_content)

        print(f"Sample environment file created at {filepath}")
        print("Copy this file to .env and update with your actual values")

    def print_config_status(self):
        """Print a formatted status of the current configuration"""
        status = self.validate()
        provider_status = self.get_provider_status()

        print("\n=== Clinical Trial AI Configuration Status ===")
        print(f"Config file: {status['config_file']} (exists: {status['config_file_exists']})")
        print(f"Current provider: {provider_status['current_provider']}")

        print("\nProvider Status:")
        for provider, details in provider_status['providers'].items():
            status_icon = "✅" if details['configured'] else "❌"
            print(f"  {status_icon} {provider.capitalize()}: {details['model']} (API key: {'Set' if details['api_key_set'] else 'Not set'})")

        print(f"\nDatabase: {status['database']['db_path']}")
        print(f"Storage: {status['database']['storage_path']} (exists: {status['database']['storage_exists']})")
        print(f"Vector DB: {status['vectordb']['host']}:{status['vectordb']['port']}")
        print(f"Embedding model: {status['embedding']['model']}")
        print("=" * 50)


# Global config instance
BASE_DIR = Path(__file__).resolve().parent  # directory of the current file
CONFIG_PATH = BASE_DIR / ".env"      # find .env

config = Config(config_file=CONFIG_PATH)


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


# Example usage and testing
if __name__ == "__main__":
    # Print current configuration status
    config.print_config_status()

    # Test provider switching
    print(f"\nAvailable providers: {config.get_available_providers()}")

    # Create sample .env file
    config.create_sample_env_file()