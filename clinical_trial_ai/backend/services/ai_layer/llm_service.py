"""
LLM Service for Clinical Trial AI - Perplexity with Local Fallback
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import aiohttp
from backend.config import get_config


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """Generate response with context"""
        pass


class PerplexityProvider(LLMProvider):
    """Perplexity API provider (OpenAI-compatible)"""

    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-small-128k-online"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def _generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Perplexity API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Perplexity API error: {response.status} - {error_text}")
        except Exception as e:
            print(f"Error calling Perplexity API: {e}")
            raise e

    async def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """Generate response with clinical trial context"""

        system_prompt = """You are a clinical trial AI assistant. You help users understand clinical trial documents, protocols, and results. 

Guidelines:
- Provide accurate, evidence-based responses
- Cite specific information from the provided context
- Use medical terminology appropriately
- If information is not available in the context, clearly state this
- Focus on clinical trial specifics like endpoints, safety, efficacy, and protocols
- Be concise but comprehensive"""

        user_prompt = f"""Context from clinical trial documents:
{context}

User Question: {query}

Please provide a detailed response based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return await self._generate_response(messages, **kwargs)


class LocalLLMProvider(LLMProvider):
    """Local LLM provider (Ollama)"""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using local LLM (Ollama)"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 1000)
                    }
                }

                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "No response generated")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Local LLM error: {response.status} - {error_text}")
        except Exception as e:
            print(f"Error calling local LLM: {e}")
            raise e

    async def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """Generate response with clinical trial context"""

        prompt = f"""You are a clinical trial AI assistant. You help users understand clinical trial documents, protocols, and results.

Guidelines:
- Provide accurate, evidence-based responses
- Cite specific information from the provided context
- Use medical terminology appropriately
- If information is not available in the context, clearly state this
- Focus on clinical trial specifics like endpoints, safety, efficacy, and protocols
- Be concise but comprehensive

Context from clinical trial documents:
{context}

User Question: {query}

Please provide a detailed response based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly.

Response:"""

        return await self._generate_response(prompt, **kwargs)


class LLMService:
    """Main LLM service that manages Perplexity with local fallback"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.primary_provider = None
        self.fallback_provider = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize Perplexity as primary and local as fallback"""
        # Primary: Perplexity
        perplexity_key = self.config.get_llm_api_key()  # Your Perplexity API key
        if perplexity_key:
            self.primary_provider = PerplexityProvider(
                api_key=perplexity_key,
                model=self.config.llm.model_name,
            )
            print("Perplexity provider initialized")
        else:
            print("Warning: No Perplexity API key found")

        # Fallback: Local model
        try:
            self.fallback_provider = LocalLLMProvider(
                model=getattr(self.config.llm, 'local_model', 'llama3.2'),
                base_url=getattr(self.config.llm, 'local_base_url', 'http://localhost:11434')
            )
            print("Local LLM fallback initialized")
        except Exception as e:
            print(f"Warning: Local LLM fallback not available: {e}")

    async def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """Generate response with automatic fallback"""

        # Try Perplexity first
        if self.primary_provider:
            try:
                print("Trying Perplexity...")
                response = await self.primary_provider.generate_with_context(query, context, **kwargs)
                print("Perplexity response received")
                return response
            except Exception as e:
                print(f"Perplexity failed: {e}")

        # Fallback to local model
        if self.fallback_provider:
            try:
                print("Falling back to local model...")
                response = await self.fallback_provider.generate_with_context(query, context, **kwargs)
                print("Local model response received")
                return response
            except Exception as e:
                print(f"Local model failed: {e}")

        return "Error: No LLM providers are available. Please check your Perplexity API key or local model setup."

    async def chat_with_context(self,
                              query: str,
                              context: str,
                              conversation_history: List[Dict[str, str]] = None,
                              **kwargs) -> Dict[str, Any]:
        """Simple chat interface"""

        response = await self.generate_with_context(query, context, **kwargs)

        # Update conversation history
        if conversation_history is None:
            conversation_history = []

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep only last 10 messages to prevent context overflow
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return {
            "response": response,
            "conversation_history": conversation_history
        }

    def is_configured(self) -> bool:
        """Check if any LLM service is available"""
        return self.primary_provider is not None or self.fallback_provider is not None

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "perplexity_available": self.primary_provider is not None,
            "local_available": self.fallback_provider is not None,
            "configured": self.is_configured(),
            "primary_model": getattr(self.primary_provider, 'model', None),
            "fallback_model": getattr(self.fallback_provider, 'model', None)
        }


# Example usage
async def test_llm_service():
    """Test the LLM service"""
    service = LLMService()

    if not service.is_configured():
        print("Error: No LLM providers configured")
        return

    print("Service Status:", service.get_status())

    # Test query
    context = "This is a Phase II clinical trial studying the efficacy of Drug X in patients with condition Y."
    query = "What phase is this clinical trial?"

    response = await service.generate_with_context(query, context)
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(test_llm_service())