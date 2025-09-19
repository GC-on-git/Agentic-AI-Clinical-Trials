"""
LLM Service for Clinical Trial AI - OpenRouter/Perplexity Support
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


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider"""

    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://clinical-trial-ai.com",  # Optional: your site URL
            "X-Title": "Clinical Trial AI"  # Optional: your app name
        }

    async def _generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenRouter API"""
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
                        raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
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


class LLMService:
    """Main LLM service that supports OpenRouter and Perplexity"""

    def __init__(self, config=None, provider_type: str = "openrouter"):
        self.config = config or get_config()
        self.provider_type = provider_type.lower()
        self.provider = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the selected provider"""
        if self.provider_type == "openrouter":
            api_key = self.config.get_openrouter_api_key()  # You'll need to add this method
            if api_key:
                self.provider = OpenRouterProvider(
                    api_key=api_key,
                    model=getattr(self.config.llm, 'openrouter_model', 'anthropic/claude-3.5-sonnet')
                )
                print("OpenRouter provider initialized")
            else:
                print("Warning: No OpenRouter API key found")

        elif self.provider_type == "perplexity":
            api_key = self.config.get_perplexity_api_key()  # You'll need to add this method
            if api_key:
                self.provider = PerplexityProvider(
                    api_key=api_key,
                    model=getattr(self.config.llm, 'perplexity_model', 'llama-3.1-sonar-small-128k-online')
                )
                print("Perplexity provider initialized")
            else:
                print("Warning: No Perplexity API key found")

        else:
            print(f"Error: Unknown provider type '{self.provider_type}'. Use 'openrouter' or 'perplexity'")

    def switch_provider(self, provider_type: str):
        """Switch between OpenRouter and Perplexity"""
        self.provider_type = provider_type.lower()
        self._initialize_provider()

    async def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """Generate response using the configured provider"""
        if not self.provider:
            return f"Error: No {self.provider_type} provider available. Please check your API key configuration."

        try:
            print(f"Using {self.provider_type} provider...")
            response = await self.provider.generate_with_context(query, context, **kwargs)
            print(f"{self.provider_type.capitalize()} response received")
            return response
        except Exception as e:
            print(f"{self.provider_type.capitalize()} failed: {e}")
            return f"Error: Failed to generate response using {self.provider_type}. {str(e)}"

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
            "conversation_history": conversation_history,
            "provider_used": self.provider_type
        }

    def is_configured(self) -> bool:
        """Check if the LLM service is available"""
        return self.provider is not None

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "provider_type": self.provider_type,
            "configured": self.is_configured(),
            "model": getattr(self.provider, 'model', None) if self.provider else None,
            "available_providers": ["openrouter", "perplexity"]
        }

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider"""
        return {
            "openrouter": [
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-haiku",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "meta-llama/llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "google/gemini-pro-1.5"
            ],
            "perplexity": [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-huge-128k-online"
            ]
        }


# Example usage
async def test_llm_service():
    """Test the LLM service"""
    # Test OpenRouter
    print("=== Testing OpenRouter ===")
    service_or = LLMService(provider_type="openrouter")

    if service_or.is_configured():
        print("OpenRouter Status:", service_or.get_status())

        context = "This is a Phase II clinical trial studying the efficacy of Drug X in patients with condition Y."
        query = "What phase is this clinical trial?"

        response = await service_or.generate_with_context(query, context)
        print(f"OpenRouter Response: {response}")
    else:
        print("OpenRouter not configured")

    print("\n=== Testing Perplexity ===")
    # Test Perplexity
    service_px = LLMService(provider_type="perplexity")

    if service_px.is_configured():
        print("Perplexity Status:", service_px.get_status())

        context = "This is a Phase II clinical trial studying the efficacy of Drug X in patients with condition Y."
        query = "What phase is this clinical trial?"

        response = await service_px.generate_with_context(query, context)
        print(f"Perplexity Response: {response}")
    else:
        print("Perplexity not configured")

    # Test switching providers
    print("\n=== Testing Provider Switching ===")
    service = LLMService(provider_type="openrouter")
    print("Available models:", service.get_available_models())

    print("Switching to Perplexity...")
    service.switch_provider("perplexity")
    print("New status:", service.get_status())


if __name__ == "__main__":
    asyncio.run(test_llm_service())