"""
LLM Service for Clinical Trial AI - Supports multiple LLM providers
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import aiohttp
from backend.config import get_config


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def generate_with_context(self, 
                                  query: str, 
                                  context: str, 
                                  **kwargs) -> str:
        """Generate response with context"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              max_tokens: int = 1000,
                              temperature: float = 0.7) -> str:
        """Generate response using OpenAI API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
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
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_with_context(self, 
                                  query: str, 
                                  context: str, 
                                  max_tokens: int = 1000,
                                  temperature: float = 0.7) -> str:
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
        
        return await self.generate_response(messages, max_tokens, temperature)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              max_tokens: int = 1000) -> str:
        """Generate response using Anthropic API"""
        try:
            # Convert OpenAI format to Anthropic format
            user_message = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_message += f"{msg['content']}\n"
                elif msg["role"] == "system":
                    user_message = f"System: {msg['content']}\n\n{user_message}"
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": user_message}]
                }
                
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_with_context(self, 
                                  query: str, 
                                  context: str, 
                                  max_tokens: int = 1000) -> str:
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
        
        return await self.generate_response(messages, max_tokens)


class LocalLLMProvider(LLMProvider):
    """Local LLM provider (placeholder for local models like Ollama)"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs) -> str:
        """Generate response using local LLM (Ollama)"""
        try:
            # Convert to Ollama format
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            
            prompt += "Assistant:"
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
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
            return f"Error generating response: {str(e)}"
    
    async def generate_with_context(self, 
                                  query: str, 
                                  context: str, 
                                  **kwargs) -> str:
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
        
        return await self.generate_response(messages, **kwargs)


class LLMService:
    """Main LLM service that manages different providers"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> Optional[LLMProvider]:
        """Initialize the appropriate LLM provider based on configuration"""
        api_key = self.config.get_llm_api_key()
        
        if not api_key and self.config.llm.provider != "local":
            print("Warning: No LLM API key found. LLM features will be disabled.")
            return None
        
        if self.config.llm.provider == "openai":
            return OpenAIProvider(
                api_key=api_key,
                model=self.config.llm.model_name,
                base_url=self.config.llm.base_url
            )
        elif self.config.llm.provider == "anthropic":
            return AnthropicProvider(
                api_key=api_key,
                model=self.config.llm.model_name
            )
        elif self.config.llm.provider == "local":
            return LocalLLMProvider(
                model=self.config.llm.model_name,
                base_url=self.config.llm.base_url or "http://localhost:11434"
            )
        else:
            print(f"Unknown LLM provider: {self.config.llm.provider}")
            return None
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs) -> str:
        """Generate response using the configured provider"""
        if not self.provider:
            return "LLM service not configured. Please set up your API key."
        
        return await self.provider.generate_response(messages, **kwargs)
    
    async def generate_with_context(self, 
                                  query: str, 
                                  context: str, 
                                  **kwargs) -> str:
        """Generate response with clinical trial context"""
        if not self.provider:
            return "LLM service not configured. Please set up your API key."
        
        return await self.provider.generate_with_context(query, context, **kwargs)
    
    async def chat_with_context(self, 
                              query: str, 
                              context: str,
                              conversation_history: List[Dict[str, str]] = None,
                              **kwargs) -> Dict[str, Any]:
        """Chat interface with context and conversation history"""
        if not self.provider:
            return {
                "response": "LLM service not configured. Please set up your API key.",
                "conversation_history": conversation_history or []
            }
        
        # Build messages with conversation history
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a clinical trial AI assistant. Help users understand clinical trial documents, protocols, and results."
        })
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Keep last 10 messages
        
        # Add current query with context
        user_message = f"""Context from clinical trial documents:
{context}

User Question: {query}

Please provide a detailed response based on the context above."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await self.provider.generate_response(messages, **kwargs)
        
        # Update conversation history
        if conversation_history is None:
            conversation_history = []
        
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "conversation_history": conversation_history
        }
    
    def is_configured(self) -> bool:
        """Check if LLM service is properly configured"""
        return self.provider is not None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        if not self.provider:
            return {"configured": False, "provider": None}
        
        return {
            "configured": True,
            "provider": self.config.llm.provider,
            "model": self.config.llm.model_name,
            "api_key_set": bool(self.config.get_llm_api_key())
        }
