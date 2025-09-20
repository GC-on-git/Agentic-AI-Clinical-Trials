#!/usr/bin/env python3
"""
Test script for LLM integration in Clinical Trial AI
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_trial_ai.backend.config import get_config
from clinical_trial_ai.backend.services.ai_layer.llm_service import LLMService


async def test_llm_configuration():
    """Test LLM configuration and API key setup"""
    
    print("🔧 Testing LLM Configuration...")
    
    # Get configuration
    config = get_config()
    validation_results = config.validate()
    
    print("\n📋 Configuration Status:")
    for key, value in validation_results.items():
        if isinstance(value, dict):
            print(f"  {key.upper()}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key.upper()}: {value}")
    
    # Test LLM service
    print("\n🤖 Testing LLM Service...")
    llm_service = LLMService()
    status_info = llm_service.get_status()
    
    print(f"  Configured: {status_info['configured']}")
    print(f"  Provider: {status_info.get('provider', 'None')}")
    print(f"  Model: {status_info.get('model', 'None')}")
    print(f"  API Key Set: {status_info.get('api_key_set', False)}")
    
    if llm_service.is_configured():
        print("\n✅ LLM service is configured!")
        
        # Test a simple query
        print("\n🧪 Testing LLM Response Generation...")
        try:
            test_query = "What is a clinical trial?"
            test_context = """
            A clinical trial is a research study that tests new treatments, interventions, 
            or tests as a means to prevent, detect, treat, or manage various diseases or 
            medical conditions. Clinical trials are conducted in phases to determine 
            safety and efficacy.
            """
            
            response = await llm_service.generate_with_context(
                query=test_query,
                context=test_context,
                max_tokens=200,
                temperature=0.7
            )
            
            print(f"  Query: {test_query}")
            print(f"  Response: {response}")
            print("\n✅ LLM integration working correctly!")
            
        except Exception as e:
            print(f"❌ Error testing LLM: {e}")
            print("  This might be due to:")
            print("  - Invalid API key")
            print("  - Network connectivity issues")
            print("  - API rate limits")
            print("  - Model availability issues")
    
    else:
        print("\n❌ LLM service is not configured!")
        print("\n🔧 To fix this:")
        print("1. Set your API key in environment variables:")
        print("   export OPENAI_API_KEY=your-api-key-here")
        print("   # OR")
        print("   export ANTHROPIC_API_KEY=your-api-key-here")
        print("\n2. Or create a .env file with:")
        print("   LLM_PROVIDER=openai")
        print("   LLM_API_KEY=your-api-key-here")
        print("\n3. See LLM_SETUP.md for detailed instructions")


async def test_rag_with_llm():
    """Test RAG system with LLM integration"""
    
    print("\n🔍 Testing RAG System with LLM...")
    
    try:
        from clinical_trial_ai.backend.services.ai_layer.langgraph_flow import LangGraphFlow
        from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
        from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
        from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
        from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService
        
        # Initialize components
        text_processor = ClinicalTextProcessor()
        embedder = SBERTEmbedder()
        vector_store = ChromaVectorStore()
        storage_service = DocumentStorageService()
        
        # Initialize the main flow
        flow = LangGraphFlow(
            storage_service=storage_service,
            text_processor=text_processor,
            embedder=embedder,
            vector_store=vector_store
        )
        
        print("✅ RAG system initialized successfully!")
        
        # Test LLM status in RAG system
        llm_status = flow.llm_service.get_status()
        print(f"  LLM Status: {'✅ Configured' if llm_status['configured'] else '❌ Not Configured'}")
        
        if llm_status['configured']:
            print("  🎉 Your RAG system is ready for intelligent responses!")
        else:
            print("  ⚠️  RAG system will use fallback responses (still functional)")
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")


async def main():
    """Main test function"""
    
    print("🚀 Clinical Trial AI - LLM Integration Test")
    print("=" * 50)
    
    # Test configuration
    await test_llm_configuration()
    
    # Test RAG system
    await test_rag_with_llm()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")
    print("\n📚 Next steps:")
    print("1. If LLM is not configured, follow LLM_SETUP.md")
    print("2. Process some clinical trial documents")
    print("3. Test queries with: python example_rag_usage.py")


if __name__ == "__main__":
    asyncio.run(main())
