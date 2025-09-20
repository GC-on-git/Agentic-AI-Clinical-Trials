#!/usr/bin/env python3
"""
Example usage of the RAG Query Handler for Clinical Trial AI

This script demonstrates how to use the RAG system to query clinical trial documents.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_trial_ai.backend.services.ai_layer.langgraph_flow import LangGraphFlow
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService


async def main():
    """Example usage of the RAG system"""
    
    print("🚀 Initializing Clinical Trial RAG System...")
    
    # Initialize components
    text_processor = ClinicalTextProcessor()
    embedder = SBERTEmbedder()
    vector_store = ChromaVectorStore()
    storage_service = DocumentStorageService()  # You'll need to implement this
    
    # Initialize the main flow
    flow = LangGraphFlow(
        storage_service=storage_service,
        text_processor=text_processor,
        embedder=embedder,
        vector_store=vector_store
    )
    
    print("✅ System initialized successfully!")
    
    # Example 1: Process a document first (if you have one)
    print("\n📄 Processing a sample document...")
    path = "/Users/anuganch/Desktop/Mock3_final/mock_data.pdf"
    document_id = await flow.run_pipeline(path)
    print(f"Document processed with ID: {document_id}")
    
    # Example 2: Query the system
    print("\n❓ Querying the system...")
    
    queries = [
        "What are the primary endpoints of the clinical trial?",
        "What adverse events were reported?",
        "What is the efficacy of the treatment?",
        "What are the inclusion and exclusion criteria?",
        "What phase is this clinical trial in?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = await flow.process_query(query, top_k=3)
            print(f"📝 Response: {result['response']}")
            print(f"📊 Confidence: {result['insights']['confidence']:.3f}")
            print(f"📚 Chunks used: {result['metadata']['chunks_used']}")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Example 3: Keyword search
    print("\n🔎 Keyword search example...")
    keywords = ["efficacy", "safety", "adverse events"]
    try:
        keyword_results = await flow.search_by_keywords(keywords, top_k=5)
        print(f"Found {keyword_results['total_matches']} matches for keywords: {keywords}")
    except Exception as e:
        print(f"❌ Error in keyword search: {e}")
    
    # Example 4: Chat interface
    print("\n💬 Chat interface example...")
    conversation_history = []
    
    chat_queries = [
        "Tell me about the clinical trial design",
        "What were the main findings?",
        "Are there any safety concerns?"
    ]
    
    for query in chat_queries:
        print(f"\n👤 User: {query}")
        try:
            chat_result = await flow.chat_with_documents(
                query=query,
                conversation_history=conversation_history,
                top_k=3
            )
            print(f"🤖 Assistant: {chat_result['response']}")
            conversation_history = chat_result['conversation_history']
        except Exception as e:
            print(f"❌ Error in chat: {e}")
    
    print("\n✅ RAG system demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
