#!/usr/bin/env python3
"""
Test script for the new agentic AI features: Tool Orchestration, State Management, and Contextual Awareness
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


async def test_agentic_features():
    """Test the new agentic AI features"""
    
    print("🚀 Testing Agentic AI Features")
    print("=" * 50)
    
    # Initialize components
    text_processor = ClinicalTextProcessor()
    embedder = SBERTEmbedder()
    vector_store = ChromaVectorStore()
    storage_service = DocumentStorageService()
    
    # Initialize the enhanced flow
    flow = LangGraphFlow(
        storage_service=storage_service,
        text_processor=text_processor,
        embedder=embedder,
        vector_store=vector_store
    )
    
    print("✅ System initialized with agentic capabilities!")
    
    # Test 1: Tool Discovery (MCP-compliant)
    print("\n🔧 Test 1: Tool Discovery")
    try:
        available_tools = await flow.discover_available_tools()
        print(f"📋 Available Tools ({len(available_tools)}):")
        for tool in available_tools:
            print(f"  - {tool['name']}: {tool['description']}")
    except Exception as e:
        print(f"❌ Error in tool discovery: {e}")
    
    # Test 2: State Management
    print("\n🎯 Test 2: State Management")
    try:
        # Set user goals
        flow.set_user_goal("Understand clinical trial efficacy", priority=1)
        flow.set_user_goal("Analyze safety profiles", priority=2)
        
        # Get system state
        system_state = flow.get_system_state()
        print(f"📊 Current State: {system_state['current_state']}")
        print(f"🎯 User Goals: {len(system_state['user_goals'])}")
        for goal in system_state['user_goals']:
            print(f"  - {goal['goal']} (priority: {goal['priority']})")
        print(f"🔧 Available Tools: {len(system_state['available_tools'])}")
    except Exception as e:
        print(f"❌ Error in state management: {e}")
    
    # Test 3: Contextual Awareness
    print("\n🧠 Test 3: Contextual Awareness")
    try:
        # Test different query types
        test_queries = [
            "What are the primary endpoints in phase III trials?",
            "How do I compare the efficacy of different treatments?",
            "Why did the trial show significant results?",
            "Find all safety-related adverse events"
        ]
        
        for query in test_queries:
            intent_analysis = flow.contextual_awareness.analyze_user_intent(query)
            print(f"\n🔍 Query: {query}")
            print(f"  Intent: {intent_analysis['intent']}")
            print(f"  Confidence: {intent_analysis['confidence']:.2f}")
            if intent_analysis['domain_context']:
                print(f"  Domain Context: {intent_analysis['domain_context']}")
    except Exception as e:
        print(f"❌ Error in contextual awareness: {e}")
    
    # Test 4: Tool Orchestration
    print("\n🎼 Test 4: Tool Orchestration")
    try:
        # Test orchestration with different query types
        orchestration_queries = [
            "What is the efficacy of the treatment?",
            "Analyze the safety profile and predict outcomes",
            "Summarize the clinical trial results"
        ]
        
        for query in orchestration_queries:
            print(f"\n🔍 Orchestrating: {query}")
            orchestration_result = await flow.tool_orchestrator.orchestrate_query(query)
            
            print(f"  Tools Used: {len(orchestration_result['tool_sequence'])}")
            for tool_name, params in orchestration_result['tool_sequence']:
                print(f"    - {tool_name}")
            
            print(f"  Success: {orchestration_result['success']}")
            print(f"  Results: {len(orchestration_result['results'])}")
    except Exception as e:
        print(f"❌ Error in tool orchestration: {e}")
    
    # Test 5: Agentic Query Processing
    print("\n🤖 Test 5: Agentic Query Processing")
    try:
        # Test the new agentic_query method
        agentic_queries = [
            "What are the primary endpoints and how effective is the treatment?",
            "Compare the safety profiles of different study arms",
            "Predict the likelihood of trial success based on current data"
        ]
        
        for query in agentic_queries:
            print(f"\n🤖 Agentic Query: {query}")
            
            result = await flow.agentic_query(
                query=query,
                user_goals=["Understand trial outcomes", "Assess safety"]
            )
            
            print(f"  Intent: {result['intent_analysis']['intent']}")
            print(f"  Decision: {result['decision']['decision']}")
            print(f"  State: {result['state']}")
            print(f"  Tools Used: {len(result['metadata']['tools_used'])}")
            print(f"  Success: {result['metadata']['execution_success']}")
            print(f"  Response Preview: {result['response'][:100]}...")
    except Exception as e:
        print(f"❌ Error in agentic query processing: {e}")
    
    # Test 6: Specific Tool Execution
    print("\n🔧 Test 6: Specific Tool Execution")
    try:
        # Test executing specific tools
        tool_tests = [
            ("search_tool", {"query": "clinical trial efficacy", "top_k": 3}),
            ("classify_tool", {"query": "classify document types", "chunks": []}),
        ]
        
        for tool_name, params in tool_tests:
            print(f"\n🔧 Testing {tool_name}")
            result = await flow.execute_specific_tool(tool_name, **params)
            print(f"  Success: {result['success']}")
            print(f"  Execution Time: {result['execution_time']:.3f}s")
            if result['error']:
                print(f"  Error: {result['error']}")
    except Exception as e:
        print(f"❌ Error in specific tool execution: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Agentic Features Test Completed!")
    print("\n📊 Summary of New Capabilities:")
    print("✅ Tool Orchestration (MCP-compliant)")
    print("✅ State Management & Decision Making")
    print("✅ Contextual Awareness & Intent Analysis")
    print("✅ Dynamic Tool Selection")
    print("✅ Goal-Oriented Behavior")
    print("✅ Enhanced Response Generation")


async def main():
    """Main test function"""
    await test_agentic_features()


if __name__ == "__main__":
    asyncio.run(main())
