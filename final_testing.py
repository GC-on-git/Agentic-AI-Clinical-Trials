#!/usr/bin/env python3
"""
Interactive Demo: Clinical Trial AI - User Query Processing
Demonstrates how agents work together to process user queries and generate responses
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_trial_ai.backend.services.ai_layer.langgraph_flow import LangGraphFlow
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService


class ClinicalTrialAIDemo:
    """Interactive demo for Clinical Trial AI system"""
    
    def __init__(self):
        self.flow = None
        self.conversation_history = []
        self.session_start = datetime.now()
        
    async def initialize_system(self):
        """Initialize the Clinical Trial AI system"""
        print("🚀 Initializing Clinical Trial AI System...")
        print("=" * 60)
        
        # Initialize components
        text_processor = ClinicalTextProcessor()
        embedder = SBERTEmbedder()
        vector_store = ChromaVectorStore()
        storage_service = DocumentStorageService()
        
        # Initialize the enhanced flow with agentic capabilities
        self.flow = LangGraphFlow(
            storage_service=storage_service,
            text_processor=text_processor,
            embedder=embedder,
            vector_store=vector_store
        )
        
        print("✅ System initialized successfully!")
        print("🤖 Agentic AI capabilities loaded:")
        print("   - Tool Orchestration")
        print("   - State Management")
        print("   - Contextual Awareness")
        print("   - LLM Integration")
        
        # Ensure document is processed
        print("\n📄 Processing clinical trial document...")
        file_path = "/Users/anuganch/Desktop/Mock3_final/mock_data.pdf"
        document_id = await self.flow.upload_tool.process_file(file_path)
        print(f"📝 Document ready: {document_id}")
        
        return True
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print("\n" + "=" * 60)
        print("🏥 CLINICAL TRIAL AI - INTERACTIVE DEMO")
        print("=" * 60)
        print("🤖 This demo shows how multiple AI agents work together")
        print("   to process your queries and generate intelligent responses.")
        print("\n💡 Try asking questions like:")
        print("   • What clinical trials are mentioned?")
        print("   • Tell me about the phase II trial with pembrolizumab")
        print("   • What are the different types of cancer treatments?")
        print("   • Are there any safety studies mentioned?")
        print("   • What is the most advanced phase trial?")
        print("\n🎯 Type 'help' for more options, 'quit' to exit")
        print("=" * 60)
    
    def display_help(self):
        """Display help information"""
        print("\n📚 HELP - Available Commands:")
        print("=" * 40)
        print("🔍 Query Commands:")
        print("   • Ask any question about clinical trials")
        print("   • Use natural language (e.g., 'What trials are there?')")
        print("\n🛠️  System Commands:")
        print("   • 'help' - Show this help message")
        print("   • 'status' - Show system status and agent information")
        print("   • 'history' - Show conversation history")
        print("   • 'agents' - Show available agents and their capabilities")
        print("   • 'clear' - Clear conversation history")
        print("   • 'quit' - Exit the demo")
        print("\n🎯 Example Queries:")
        print("   • 'What clinical trials are mentioned in the document?'")
        print("   • 'Tell me about the pembrolizumab trial'")
        print("   • 'What are the safety endpoints?'")
        print("   • 'Compare the different treatment approaches'")
        print("=" * 40)
    
    async def show_system_status(self):
        """Show system status and agent information"""
        print("\n🔧 SYSTEM STATUS")
        print("=" * 30)
        
        # Get system state
        system_state = self.flow.get_system_state()
        print(f"📊 Current State: {system_state['current_state']}")
        print(f"🎯 User Goals: {len(system_state['user_goals'])}")
        print(f"🔧 Available Tools: {len(system_state['available_tools'])}")
        
        # Show LLM status
        llm_status = self.flow.llm_service.get_status()
        print(f"🤖 LLM Provider: {llm_status['provider_type']}")
        print(f"✅ LLM Configured: {llm_status['configured']}")
        if llm_status['model']:
            print(f"🧠 Model: {llm_status['model']}")
        
        # Show available agents
        available_tools = await self.flow.discover_available_tools()
        print(f"\n🤖 Available Agents ({len(available_tools)}):")
        for tool in available_tools:
            print(f"   • {tool['name']}: {tool['description']}")
        
        print("=" * 30)
    
    async def show_agent_capabilities(self):
        """Show detailed agent capabilities"""
        print("\n🤖 AGENT CAPABILITIES")
        print("=" * 40)
        
        available_tools = await self.flow.discover_available_tools()
        
        for tool in available_tools:
            print(f"\n🔧 {tool['name'].upper()}:")
            print(f"   Description: {tool['description']}")
            print(f"   Input Schema: {tool['inputSchema']['properties'].keys()}")
            print(f"   Output Schema: {tool['outputSchema']['properties'].keys()}")
        
        print("\n🎯 Agent Collaboration:")
        print("   • Search Agent: Finds relevant documents")
        print("   • Ranking Agent: Prioritizes information by relevance")
        print("   • Context Agent: Prepares data for processing")
        print("   • LLM Agent: Generates intelligent responses")
        print("   • Insights Agent: Provides confidence and analysis")
        print("=" * 40)
    
    def show_conversation_history(self):
        """Show conversation history"""
        print("\n📜 CONVERSATION HISTORY")
        print("=" * 40)
        
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. {entry['timestamp']}")
            print(f"   User: {entry['user_query']}")
            print(f"   Agent Response: {entry['response'][:100]}...")
            print(f"   Confidence: {entry['confidence']:.3f}")
            print(f"   Chunks Used: {entry['chunks_used']}")
        
        print("=" * 40)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared!")
    
    async def process_user_query(self, query: str):
        """Process user query using the agentic AI system"""
        print(f"\n🔍 Processing: '{query}'")
        print("-" * 50)
        
        start_time = datetime.now()
        
        try:
            # Process the query using the agentic system
            result = await self.flow.process_query(query, top_k=5)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Display the response
            print(f"🤖 Agent Response:")
            print(f"   {result['response']}")
            
            # Display metadata
            print(f"\n📊 Response Metadata:")
            print(f"   ⏱️  Processing Time: {processing_time:.2f} seconds")
            print(f"   📈 Confidence: {result['insights']['confidence']:.3f}")
            print(f"   📚 Chunks Retrieved: {result['metadata']['chunks_retrieved']}")
            print(f"   📝 Chunks Used: {result['metadata']['chunks_used']}")
            print(f"   🔍 Query Embedding Dim: {result['metadata']['query_embedding_dim']}")
            
            # Show relevant chunks if available
            if result['relevant_chunks']:
                print(f"\n🔍 Relevant Information Found:")
                for i, chunk in enumerate(result['relevant_chunks'][:3], 1):
                    print(f"   {i}. {chunk['content'][:100]}...")
                    print(f"      Relevance: {chunk.get('relevance_score', 0):.3f}")
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': start_time.strftime("%H:%M:%S"),
                'user_query': query,
                'response': result['response'],
                'confidence': result['insights']['confidence'],
                'chunks_used': result['metadata']['chunks_used'],
                'processing_time': processing_time
            })
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("Please try again with a different query.")
    
    async def run_demo_queries(self):
        """Run predefined demo queries to show the system in action"""
        print("\n🎯 Running Demo Queries to Show Agentic AI in Action")
        print("=" * 60)
        
        demo_queries = [
            "What clinical trials are mentioned in the document?",
            "Tell me about the phase II trial with pembrolizumab",
            "What are the different types of cancer treatments being studied?",
            "Are there any safety studies mentioned?",
            "What is the most advanced phase trial mentioned?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*20} DEMO QUERY {i} {'='*20}")
            await self.process_user_query(query)
            
            # Add a pause between queries for better readability
            if i < len(demo_queries):
                print("\n⏳ Processing next query...")
                await asyncio.sleep(1)
        
        print(f"\n{'='*60}")
        print("🎉 Demo completed! All queries processed by the agentic AI system.")
        print(f"📊 Session Summary:")
        print(f"   • Queries Processed: {len(self.conversation_history)}")
        print(f"   • Session Duration: {datetime.now() - self.session_start}")
        print(f"   • Average Processing Time: {sum(h['processing_time'] for h in self.conversation_history) / len(self.conversation_history):.2f}s")
    
    async def run_interactive_demo(self):
        """Run the interactive demo"""
        # Initialize system
        if not await self.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Display welcome
        self.display_welcome()
        
        # Ask user what they want to do
        print("\n🎯 What would you like to do?")
        print("1. Run demo queries (automated)")
        print("2. Interactive mode (manual input)")
        print("3. Show system status")
        print("4. Show agent capabilities")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                await self.run_demo_queries()
            elif choice == "2":
                await self.run_manual_interactive()
            elif choice == "3":
                await self.show_system_status()
            elif choice == "4":
                await self.show_agent_capabilities()
            elif choice == "5":
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice. Running demo queries by default.")
                await self.run_demo_queries()
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Running demo queries by default...")
            await self.run_demo_queries()
    
    async def run_manual_interactive(self):
        """Run manual interactive mode"""
        print("\n🔄 Manual Interactive Mode")
        print("Type your queries below. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\n💬 Your Query: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Exiting interactive mode.")
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                elif user_input.lower() == 'status':
                    await self.show_system_status()
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                elif user_input.lower() == 'agents':
                    await self.show_agent_capabilities()
                elif user_input.lower() == 'clear':
                    self.clear_history()
                elif not user_input:
                    print("Please enter a query or command.")
                else:
                    await self.process_user_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main function to run the demo"""
    demo = ClinicalTrialAIDemo()
    await demo.run_interactive_demo()


if __name__ == "__main__":
    print("🏥 Clinical Trial AI - Interactive Demo")
    print("Loading agentic AI system...")
    asyncio.run(main())