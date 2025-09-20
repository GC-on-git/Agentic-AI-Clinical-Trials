import asyncio
from typing import Dict, Any, List
from clinical_trial_ai.backend.services.text_processing.preprocessing import ClinicalTextProcessor
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.storage.object_store import DocumentStorageService
# from backend.services.data_collection.user_uploads import FileUploadCollector

from .upload_tool import UploadTool
from .predictive_models import PredictiveModels
from .reasoning_tool import ReasoningTool
from .search_tool import SearchTool
from .summary_tool import SummaryTool
from .rag_query_tool import RAGQueryHandler
from .llm_service import LLMService
from .tool_orchestrator import ToolOrchestrator, StateManager, ContextualAwareness
from .mcp_tools import (
    MCPSearchTool, MCPPredictTool, MCPReasonTool, MCPSummarizeTool,
    MCPAnalyzeTool, MCPExtractTool, MCPClassifyTool
)


class LangGraphFlow:
    """Orchestrates the complete AI-driven document pipeline"""

    def __init__(self,
                 storage_service: DocumentStorageService,
                 text_processor: ClinicalTextProcessor,
                 embedder: SBERTEmbedder,
                 vector_store: ChromaVectorStore):
        self.upload_tool = UploadTool(storage_service, text_processor, embedder, vector_store)
        self.predictive_models = PredictiveModels()
        self.reasoning_tool = ReasoningTool()
        self.search_tool = SearchTool(vector_store)
        self.summary_tool = SummaryTool()
        self.llm_service = LLMService()
        self.rag_query_handler = RAGQueryHandler(
            embedder=embedder,
            vector_store=vector_store,
            search_tool=self.search_tool,
            summary_tool=self.summary_tool,
            reasoning_tool=self.reasoning_tool,
            llm_service=self.llm_service
        )
        
        # Initialize agentic AI components
        self.tool_orchestrator = ToolOrchestrator()
        self.state_manager = StateManager()
        self.contextual_awareness = ContextualAwareness()
        
        # Register MCP-compliant tools
        self._register_mcp_tools()

    async def run_pipeline(self, file_path: str) -> Dict[str, Any]:
        """Run full pipeline: upload → chunk → embed → summarize"""
        document_id = await self.upload_tool.process_file(file_path)

        # Retrieve stored chunks
        chunks = await self.search_tool.get_chunks_by_document_id(document_id)

        # Generate predictions or insights
        predictions = await self.predictive_models.predict(chunks)

        # Reasoning
        reasoning_result = await self.reasoning_tool.analyze(predictions)

        # Summarize
        summary = await self.summary_tool.generate_summary(chunks)

        return {
            "document_id": document_id,
            "predictions": predictions,
            "reasoning": reasoning_result,
            "summary": summary
        }

    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a user query using RAG pipeline"""
        return await self.rag_query_handler.process_query(query, top_k=top_k)

    async def search_by_keywords(self, keywords: List[str], top_k: int = 10) -> Dict[str, Any]:
        """Search for documents containing specific keywords"""
        return await self.rag_query_handler.search_by_keywords(keywords, top_k=top_k)

    async def get_document_overview(self, document_id: str) -> Dict[str, Any]:
        """Get overview of a specific document"""
        return await self.rag_query_handler.get_document_overview(document_id)

    async def chat_with_documents(self, 
                                query: str, 
                                conversation_history: List[Dict[str, str]] = None,
                                top_k: int = 5) -> Dict[str, Any]:
        """
        Chat interface for querying documents with conversation context
        
        Args:
            query: Current user query
            conversation_history: Previous conversation turns
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Response with answer, context, and updated conversation history
        """
        # Process the current query
        query_result = await self.process_query(query, top_k=top_k)
        
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Add current turn to history
        conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        conversation_history.append({
            "role": "assistant", 
            "content": query_result["response"],
            "context": query_result["context"],
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return {
            "response": query_result["response"],
            "context": query_result["context"],
            "relevant_chunks": query_result["relevant_chunks"],
            "insights": query_result["insights"],
            "conversation_history": conversation_history,
            "metadata": query_result["metadata"]
        }

    def _register_mcp_tools(self):
        """Register MCP-compliant tools with the orchestrator"""
        self.tool_orchestrator.register_tool(MCPSearchTool(self.search_tool))
        self.tool_orchestrator.register_tool(MCPPredictTool(self.predictive_models))
        self.tool_orchestrator.register_tool(MCPReasonTool(self.reasoning_tool))
        self.tool_orchestrator.register_tool(MCPSummarizeTool(self.summary_tool))
        self.tool_orchestrator.register_tool(MCPAnalyzeTool(self.predictive_models))
        self.tool_orchestrator.register_tool(MCPExtractTool(self.predictive_models))
        self.tool_orchestrator.register_tool(MCPClassifyTool(self.predictive_models))

    async def agentic_query(self, 
                          query: str, 
                          conversation_history: List[Dict[str, str]] = None,
                          user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Advanced agentic query processing with tool orchestration, state management, and contextual awareness
        """
        
        # Update state and goals
        self.state_manager.set_state("processing", {"query": query})
        if user_goals:
            for goal in user_goals:
                self.state_manager.add_goal(goal)
        
        # Analyze user intent and context
        intent_analysis = self.contextual_awareness.analyze_user_intent(query, conversation_history)
        
        # Make decision about approach
        decision = self.state_manager.make_decision(
            options=["search", "analyze", "predict", "summarize"],
            context=intent_analysis
        )
        
        # Orchestrate tool execution
        orchestration_result = await self.tool_orchestrator.orchestrate_query(
            query=query,
            context={
                "intent_analysis": intent_analysis,
                "decision": decision,
                "conversation_history": conversation_history or []
            }
        )
        
        # Generate enhanced response with contextual awareness
        base_response = orchestration_result.get("results", {}).get("search_tool", {}).get("data", [])
        if base_response and self.llm_service.is_configured():
            # Use LLM with enhanced context
            context_text = self._prepare_agentic_context(orchestration_result)
            enhanced_response = await self.llm_service.generate_with_context(
                query=query,
                context=context_text
            )
        else:
            # Fallback to basic response
            enhanced_response = f"Based on the analysis: {query}"
        
        # Apply contextual enhancement
        final_response = self.contextual_awareness.enhance_response_context(
            enhanced_response, intent_analysis
        )
        
        # Update state
        self.state_manager.set_state("completed", {"response_generated": True})
        
        return {
            "response": final_response,
            "intent_analysis": intent_analysis,
            "decision": decision,
            "orchestration_result": orchestration_result,
            "state": self.state_manager.current_state,
            "context": self.tool_orchestrator.context,
            "metadata": {
                "tools_used": list(orchestration_result.get("tool_sequence", [])),
                "execution_success": orchestration_result.get("success", False),
                "conversation_depth": len(conversation_history or [])
            }
        }

    def _prepare_agentic_context(self, orchestration_result: Dict[str, Any]) -> str:
        """Prepare context for LLM from orchestration results"""
        context_parts = []
        
        # Add search results
        search_result = orchestration_result.get("results", {}).get("search_tool")
        if search_result and search_result.get("success"):
            search_data = search_result.get("data", [])
            if search_data:
                context_parts.append("Relevant Documents:")
                for i, doc in enumerate(search_data[:3]):
                    context_parts.append(f"{i+1}. {doc.get('content', '')[:300]}...")
        
        # Add analysis results
        analysis_result = orchestration_result.get("results", {}).get("analyze_tool")
        if analysis_result and analysis_result.get("success"):
            analysis_data = analysis_result.get("data", {})
            if analysis_data:
                context_parts.append("\nAnalysis Results:")
                context_parts.append(f"Document Types: {', '.join(analysis_data.get('document_types', []))}")
                context_parts.append(f"Risk Levels: {', '.join(analysis_data.get('risk_levels', []))}")
                context_parts.append(f"Medical Terms: {', '.join(analysis_data.get('medical_terms', [])[:10])}")
        
        # Add prediction results
        predict_result = orchestration_result.get("results", {}).get("predict_tool")
        if predict_result and predict_result.get("success"):
            predict_data = predict_result.get("data", [])
            if predict_data:
                context_parts.append("\nPredictions:")
                for pred in predict_data[:3]:
                    context_parts.append(f"- {pred.get('prediction', 'Unknown')} (confidence: {pred.get('confidence', 0):.2f})")
        
        return "\n".join(context_parts)

    async def discover_available_tools(self) -> List[Dict[str, Any]]:
        """MCP-compliant tool discovery"""
        return await self.tool_orchestrator.discover_tools()

    async def execute_specific_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool directly"""
        result = await self.tool_orchestrator.execute_tool(tool_name, **kwargs)
        return {
            "tool": tool_name,
            "success": result.success,
            "data": result.data,
            "metadata": result.metadata,
            "error": result.error,
            "execution_time": result.execution_time
        }

    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state and context"""
        return {
            "current_state": self.state_manager.current_state,
            "state_history": self.state_manager.state_history[-5:],  # Last 5 states
            "user_goals": self.state_manager.user_goals,
            "execution_history": self.tool_orchestrator.execution_history[-10:],  # Last 10 executions
            "context": self.tool_orchestrator.context,
            "available_tools": list(self.tool_orchestrator.tools.keys())
        }

    def set_user_goal(self, goal: str, priority: int = 1):
        """Set a user goal for the agent to pursue"""
        self.state_manager.add_goal(goal, priority)

    def update_goal_status(self, goal: str, status: str):
        """Update the status of a user goal"""
        self.state_manager.update_goal_status(goal, status)
