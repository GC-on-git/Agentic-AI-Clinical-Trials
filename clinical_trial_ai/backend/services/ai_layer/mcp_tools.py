"""
MCP-compliant tool implementations for Clinical Trial AI
"""

import asyncio
from typing import Dict, Any, List
from .tool_orchestrator import BaseTool, ToolDefinition, ToolResult, ToolType
from .search_tool import SearchTool
from .predictive_models import PredictiveModels
from .reasoning_tool import ReasoningTool
from .summary_tool import SummaryTool


class MCPSearchTool(BaseTool):
    """MCP-compliant search tool"""
    
    def __init__(self, search_tool: SearchTool):
        self.search_tool = search_tool
        
    async def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """Execute search with MCP compliance"""
        try:
            results = await self.search_tool.search_similar(query, top_k)
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "results_count": len(results)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_tool",
            description="Search for relevant clinical trial documents and chunks",
            tool_type=ToolType.SEARCH,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array", "description": "Search results"},
                    "metadata": {"type": "object", "description": "Search metadata"}
                }
            },
            handler=self.execute
        )


class MCPPredictTool(BaseTool):
    """MCP-compliant prediction tool"""
    
    def __init__(self, predictive_models: PredictiveModels):
        self.predictive_models = predictive_models
        
    async def execute(self, query: str, chunks: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute prediction with MCP compliance"""
        try:
            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No chunks provided for prediction"
                )
            
            predictions = await self.predictive_models.predict(chunks)
            return ToolResult(
                success=True,
                data=predictions,
                metadata={
                    "query": query,
                    "chunks_analyzed": len(chunks),
                    "predictions_count": len(predictions)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="predict_tool",
            description="Generate predictions and insights from clinical trial data",
            tool_type=ToolType.PREDICT,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Prediction query"},
                    "chunks": {"type": "array", "description": "Document chunks to analyze"}
                },
                "required": ["query", "chunks"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "predictions": {"type": "array", "description": "Prediction results"},
                    "metadata": {"type": "object", "description": "Prediction metadata"}
                }
            },
            handler=self.execute
        )


class MCPReasonTool(BaseTool):
    """MCP-compliant reasoning tool"""
    
    def __init__(self, reasoning_tool: ReasoningTool):
        self.reasoning_tool = reasoning_tool
        
    async def execute(self, query: str, predictions: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute reasoning with MCP compliance"""
        try:
            if not predictions:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No predictions provided for reasoning"
                )
            
            reasoning = await self.reasoning_tool.analyze(predictions)
            return ToolResult(
                success=True,
                data=reasoning,
                metadata={
                    "query": query,
                    "predictions_analyzed": len(predictions)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="reason_tool",
            description="Analyze predictions and provide reasoning insights",
            tool_type=ToolType.REASON,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Reasoning query"},
                    "predictions": {"type": "array", "description": "Predictions to analyze"}
                },
                "required": ["query", "predictions"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "reasoning": {"type": "object", "description": "Reasoning analysis"},
                    "metadata": {"type": "object", "description": "Reasoning metadata"}
                }
            },
            handler=self.execute
        )


class MCPSummarizeTool(BaseTool):
    """MCP-compliant summarization tool"""
    
    def __init__(self, summary_tool: SummaryTool):
        self.summary_tool = summary_tool
        
    async def execute(self, query: str, chunks: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute summarization with MCP compliance"""
        try:
            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No chunks provided for summarization"
                )
            
            summary = await self.summary_tool.generate_summary(chunks)
            return ToolResult(
                success=True,
                data={"summary": summary},
                metadata={
                    "query": query,
                    "chunks_summarized": len(chunks),
                    "summary_length": len(summary)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="summarize_tool",
            description="Generate summaries of clinical trial documents",
            tool_type=ToolType.SUMMARIZE,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Summarization query"},
                    "chunks": {"type": "array", "description": "Document chunks to summarize"}
                },
                "required": ["query", "chunks"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Generated summary"},
                    "metadata": {"type": "object", "description": "Summary metadata"}
                }
            },
            handler=self.execute
        )


class MCPAnalyzeTool(BaseTool):
    """MCP-compliant analysis tool"""
    
    def __init__(self, predictive_models: PredictiveModels):
        self.predictive_models = predictive_models
        
    async def execute(self, query: str, chunks: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute analysis with MCP compliance"""
        try:
            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No chunks provided for analysis"
                )
            
            # Perform comprehensive analysis
            predictions = await self.predictive_models.predict(chunks)
            
            # Extract key insights
            analysis = {
                "document_types": list(set(p.get("document_type", "unknown") for p in predictions)),
                "risk_levels": list(set(p.get("risk_level", "unknown") for p in predictions)),
                "outcome_predictions": list(set(p.get("outcome_prediction", "unknown") for p in predictions)),
                "confidence_scores": [p.get("confidence", 0) for p in predictions],
                "medical_terms": []
            }
            
            # Aggregate medical terms
            for pred in predictions:
                clinical_info = pred.get("clinical_info", {})
                if clinical_info.get("medical_terms"):
                    analysis["medical_terms"].extend(clinical_info["medical_terms"])
            
            analysis["medical_terms"] = list(set(analysis["medical_terms"]))
            
            return ToolResult(
                success=True,
                data=analysis,
                metadata={
                    "query": query,
                    "chunks_analyzed": len(chunks),
                    "predictions_count": len(predictions)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="analyze_tool",
            description="Perform comprehensive analysis of clinical trial data",
            tool_type=ToolType.ANALYZE,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Analysis query"},
                    "chunks": {"type": "array", "description": "Document chunks to analyze"}
                },
                "required": ["query", "chunks"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "object", "description": "Analysis results"},
                    "metadata": {"type": "object", "description": "Analysis metadata"}
                }
            },
            handler=self.execute
        )


class MCPExtractTool(BaseTool):
    """MCP-compliant extraction tool"""
    
    def __init__(self, predictive_models: PredictiveModels):
        self.predictive_models = predictive_models
        
    async def execute(self, query: str, chunks: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute extraction with MCP compliance"""
        try:
            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No chunks provided for extraction"
                )
            
            # Extract entities and information
            extractions = []
            for chunk in chunks:
                content = chunk.get("content", "")
                
                # Extract clinical information
                clinical_info = self.predictive_models._extract_clinical_info(content)
                
                # Extract entities
                entities = self.predictive_models._extract_entities(content)
                
                extractions.append({
                    "chunk_id": chunk.get("id", "unknown"),
                    "clinical_info": clinical_info,
                    "entities": entities,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })
            
            return ToolResult(
                success=True,
                data=extractions,
                metadata={
                    "query": query,
                    "chunks_processed": len(chunks),
                    "extractions_count": len(extractions)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="extract_tool",
            description="Extract entities and clinical information from documents",
            tool_type=ToolType.EXTRACT,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Extraction query"},
                    "chunks": {"type": "array", "description": "Document chunks to extract from"}
                },
                "required": ["query", "chunks"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "extractions": {"type": "array", "description": "Extracted information"},
                    "metadata": {"type": "object", "description": "Extraction metadata"}
                }
            },
            handler=self.execute
        )


class MCPClassifyTool(BaseTool):
    """MCP-compliant classification tool"""
    
    def __init__(self, predictive_models: PredictiveModels):
        self.predictive_models = predictive_models
        
    async def execute(self, query: str, chunks: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute classification with MCP compliance"""
        try:
            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="No chunks provided for classification"
                )
            
            # Classify documents
            classifications = []
            for chunk in chunks:
                content = chunk.get("content", "")
                doc_type = await self.predictive_models._predict_document_type(content)
                
                classifications.append({
                    "chunk_id": chunk.get("id", "unknown"),
                    "document_type": doc_type,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })
            
            return ToolResult(
                success=True,
                data=classifications,
                metadata={
                    "query": query,
                    "chunks_classified": len(chunks),
                    "unique_types": list(set(c["document_type"] for c in classifications))
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="classify_tool",
            description="Classify clinical trial documents by type and category",
            tool_type=ToolType.CLASSIFY,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Classification query"},
                    "chunks": {"type": "array", "description": "Document chunks to classify"}
                },
                "required": ["query", "chunks"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "classifications": {"type": "array", "description": "Classification results"},
                    "metadata": {"type": "object", "description": "Classification metadata"}
                }
            },
            handler=self.execute
        )
