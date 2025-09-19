"""
Tool Orchestrator - MCP-inspired tool management and workflow orchestration
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ToolType(Enum):
    """Types of tools available in the system"""
    SEARCH = "search"
    PREDICT = "predict"
    REASON = "reason"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    EXTRACT = "extract"
    CLASSIFY = "classify"


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ToolDefinition:
    """Definition of a tool following MCP standards"""
    name: str
    description: str
    tool_type: ToolType
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable


class BaseTool(ABC):
    """Base class for all tools following MCP pattern"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get tool definition for MCP discovery"""
        pass


class ToolOrchestrator:
    """Orchestrates tool execution based on query analysis and context"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        
    def register_tool(self, tool: BaseTool):
        """Register a tool for orchestration"""
        definition = tool.get_definition()
        self.tools[definition.name] = tool
        
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """MCP-compliant tool discovery"""
        return [
            {
                "name": tool.get_definition().name,
                "description": tool.get_definition().description,
                "inputSchema": tool.get_definition().input_schema,
                "outputSchema": tool.get_definition().output_schema
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=f"Tool '{tool_name}' not found"
            )
        
        start_time = asyncio.get_event_loop().time()
        try:
            result = await self.tools[tool_name].execute(**kwargs)
            result.execution_time = asyncio.get_event_loop().time() - start_time
            
            # Log execution
            self.execution_history.append({
                "tool": tool_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={},
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def orchestrate_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Orchestrate tool execution based on query analysis"""
        self.context = context or {}
        
        # Analyze query to determine tool sequence
        tool_sequence = await self._analyze_query_for_tools(query)
        
        results = {}
        for tool_name, params in tool_sequence:
            result = await self.execute_tool(tool_name, **params)
            results[tool_name] = result
            
            # Update context with results
            if result.success:
                self.context[f"{tool_name}_result"] = result.data
        
        return {
            "query": query,
            "tool_sequence": tool_sequence,
            "results": results,
            "context": self.context,
            "success": all(r.success for r in results.values())
        }
    
    async def _analyze_query_for_tools(self, query: str) -> List[tuple]:
        """Analyze query to determine optimal tool sequence"""
        query_lower = query.lower()
        tool_sequence = []
        
        # Always start with search for document retrieval
        tool_sequence.append(("search_tool", {"query": query, "top_k": 5}))
        
        # Determine additional tools based on query content
        if any(keyword in query_lower for keyword in ['predict', 'outcome', 'result', 'success']):
            tool_sequence.append(("predict_tool", {"query": query}))
        
        if any(keyword in query_lower for keyword in ['why', 'how', 'explain', 'reason']):
            tool_sequence.append(("reason_tool", {"query": query}))
        
        if any(keyword in query_lower for keyword in ['summarize', 'overview', 'summary']):
            tool_sequence.append(("summarize_tool", {"query": query}))
        
        if any(keyword in query_lower for keyword in ['analyze', 'compare', 'evaluate']):
            tool_sequence.append(("analyze_tool", {"query": query}))
        
        if any(keyword in query_lower for keyword in ['extract', 'find', 'identify']):
            tool_sequence.append(("extract_tool", {"query": query}))
        
        if any(keyword in query_lower for keyword in ['classify', 'type', 'category']):
            tool_sequence.append(("classify_tool", {"query": query}))
        
        return tool_sequence


class StateManager:
    """Manages conversation state and decision making"""
    
    def __init__(self):
        self.current_state = "idle"
        self.state_history = []
        self.user_goals = []
        self.decision_context = {}
        
    def set_state(self, new_state: str, context: Dict[str, Any] = None):
        """Transition to new state"""
        self.state_history.append({
            "from": self.current_state,
            "to": new_state,
            "context": context or {},
            "timestamp": asyncio.get_event_loop().time()
        })
        self.current_state = new_state
        
    def add_goal(self, goal: str, priority: int = 1):
        """Add user goal to track"""
        self.user_goals.append({
            "goal": goal,
            "priority": priority,
            "status": "active",
            "created_at": asyncio.get_event_loop().time()
        })
        
    def update_goal_status(self, goal: str, status: str):
        """Update goal status"""
        for g in self.user_goals:
            if g["goal"] == goal:
                g["status"] = status
                break
                
    def make_decision(self, options: List[Dict[str, Any]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make decision based on current state and context"""
        self.decision_context = context or {}
        
        # Simple decision logic based on state and context
        if self.current_state == "idle":
            return self._decide_initial_action(options)
        elif self.current_state == "searching":
            return self._decide_search_strategy(options)
        elif self.current_state == "analyzing":
            return self._decide_analysis_approach(options)
        else:
            return {"decision": "continue", "reasoning": "No specific decision logic for current state"}
    
    def _decide_initial_action(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decide initial action based on user goals"""
        if self.user_goals:
            highest_priority = max(self.user_goals, key=lambda g: g["priority"])
            return {
                "decision": "pursue_goal",
                "goal": highest_priority["goal"],
                "reasoning": f"Highest priority goal: {highest_priority['goal']}"
            }
        return {"decision": "wait_for_input", "reasoning": "No active goals"}
    
    def _decide_search_strategy(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decide search strategy"""
        return {
            "decision": "comprehensive_search",
            "reasoning": "Need comprehensive information for analysis"
        }
    
    def _decide_analysis_approach(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decide analysis approach"""
        return {
            "decision": "multi_tool_analysis",
            "reasoning": "Complex analysis requires multiple tools"
        }


class ContextualAwareness:
    """Provides contextual awareness for better responses"""
    
    def __init__(self):
        self.domain_context = {
            "clinical_trials": {
                "phases": ["I", "II", "III", "IV"],
                "endpoints": ["primary", "secondary", "safety"],
                "study_types": ["randomized", "controlled", "double_blind", "placebo_controlled"],
                "populations": ["adult", "pediatric", "geriatric", "pregnant_women"]
            }
        }
        self.temporal_context = {}
        self.user_intent_history = []
        
    def analyze_user_intent(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze user intent and context"""
        query_lower = query.lower()
        
        # Intent classification
        intent = self._classify_intent(query_lower)
        
        # Domain-specific context
        domain_context = self._extract_domain_context(query_lower)
        
        # Temporal context
        temporal_context = self._analyze_temporal_context(query, conversation_history)
        
        return {
            "intent": intent,
            "domain_context": domain_context,
            "temporal_context": temporal_context,
            "confidence": self._calculate_intent_confidence(query_lower, intent)
        }
    
    def _classify_intent(self, query_lower: str) -> str:
        """Classify user intent"""
        if any(keyword in query_lower for keyword in ['what', 'define', 'explain']):
            return "information_seeking"
        elif any(keyword in query_lower for keyword in ['how', 'process', 'method']):
            return "procedural_inquiry"
        elif any(keyword in query_lower for keyword in ['why', 'reason', 'cause']):
            return "causal_inquiry"
        elif any(keyword in query_lower for keyword in ['compare', 'difference', 'versus']):
            return "comparative_analysis"
        elif any(keyword in query_lower for keyword in ['predict', 'outcome', 'result']):
            return "predictive_inquiry"
        elif any(keyword in query_lower for keyword in ['find', 'search', 'locate']):
            return "search_request"
        else:
            return "general_inquiry"
    
    def _extract_domain_context(self, query_lower: str) -> Dict[str, Any]:
        """Extract clinical trial domain context"""
        context = {}
        
        # Check for clinical trial phases
        for phase in self.domain_context["clinical_trials"]["phases"]:
            if f"phase {phase.lower()}" in query_lower or f"phase {phase}" in query_lower:
                context["phase"] = phase
        
        # Check for endpoint types
        for endpoint in self.domain_context["clinical_trials"]["endpoints"]:
            if endpoint in query_lower:
                context["endpoint_type"] = endpoint
        
        # Check for study types
        for study_type in self.domain_context["clinical_trials"]["study_types"]:
            if study_type.replace("_", " ") in query_lower:
                context["study_type"] = study_type
        
        return context
    
    def _analyze_temporal_context(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze temporal context from conversation"""
        temporal_context = {
            "is_follow_up": False,
            "previous_topics": [],
            "conversation_depth": 0
        }
        
        if conversation_history:
            temporal_context["conversation_depth"] = len(conversation_history)
            temporal_context["previous_topics"] = [
                msg.get("topic", "unknown") for msg in conversation_history[-5:]
            ]
            
            # Check if this is a follow-up question
            if any(keyword in query.lower() for keyword in ['also', 'additionally', 'furthermore', 'moreover']):
                temporal_context["is_follow_up"] = True
        
        return temporal_context
    
    def _calculate_intent_confidence(self, query_lower: str, intent: str) -> float:
        """Calculate confidence in intent classification"""
        intent_keywords = {
            "information_seeking": ['what', 'define', 'explain', 'tell me about'],
            "procedural_inquiry": ['how', 'process', 'method', 'steps'],
            "causal_inquiry": ['why', 'reason', 'cause', 'because'],
            "comparative_analysis": ['compare', 'difference', 'versus', 'vs'],
            "predictive_inquiry": ['predict', 'outcome', 'result', 'will'],
            "search_request": ['find', 'search', 'locate', 'where']
        }
        
        keywords = intent_keywords.get(intent, [])
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        return min(matches / len(keywords) if keywords else 0, 1.0)
    
    def enhance_response_context(self, response: str, intent_analysis: Dict[str, Any]) -> str:
        """Enhance response with contextual information"""
        enhanced_response = response
        
        # Add domain-specific context
        if intent_analysis["domain_context"]:
            context_info = []
            if "phase" in intent_analysis["domain_context"]:
                context_info.append(f"Phase {intent_analysis['domain_context']['phase']} clinical trial")
            if "endpoint_type" in intent_analysis["domain_context"]:
                context_info.append(f"{intent_analysis['domain_context']['endpoint_type']} endpoint")
            
            if context_info:
                enhanced_response = f"Regarding {' '.join(context_info)}: {response}"
        
        # Add temporal context
        if intent_analysis["temporal_context"]["is_follow_up"]:
            enhanced_response = f"Building on our previous discussion: {response}"
        
        return enhanced_response
