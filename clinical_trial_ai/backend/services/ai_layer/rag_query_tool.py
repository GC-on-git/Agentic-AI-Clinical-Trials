import asyncio
from typing import List, Dict, Any, Optional
from clinical_trial_ai.backend.services.embeddings.sbert_embed import SBERTEmbedder
from clinical_trial_ai.backend.services.vector_db.chroma_store import ChromaVectorStore
from clinical_trial_ai.backend.services.ai_layer.search_tool import SearchTool
from clinical_trial_ai.backend.services.ai_layer.summary_tool import SummaryTool
from clinical_trial_ai.backend.services.ai_layer.reasoning_tool import ReasoningTool
from clinical_trial_ai.backend.services.ai_layer.llm_service import LLMService


class RAGQueryHandler:
    """Handles RAG (Retrieval-Augmented Generation) queries for clinical trial documents"""

    def __init__(self, 
                 embedder: SBERTEmbedder,
                 vector_store: ChromaVectorStore,
                 search_tool: SearchTool,
                 summary_tool: SummaryTool,
                 reasoning_tool: ReasoningTool,
                 llm_service: Optional[LLMService] = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.search_tool = search_tool
        self.summary_tool = summary_tool
        self.reasoning_tool = reasoning_tool
        self.llm_service = llm_service or LLMService()

    async def process_query(self, 
                          query: str, 
                          top_k: int = 5,
                          include_context: bool = True,
                          generate_summary: bool = True) -> Dict[str, Any]:
        """
        Process a user query using RAG pipeline
        
        Args:
            query: User's question/query
            top_k: Number of relevant chunks to retrieve
            include_context: Whether to include retrieved context in response
            generate_summary: Whether to generate a summary of retrieved content
            
        Returns:
            Dictionary containing query results, context, and generated response
        """
        
        # Step 1: Generate embedding for the query
        query_embedding = await self._embed_query(query)
        
        # Step 2: Retrieve relevant chunks using vector similarity
        relevant_chunks = await self._retrieve_relevant_chunks(query_embedding, top_k)
        
        # Step 3: Rank and filter chunks by relevance
        ranked_chunks = await self._rank_chunks_by_relevance(query, relevant_chunks)
        
        # Step 4: Generate context from retrieved chunks
        context = await self._generate_context(ranked_chunks) if include_context else None
        
        # Step 5: Generate response (placeholder for LLM integration)
        response = await self._generate_response(query, ranked_chunks, context)
        
        # Step 6: Generate summary if requested
        summary = None
        if generate_summary and ranked_chunks:
            summary = await self.summary_tool.generate_summary(ranked_chunks)
        
        # Step 7: Extract insights and reasoning
        insights = await self._extract_insights(query, ranked_chunks)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "relevant_chunks": ranked_chunks,
            "summary": summary,
            "insights": insights,
            "metadata": {
                "chunks_retrieved": len(relevant_chunks),
                "chunks_used": len(ranked_chunks),
                "query_embedding_dim": len(query_embedding) if query_embedding else 0
            }
        }

    async def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for the user query"""
        try:
            # Use the same embedder as documents
            query_chunk = [{"content": query, "metadata": {"id": "query"}}]
            embeddings = await self.embedder.generate_embeddings(query_chunk)
            return embeddings[0]["embedding"] if embeddings else []
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

    async def _retrieve_relevant_chunks(self, 
                                      query_embedding: List[float], 
                                      top_k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using vector similarity search"""
        try:
            # Use ChromaDB's similarity search
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=0.3  # Adjust threshold as needed
            )
            
            # Convert results to chunk format
            chunks = []
            for metadata, similarity_score in results:
                chunk = {
                    "id": metadata.get("id", "unknown"),
                    "content": metadata.get("content", ""),
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "document_id": metadata.get("document_id", "unknown")
                }
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    async def _rank_chunks_by_relevance(self, 
                                      query: str, 
                                      chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank chunks by relevance to the query"""
        if not chunks:
            return []
        
        # Simple ranking based on similarity score and keyword matching
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            content_keywords = set(content_lower.split())
            
            # Calculate keyword overlap score
            keyword_overlap = len(query_keywords.intersection(content_keywords))
            keyword_score = keyword_overlap / max(len(query_keywords), 1)
            
            # Combine similarity score with keyword score
            similarity_score = chunk.get("similarity_score", 0)
            relevance_score = (similarity_score * 0.7) + (keyword_score * 0.3)
            
            chunk["relevance_score"] = relevance_score
        
        # Sort by relevance score
        ranked_chunks = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_chunks

    async def _generate_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate context string from relevant chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks[:3]):  # Use top 3 chunks for context
            content = chunk["content"]
            similarity = chunk.get("similarity_score", 0)
            context_parts.append(f"[Context {i+1} (similarity: {similarity:.3f})]: {content}")
        
        return "\n\n".join(context_parts)

    async def _generate_response(self, 
                               query: str, 
                               chunks: List[Dict[str, Any]], 
                               context: Optional[str]) -> str:
        """Generate response to the query using LLM or fallback to rule-based"""
        
        if not chunks:
            return "I couldn't find relevant information in the clinical trial documents to answer your question."
        
        # Use LLM if available
        if self.llm_service.is_configured():
            try:
                # Prepare context for LLM
                llm_context = context or self._prepare_context_for_llm(chunks)
                response = await self.llm_service.generate_with_context(
                    query=query,
                    context=llm_context,
                    max_tokens=1000,
                    temperature=0.7
                )
                return response
            except Exception as e:
                print(f"Error generating LLM response: {e}")
                # Fall back to rule-based response
        
        # Fallback: Simple rule-based response generation
        response_parts = []
        
        # Analyze query type
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['what', 'describe', 'explain']):
            response_parts.append("Based on the clinical trial documents, here's what I found:")
        elif any(keyword in query_lower for keyword in ['how', 'method', 'procedure']):
            response_parts.append("According to the clinical trial protocols:")
        elif any(keyword in query_lower for keyword in ['results', 'outcome', 'efficacy']):
            response_parts.append("The clinical trial results show:")
        else:
            response_parts.append("Here's the relevant information from the clinical trials:")
        
        # Add key information from chunks
        key_info = []
        for chunk in chunks[:2]:  # Use top 2 chunks
            content = chunk["content"]
            # Extract key sentences (simple approach)
            sentences = content.split('. ')
            if sentences:
                key_sentence = sentences[0] if len(sentences[0]) > 50 else content[:200]
                key_info.append(key_sentence)
        
        if key_info:
            response_parts.append(" ".join(key_info))
        
        # Add context if available
        if context:
            response_parts.append(f"\nAdditional context: {context[:300]}...")
        
        return " ".join(response_parts)
    
    def _prepare_context_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from chunks for LLM processing"""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
            content = chunk["content"]
            similarity = chunk.get("similarity_score", 0)
            context_parts.append(f"[Document {i+1} (relevance: {similarity:.3f})]: {content}")
        
        return "\n\n".join(context_parts)

    async def _extract_insights(self, 
                              query: str, 
                              chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract insights from the retrieved chunks"""
        if not chunks:
            return {"insights": [], "confidence": 0.0}
        
        insights = []
        total_confidence = 0
        
        for chunk in chunks:
            # Extract clinical information from chunk
            content = chunk["content"]
            relevance_score = chunk.get("relevance_score", 0)
            
            # Simple insight extraction
            insight = {
                "content": content[:200] + "..." if len(content) > 200 else content,
                "relevance_score": relevance_score,
                "document_id": chunk.get("document_id", "unknown"),
                "chunk_id": chunk.get("id", "unknown")
            }
            insights.append(insight)
            total_confidence += relevance_score
        
        avg_confidence = total_confidence / len(chunks) if chunks else 0
        
        return {
            "insights": insights,
            "confidence": avg_confidence,
            "total_chunks_analyzed": len(chunks)
        }

    async def search_by_keywords(self, 
                               keywords: List[str], 
                               top_k: int = 10) -> Dict[str, Any]:
        """Search for documents containing specific keywords"""
        try:
            # Create a query from keywords
            query = " ".join(keywords)
            
            # Use the main query processing pipeline
            results = await self.process_query(query, top_k=top_k)
            
            # Filter results by keyword presence
            keyword_results = []
            for chunk in results["relevant_chunks"]:
                content_lower = chunk["content"].lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
                if keyword_matches > 0:
                    chunk["keyword_matches"] = keyword_matches
                    keyword_results.append(chunk)
            
            return {
                "keywords": keywords,
                "matching_chunks": keyword_results,
                "total_matches": len(keyword_results),
                "query_results": results
            }
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return {"keywords": keywords, "matching_chunks": [], "error": str(e)}

    async def get_document_overview(self, document_id: str) -> Dict[str, Any]:
        """Get overview of a specific document"""
        try:
            # Get all chunks for the document
            chunks = await self.search_tool.get_chunks_by_document_id(document_id)
            
            if not chunks:
                return {"document_id": document_id, "error": "Document not found"}
            
            # Generate summary
            summary = await self.summary_tool.generate_summary(chunks)
            
            # Extract key information
            total_content = " ".join([chunk["content"] for chunk in chunks])
            
            return {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "summary": summary,
                "total_content_length": len(total_content),
                "chunks": chunks[:5]  # Return first 5 chunks as preview
            }
        except Exception as e:
            return {"document_id": document_id, "error": str(e)}
