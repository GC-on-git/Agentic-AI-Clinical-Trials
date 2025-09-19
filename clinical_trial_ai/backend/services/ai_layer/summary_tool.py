import asyncio
from typing import List, Dict, Any

class SummaryTool:
    """Summarizes document content"""

    async def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Simple async summary: concatenate top content"""
        top_texts = [chunk["content"] for chunk in chunks[:5]]  # top 5 chunks
        summary = "\n".join(top_texts)
        await asyncio.sleep(0)
        return summary
