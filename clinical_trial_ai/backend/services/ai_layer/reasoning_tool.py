import asyncio
from typing import List, Dict, Any

class ReasoningTool:
    """Analyze predictions and provide reasoning insights"""

    async def analyze(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions and generate reasoning"""
        reasoning_summary = {
            "total_chunks": len(predictions),
            "predicted_labels": list({p["prediction"] for p in predictions}),
            "confidence_avg": sum(p["confidence"] for p in predictions) / max(len(predictions), 1)
        }
        await asyncio.sleep(0)
        return reasoning_summary
