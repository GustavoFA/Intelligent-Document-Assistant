
from typing import Optional, List, Dict
from .config import MIN_CONTEXT_QUALITY

"""
Retrieval and context validation utilities for RAG pipeline.
"""

def check_context_quality(context: Optional[List[Dict]]) -> bool:
    """
    Check if context has sufficient quality based on search distances.
    
    Args:
        context: Retrieved document chunks with distance scores
        
    Returns:
        True if context is sufficient, False otherwise
    """
    if not context:
        return False
    
    # Check if any chunk has distance below threshold (higher quality)
    if context and len(context) > 0:
        best_distance = min(chunk.get('distance', float('inf')) for chunk in context)
        return best_distance < MIN_CONTEXT_QUALITY
    
    return False
