"""Re-ranking module for improving retrieval results."""

from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from pocketrag.retrieval.hybrid_retriever import RetrievalResult


class CrossEncoderReranker:
    """Re-ranks retrieval results using a cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize re-ranker.
        
        Args:
            model_name: Name of cross-encoder model
        """
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is not installed. Install it with: pip install sentence-transformers")
        
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Re-rank retrieval results.
        
        Args:
            query: Query string
            results: List of retrieval results
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of results
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, result.chunk.content] for result in results]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Create new results with updated scores
        reranked_results = []
        for result, score in zip(results, scores):
            new_result = RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                rank=result.rank,
                retrieval_method=f"{result.retrieval_method}_reranked"
            )
            reranked_results.append(new_result)
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results[:top_k], 1):
            result.rank = rank
        
        return reranked_results[:top_k]
