"""Evaluation metrics for retrieval and citation quality."""

from typing import List, Dict, Set
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pocketrag.retrieval.hybrid_retriever import RetrievalResult


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate precision at k.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision at k
        """
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant)
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate recall at k.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall at k
        """
        if not relevant:
            return 1.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant)
        
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_lists: List[List[str]],
        relevant_lists: List[Set[str]]
    ) -> float:
        """Calculate mean reciprocal rank.
        
        Args:
            retrieved_lists: List of retrieved document ID lists
            relevant_lists: List of relevant document ID sets
            
        Returns:
            Mean reciprocal rank
        """
        if not retrieved_lists:
            return 0.0
        
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    @staticmethod
    def evaluate_retrieval(
        results: List[RetrievalResult],
        relevant_doc_ids: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """Evaluate retrieval results.
        
        Args:
            results: Retrieved results
            relevant_doc_ids: Set of relevant document IDs
            k_values: K values to evaluate
            
        Returns:
            Dictionary of metrics
        """
        retrieved_ids = [r.chunk.doc_id for r in results]
        
        metrics = {}
        
        for k in k_values:
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(
                retrieved_ids, relevant_doc_ids, k
            )
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(
                retrieved_ids, relevant_doc_ids, k
            )
        
        return metrics


class CitationMetrics:
    """Metrics for evaluating citation quality."""
    
    @staticmethod
    def citation_precision(
        answer_citations: Set[int],
        used_sources: Set[int]
    ) -> float:
        """Calculate citation precision.
        
        Args:
            answer_citations: Citations in the answer
            used_sources: Sources actually used
            
        Returns:
            Citation precision
        """
        if not answer_citations:
            return 0.0
        
        correct_citations = answer_citations.intersection(used_sources)
        return len(correct_citations) / len(answer_citations)
    
    @staticmethod
    def citation_recall(
        answer_citations: Set[int],
        used_sources: Set[int]
    ) -> float:
        """Calculate citation recall.
        
        Args:
            answer_citations: Citations in the answer
            used_sources: Sources actually used
            
        Returns:
            Citation recall
        """
        if not used_sources:
            return 1.0
        
        correct_citations = answer_citations.intersection(used_sources)
        return len(correct_citations) / len(used_sources)
    
    @staticmethod
    def citation_f1(
        answer_citations: Set[int],
        used_sources: Set[int]
    ) -> float:
        """Calculate citation F1 score.
        
        Args:
            answer_citations: Citations in the answer
            used_sources: Sources actually used
            
        Returns:
            Citation F1 score
        """
        precision = CitationMetrics.citation_precision(answer_citations, used_sources)
        recall = CitationMetrics.citation_recall(answer_citations, used_sources)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_citations(
        answer: str,
        num_sources: int
    ) -> Dict:
        """Evaluate citation quality in answer.
        
        Args:
            answer: Generated answer
            num_sources: Number of source documents
            
        Returns:
            Dictionary of citation metrics
        """
        import re
        
        # Extract citations
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(int(c) for c in citations)
        
        # All sources could potentially be used
        all_sources = set(range(1, num_sources + 1))
        
        metrics = {
            "total_citations": len(citations),
            "unique_citations": len(unique_citations),
            "citation_density": len(citations) / len(answer.split()) if answer else 0,
            "has_citations": len(unique_citations) > 0
        }
        
        return metrics
