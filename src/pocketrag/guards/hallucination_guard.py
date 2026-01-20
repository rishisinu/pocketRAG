"""Hallucination guard using entity verification."""

from typing import List, Set, Dict
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import spacy
    from spacy.cli import download
except ImportError:
    spacy = None

from pocketrag.retrieval.hybrid_retriever import RetrievalResult


class EntityBasedHallucinationGuard:
    """Verifies answers against source entities to detect hallucinations."""
    
    def __init__(self, threshold: float = 0.7, max_mismatch: int = 2):
        """Initialize hallucination guard.
        
        Args:
            threshold: Minimum entity overlap threshold
            max_mismatch: Maximum allowed entity mismatches
        """
        self.threshold = threshold
        self.max_mismatch = max_mismatch
        self.nlp = None
        
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading spacy model...")
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of entity strings
        """
        if self.nlp is None:
            # Fallback: extract capitalized words
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return set(words)
        
        doc = self.nlp(text)
        entities = {ent.text.lower() for ent in doc.ents}
        
        return entities
    
    def verify_answer(
        self,
        answer: str,
        sources: List[RetrievalResult]
    ) -> Dict:
        """Verify answer against source documents.
        
        Args:
            answer: Generated answer
            sources: Source retrieval results
            
        Returns:
            Verification result dictionary
        """
        # Extract entities from answer
        answer_entities = self.extract_entities(answer)
        
        if not answer_entities:
            return {
                "verified": True,
                "confidence": 1.0,
                "reason": "No entities to verify",
                "mismatched_entities": []
            }
        
        # Extract entities from all source chunks
        source_entities = set()
        for result in sources:
            chunk_entities = self.extract_entities(result.chunk.content)
            source_entities.update(chunk_entities)
        
        # Calculate overlap
        matched_entities = answer_entities.intersection(source_entities)
        mismatched_entities = answer_entities - source_entities
        
        overlap_ratio = len(matched_entities) / len(answer_entities) if answer_entities else 1.0
        
        # Determine if verified
        verified = (
            overlap_ratio >= self.threshold and
            len(mismatched_entities) <= self.max_mismatch
        )
        
        return {
            "verified": verified,
            "confidence": overlap_ratio,
            "reason": "Passed entity verification" if verified else "Entity mismatch detected",
            "mismatched_entities": list(mismatched_entities),
            "matched_entities": list(matched_entities),
            "total_answer_entities": len(answer_entities),
            "total_source_entities": len(source_entities)
        }
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation markers from text.
        
        Args:
            text: Text with citations
            
        Returns:
            List of citation markers
        """
        # Match citation patterns like [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', text)
        return citations
    
    def validate_citations(
        self,
        answer: str,
        num_sources: int,
        min_citations: int = 1
    ) -> Dict:
        """Validate that answer has proper citations.
        
        Args:
            answer: Generated answer
            num_sources: Number of available sources
            min_citations: Minimum required citations
            
        Returns:
            Validation result dictionary
        """
        citations = self.extract_citations(answer)
        unique_citations = set(citations)
        
        # Check if citations exist
        if len(unique_citations) < min_citations:
            return {
                "valid": False,
                "reason": f"Insufficient citations. Found {len(unique_citations)}, required {min_citations}",
                "citations": list(unique_citations)
            }
        
        # Check if citations are valid (within range)
        invalid_citations = [c for c in unique_citations if int(c) > num_sources or int(c) < 1]
        
        if invalid_citations:
            return {
                "valid": False,
                "reason": f"Invalid citation numbers: {invalid_citations}",
                "citations": list(unique_citations)
            }
        
        return {
            "valid": True,
            "reason": "Citations validated",
            "citations": list(unique_citations),
            "citation_count": len(citations)
        }
