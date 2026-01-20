"""Tests for hallucination guard."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pocketrag.guards.hallucination_guard import EntityBasedHallucinationGuard


def test_extract_citations():
    """Test citation extraction."""
    guard = EntityBasedHallucinationGuard()
    
    text = "According to [1], machine learning [2] is a subset of AI [3]."
    citations = guard.extract_citations(text)
    
    assert len(citations) == 3
    assert "1" in citations
    assert "2" in citations
    assert "3" in citations


def test_validate_citations_valid():
    """Test citation validation with valid citations."""
    guard = EntityBasedHallucinationGuard()
    
    answer = "Machine learning [1] is a subset of AI [2]."
    result = guard.validate_citations(answer, num_sources=3, min_citations=1)
    
    assert result["valid"] is True
    assert len(result["citations"]) == 2


def test_validate_citations_invalid():
    """Test citation validation with invalid citations."""
    guard = EntityBasedHallucinationGuard()
    
    answer = "Machine learning [5] is a subset of AI [10]."
    result = guard.validate_citations(answer, num_sources=3, min_citations=1)
    
    assert result["valid"] is False


def test_extract_entities():
    """Test entity extraction."""
    guard = EntityBasedHallucinationGuard()
    
    text = "TensorFlow and PyTorch are popular Machine Learning frameworks."
    entities = guard.extract_entities(text)
    
    assert isinstance(entities, set)
    assert len(entities) > 0
