"""Tests for utility functions."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pocketrag.utils.tokenizer import count_tokens, split_by_tokens, split_by_chars


def test_count_tokens():
    """Test token counting."""
    text = "This is a test."
    count = count_tokens(text)
    
    assert isinstance(count, int)
    assert count > 0


def test_split_by_tokens():
    """Test splitting by tokens."""
    text = "This is a test. " * 50
    chunks = split_by_tokens(text, max_tokens=50, overlap=10)
    
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_by_chars():
    """Test splitting by characters."""
    text = "a" * 1000
    chunks = split_by_chars(text, max_chars=100, overlap=10)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
