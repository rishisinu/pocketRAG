"""Tests for document ingestion."""

import pytest
from pathlib import Path
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pocketrag.ingestion.document_ingester import DocumentIngester, Document


def test_document_creation():
    """Test document creation."""
    doc = Document(
        doc_id="test_doc",
        content="Test content",
        metadata={"test": "value"},
        source="test.txt"
    )
    
    assert doc.doc_id == "test_doc"
    assert doc.content == "Test content"
    assert doc.metadata["test"] == "value"


def test_ingest_text_file():
    """Test ingesting a text file."""
    ingester = DocumentIngester()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for ingestion")
        temp_path = Path(f.name)
    
    try:
        doc = ingester.ingest_text(temp_path)
        
        assert "Test content for ingestion" in doc.content
        assert doc.metadata["filename"] == temp_path.name
        assert doc.metadata["file_type"] == "txt"
    finally:
        temp_path.unlink()


def test_ingest_directory():
    """Test ingesting multiple files from directory."""
    ingester = DocumentIngester()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        (tmpdir / "test1.txt").write_text("Content 1")
        (tmpdir / "test2.txt").write_text("Content 2")
        (tmpdir / "test3.md").write_text("Content 3")
        
        docs = ingester.ingest_directory(tmpdir)
        
        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)
