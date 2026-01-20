"""Tests for chunking module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pocketrag.ingestion.document_ingester import Document
from pocketrag.chunking.chunker import TokenAwareChunker, Chunk


def test_chunk_creation():
    """Test chunk creation."""
    chunk = Chunk(
        chunk_id="test_chunk",
        doc_id="test_doc",
        content="Test chunk content",
        metadata={"test": "value"},
        start_pos=0,
        end_pos=10
    )
    
    assert chunk.chunk_id == "test_chunk"
    assert chunk.doc_id == "test_doc"


def test_chunk_document():
    """Test chunking a document."""
    doc = Document(
        doc_id="test_doc",
        content="This is a test document. " * 100,  # Long content
        metadata={"filename": "test.txt"},
        source="test.txt"
    )
    
    chunker = TokenAwareChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.doc_id == doc.doc_id for chunk in chunks)


def test_chunk_multiple_documents():
    """Test chunking multiple documents."""
    docs = [
        Document(
            doc_id=f"doc_{i}",
            content=f"Document {i} content. " * 50,
            metadata={"filename": f"doc{i}.txt"},
            source=f"doc{i}.txt"
        )
        for i in range(3)
    ]
    
    chunker = TokenAwareChunker()
    chunks = chunker.chunk_documents(docs)
    
    assert len(chunks) > 0
    assert len(set(chunk.doc_id for chunk in chunks)) == 3
