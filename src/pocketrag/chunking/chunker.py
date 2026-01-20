"""Document chunking module with token awareness."""

from typing import List, Dict
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pocketrag.utils.tokenizer import count_tokens, split_by_tokens
from pocketrag.ingestion.document_ingester import Document


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict
    start_pos: int
    end_pos: int
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_chunk_{self.start_pos}"


class TokenAwareChunker:
    """Chunks documents with token-aware splitting."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens
            min_chunk_size: Minimum tokens per chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        content = document.content
        
        # Split by tokens
        text_chunks = split_by_tokens(
            content,
            max_tokens=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        chunks = []
        position = 0
        
        for i, text in enumerate(text_chunks):
            # Skip chunks that are too small
            if count_tokens(text) < self.min_chunk_size and i < len(text_chunks) - 1:
                continue
            
            chunk_metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "doc_source": document.source,
            }
            
            chunk = Chunk(
                chunk_id="",
                doc_id=document.doc_id,
                content=text,
                metadata=chunk_metadata,
                start_pos=position,
                end_pos=position + len(text)
            )
            
            chunks.append(chunk)
            position += len(text)
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
