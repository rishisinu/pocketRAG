"""Document ingestion module for PocketRAG."""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib
from datetime import datetime

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class Document:
    """Represents a document with metadata."""
    
    doc_id: str
    content: str
    metadata: Dict
    source: str
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID from content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"


class DocumentIngester:
    """Handles document ingestion from various sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize document ingester.
        
        Args:
            data_dir: Directory to store ingested documents
        """
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.documents: Dict[str, Document] = {}
    
    def ingest_pdf(self, file_path: Path) -> Document:
        """Ingest a PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document object
        """
        if PdfReader is None:
            raise ImportError("pypdf is not installed. Install it with: pip install pypdf")
        
        reader = PdfReader(str(file_path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        content = "\n\n".join(text_parts)
        
        metadata = {
            "filename": file_path.name,
            "num_pages": len(reader.pages),
            "file_type": "pdf",
            "ingested_at": datetime.now().isoformat(),
        }
        
        # Extract PDF metadata if available
        if reader.metadata:
            for key in ["/Title", "/Author", "/Subject", "/Creator"]:
                if key in reader.metadata:
                    metadata[key.lstrip("/")] = reader.metadata[key]
        
        doc = Document(
            doc_id="",
            content=content,
            metadata=metadata,
            source=str(file_path)
        )
        
        self.documents[doc.doc_id] = doc
        return doc
    
    def ingest_text(self, file_path: Path) -> Document:
        """Ingest a text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lstrip('.'),
            "ingested_at": datetime.now().isoformat(),
        }
        
        doc = Document(
            doc_id="",
            content=content,
            metadata=metadata,
            source=str(file_path)
        )
        
        self.documents[doc.doc_id] = doc
        return doc
    
    def ingest_file(self, file_path: Path) -> Document:
        """Ingest a file based on its extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Document object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.ingest_pdf(file_path)
        elif suffix in ['.txt', '.md']:
            return self.ingest_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def ingest_directory(self, dir_path: Path, recursive: bool = True) -> List[Document]:
        """Ingest all supported files from a directory.
        
        Args:
            dir_path: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of Document objects
        """
        dir_path = Path(dir_path)
        documents = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                try:
                    doc = self.ingest_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document object or None
        """
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Document]:
        """List all ingested documents.
        
        Returns:
            List of Document objects
        """
        return list(self.documents.values())
