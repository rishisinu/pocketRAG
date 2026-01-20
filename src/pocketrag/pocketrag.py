"""Main PocketRAG system integrating all components."""

from typing import List, Dict, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pocketrag.ingestion.document_ingester import DocumentIngester, Document
from pocketrag.chunking.chunker import TokenAwareChunker, Chunk
from pocketrag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from pocketrag.ranking.reranker import CrossEncoderReranker
from pocketrag.llm.answer_generator import AnswerGenerator, LocalLLM
from pocketrag.guards.hallucination_guard import EntityBasedHallucinationGuard
from pocketrag.evaluation.metrics import RetrievalMetrics, CitationMetrics


class PocketRAG:
    """Main PocketRAG system for offline document QA."""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        use_llm: bool = False
    ):
        """Initialize PocketRAG system.
        
        Args:
            data_dir: Directory for data storage
            embedding_model: Name of embedding model
            rerank_model: Name of re-ranking model
            llm_model: Name of LLM model (optional)
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
            top_k_retrieval: Number of results to retrieve
            top_k_rerank: Number of results after re-ranking
            use_llm: Whether to use LLM for generation
        """
        self.data_dir = data_dir or Path("data")
        
        # Initialize components
        print("Initializing PocketRAG components...")
        
        self.ingester = DocumentIngester(self.data_dir)
        self.chunker = TokenAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.retriever = HybridRetriever(embedding_model=embedding_model)
        self.reranker = CrossEncoderReranker(model_name=rerank_model)
        self.guard = EntityBasedHallucinationGuard()
        
        # Initialize LLM if requested
        self.llm = None
        if use_llm and llm_model:
            self.llm = LocalLLM(model_name=llm_model)
        
        self.answer_generator = AnswerGenerator(llm=self.llm)
        
        # Configuration
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        
        # Storage
        self.chunks: List[Chunk] = []
        self.indexed = False
        
        print("PocketRAG initialized successfully!")
    
    def ingest_documents(
        self,
        file_paths: Optional[List[Path]] = None,
        directory: Optional[Path] = None
    ) -> List[Document]:
        """Ingest documents from files or directory.
        
        Args:
            file_paths: List of file paths to ingest
            directory: Directory to ingest from
            
        Returns:
            List of ingested documents
        """
        documents = []
        
        if file_paths:
            for file_path in file_paths:
                try:
                    doc = self.ingester.ingest_file(Path(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        if directory:
            docs = self.ingester.ingest_directory(Path(directory))
            documents.extend(docs)
        
        if documents:
            print(f"Ingested {len(documents)} documents")
            self._process_and_index(documents)
        
        return documents
    
    def _process_and_index(self, documents: List[Document]):
        """Process documents into chunks and index them.
        
        Args:
            documents: List of documents to process
        """
        print("Chunking documents...")
        self.chunks = self.chunker.chunk_documents(documents)
        print(f"Created {len(self.chunks)} chunks")
        
        print("Indexing chunks...")
        self.retriever.index(self.chunks)
        self.indexed = True
        print("Indexing complete!")
    
    def query(
        self,
        query: str,
        return_diagnostics: bool = False
    ) -> Dict:
        """Query the system for an answer.
        
        Args:
            query: User query
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.indexed:
            return {
                "answer": "No documents have been ingested yet. Please ingest documents first.",
                "citations": [],
                "sources": []
            }
        
        # Retrieve relevant chunks
        print(f"Retrieving top {self.top_k_retrieval} results...")
        retrieval_results = self.retriever.retrieve(query, top_k=self.top_k_retrieval)
        
        # Re-rank results
        print(f"Re-ranking to top {self.top_k_rerank}...")
        reranked_results = self.reranker.rerank(
            query,
            retrieval_results,
            top_k=self.top_k_rerank
        )
        
        # Generate answer
        print("Generating answer...")
        answer_data = self.answer_generator.generate_answer(
            query,
            reranked_results
        )
        
        # Validate citations
        citation_validation = self.guard.validate_citations(
            answer_data["answer"],
            num_sources=len(reranked_results),
            min_citations=1
        )
        
        # Verify against hallucinations
        entity_verification = self.guard.verify_answer(
            answer_data["answer"],
            reranked_results
        )
        
        response = {
            "query": query,
            "answer": answer_data["answer"],
            "citations": answer_data["citations"],
            "sources": answer_data["sources"],
            "citation_valid": citation_validation["valid"],
            "entity_verified": entity_verification["verified"]
        }
        
        if return_diagnostics:
            response["diagnostics"] = {
                "num_retrieved": len(retrieval_results),
                "num_reranked": len(reranked_results),
                "citation_validation": citation_validation,
                "entity_verification": entity_verification,
                "retrieval_scores": [r.score for r in reranked_results]
            }
        
        return response
    
    def evaluate_retrieval(
        self,
        query: str,
        relevant_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """Evaluate retrieval performance.
        
        Args:
            query: Query string
            relevant_doc_ids: List of relevant document IDs
            k_values: K values to evaluate
            
        Returns:
            Dictionary of metrics
        """
        results = self.retriever.retrieve(query, top_k=max(k_values))
        
        metrics = RetrievalMetrics.evaluate_retrieval(
            results,
            set(relevant_doc_ids),
            k_values
        )
        
        return metrics
    
    def get_statistics(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "num_documents": len(self.ingester.documents),
            "num_chunks": len(self.chunks),
            "indexed": self.indexed,
            "avg_chunk_size": sum(len(c.content) for c in self.chunks) / len(self.chunks) if self.chunks else 0
        }
