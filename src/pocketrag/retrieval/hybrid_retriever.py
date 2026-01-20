"""Hybrid retrieval system combining BM25 and FAISS."""

from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from pocketrag.chunking.chunker import Chunk


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    
    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str


class BM25Retriever:
    """BM25-based retriever for keyword matching."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Install it with: pip install rank-bm25")
        
        self.bm25 = None
        self.chunks: List[Chunk] = []
    
    def index(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval.
        
        Args:
            chunks: List of chunks to index
        """
        self.chunks = chunks
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))
        
        return results


class FAISSRetriever:
    """FAISS-based retriever for semantic search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize FAISS retriever.
        
        Args:
            model_name: Name of sentence transformer model
        """
        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Install it with: pip install faiss-cpu")
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Install it with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Chunk] = []
    
    def index(self, chunks: List[Chunk]):
        """Index chunks for FAISS retrieval.
        
        Args:
            chunks: List of chunks to index
        """
        self.chunks = chunks
        
        # Encode all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks using FAISS.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results


class HybridRetriever:
    """Combines BM25 and FAISS for hybrid retrieval."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5
    ):
        """Initialize hybrid retriever.
        
        Args:
            embedding_model: Name of sentence transformer model
            bm25_weight: Weight for BM25 scores
            faiss_weight: Weight for FAISS scores
        """
        self.bm25_retriever = BM25Retriever()
        self.faiss_retriever = FAISSRetriever(embedding_model)
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
    
    def index(self, chunks: List[Chunk]):
        """Index chunks for both retrievers.
        
        Args:
            chunks: List of chunks to index
        """
        print("Indexing with BM25...")
        self.bm25_retriever.index(chunks)
        
        print("Indexing with FAISS...")
        self.faiss_retriever.index(chunks)
        
        print(f"Indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        faiss_results = self.faiss_retriever.retrieve(query, top_k=top_k * 2)
        
        # Normalize scores
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        faiss_scores = self._normalize_scores([score for _, score in faiss_results])
        
        # Combine scores
        chunk_scores: Dict[str, Tuple[Chunk, float]] = {}
        
        for (chunk, _), norm_score in zip(bm25_results, bm25_scores):
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = (chunk, 0.0)
            chunk_scores[chunk_id] = (
                chunk_scores[chunk_id][0],
                chunk_scores[chunk_id][1] + norm_score * self.bm25_weight
            )
        
        for (chunk, _), norm_score in zip(faiss_results, faiss_scores):
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = (chunk, 0.0)
            chunk_scores[chunk_id] = (
                chunk_scores[chunk_id][0],
                chunk_scores[chunk_id][1] + norm_score * self.faiss_weight
            )
        
        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Create result objects
        results = []
        for rank, (chunk, score) in enumerate(sorted_results, 1):
            result = RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="hybrid"
            )
            results.append(result)
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
