"""Local LLM wrapper for answer generation."""

from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

from pocketrag.retrieval.hybrid_retriever import RetrievalResult


class LocalLLM:
    """Wrapper for local LLM inference."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 512,
        temperature: float = 0.1,
        device: str = "cpu"
    ):
        """Initialize local LLM.
        
        Args:
            model_name: Name of the model
            max_length: Maximum generation length
            temperature: Sampling temperature
            device: Device to use (cpu/cuda)
        """
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is not installed. Install it with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.device = device
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded on {device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text


class AnswerGenerator:
    """Generates citation-backed answers from retrieved passages."""
    
    def __init__(
        self,
        llm: Optional[LocalLLM] = None,
        citation_format: str = "[{doc_id}]"
    ):
        """Initialize answer generator.
        
        Args:
            llm: Local LLM instance
            citation_format: Format for citations
        """
        self.llm = llm
        self.citation_format = citation_format
    
    def generate_answer(
        self,
        query: str,
        results: List[RetrievalResult],
        max_tokens: int = 512
    ) -> Dict:
        """Generate answer from retrieved results.
        
        Args:
            query: User query
            results: Retrieved and re-ranked results
            max_tokens: Maximum tokens for answer
            
        Returns:
            Dictionary with answer and citations
        """
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "citations": [],
                "sources": []
            }
        
        # Build context from results
        context_parts = []
        citations = []
        sources = []
        
        for i, result in enumerate(results, 1):
            citation = self.citation_format.format(doc_id=i)
            context_parts.append(f"{citation} {result.chunk.content}")
            
            citations.append({
                "citation_id": i,
                "doc_id": result.chunk.doc_id,
                "chunk_id": result.chunk.chunk_id,
                "score": result.score,
                "source": result.chunk.metadata.get("doc_source", "unknown")
            })
            
            sources.append(result.chunk.metadata.get("filename", "unknown"))
        
        context = "\n\n".join(context_parts)
        
        # Build prompt with citation enforcement
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        if self.llm:
            answer = self.llm.generate(prompt, max_new_tokens=max_tokens)
        else:
            # Fallback: return first passage with citation
            answer = f"Based on the retrieved information {self.citation_format.format(doc_id=1)}, {results[0].chunk.content[:200]}..."
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": list(set(sources))
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for answer generation.
        
        Args:
            query: User query
            context: Retrieved context with citations
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Only use information from the context below
2. Always cite your sources using the citation markers [1], [2], etc.
3. If you cannot answer the question from the context, say so
4. Do not add information that is not in the context
5. Be concise and accurate

Context:
{context}

Question: {query}

Answer (with citations):"""
        
        return prompt
