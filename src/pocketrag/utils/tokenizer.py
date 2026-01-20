"""Utility functions for PocketRAG."""

from typing import List

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens(text: str, model: str = "gpt2") -> int:
    """Count tokens in text using tiktoken or approximate count.
    
    Args:
        text: Input text
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except (KeyError, ValueError, Exception):
            pass
    
    # Fallback: approximate token count (average ~4 chars per token)
    return len(text) // 4 + len(text.split())


def split_by_tokens(text: str, max_tokens: int, overlap: int = 0) -> List[str]:
    """Split text into chunks based on token count.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens
        
    Returns:
        List of text chunks
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            chunks = []
            
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if end >= len(tokens):
                    break
                
                start = end - overlap
            
            return chunks
        except (ImportError, ValueError, Exception):
            pass
    
    # Fallback to simple character-based splitting
    chars_per_token = 4  # Approximate
    return split_by_chars(text, max_tokens * chars_per_token, overlap * chars_per_token)


def split_by_chars(text: str, max_chars: int, overlap: int = 0) -> List[str]:
    """Fallback method to split text by characters.
    
    Args:
        text: Input text
        max_chars: Maximum characters per chunk
        overlap: Number of overlapping characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks
