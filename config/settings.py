"""Configuration settings for PocketRAG."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Document ingestion settings
SUPPORTED_FORMATS = [".pdf", ".txt", ".md"]
MAX_FILE_SIZE_MB = 50

# Chunking settings
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens
MIN_CHUNK_SIZE = 100  # tokens

# Retrieval settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 10
BM25_WEIGHT = 0.5
FAISS_WEIGHT = 0.5

# Re-ranking settings
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 5

# LLM settings
LLM_MODEL = "gpt2"  # Placeholder - can be replaced with local models
MAX_CONTEXT_LENGTH = 2048
MAX_GENERATION_LENGTH = 512
TEMPERATURE = 0.1

# Citation settings
MIN_CITATIONS_REQUIRED = 1
CITATION_FORMAT = "[{doc_id}]"

# Hallucination guard settings
ENTITY_VERIFICATION_THRESHOLD = 0.7
MAX_ENTITY_MISMATCH = 2

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Performance settings
BATCH_SIZE = 32
USE_GPU = False
