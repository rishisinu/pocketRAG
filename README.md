# PocketRAG

A fully offline document QA system for querying local PDFs and notes with citation-backed answers.

## Features

🔒 **Fully Offline** - All processing happens locally, no external API calls  
📚 **Multi-format Support** - PDF, TXT, and Markdown files  
🧩 **Token-aware Chunking** - Smart document splitting with metadata preservation  
🔍 **Hybrid Retrieval** - Combines BM25 keyword search with FAISS semantic search  
🎯 **Re-ranking** - Cross-encoder re-ranking for improved precision  
📖 **Citation-backed Answers** - All answers include source citations  
🛡️ **Hallucination Guard** - Entity-based verification prevents false information  
⚡ **Sub-second Latency** - Optimized for fast responses  
🚀 **FastAPI Server** - Production-ready REST API  
📊 **Evaluation Metrics** - Built-in retrieval and citation quality metrics  

## Architecture

```
┌─────────────────┐
│  PDF/TXT/MD     │
│  Documents      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │
│   & Chunking    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Hybrid Retrieval       │
│  ┌─────────┬─────────┐  │
│  │  BM25   │ FAISS   │  │
│  └─────────┴─────────┘  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│   Re-ranking    │
│  (Cross-encoder)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator  │
│  (Local Model)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hallucination  │
│     Guard       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Citation-backed │
│     Answer      │
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download spaCy Model (for hallucination guard)

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Using Python API

```python
from pathlib import Path
from pocketrag.pocketrag import PocketRAG

# Initialize PocketRAG
rag = PocketRAG(
    data_dir=Path("data"),
    use_llm=False  # Set to True for LLM-based generation
)

# Ingest documents
rag.ingest_documents(directory=Path("path/to/documents"))

# Query the system
result = rag.query("What is machine learning?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Citations: {result['citations']}")
```

### 2. Using CLI

```bash
# Ingest a single file
python cli.py ingest --file path/to/document.pdf

# Ingest a directory
python cli.py ingest --directory path/to/documents/

# Query the system
python cli.py query "What is machine learning?"

# Query with diagnostics
python cli.py query "What is machine learning?" --diagnostics

# Show statistics
python cli.py stats
```

### 3. Using FastAPI Server

```bash
# Start the server
python cli.py server --host 0.0.0.0 --port 8000

# Or directly
cd src/pocketrag/api
python server.py
```

Then access the API at `http://localhost:8000`

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `POST /ingest/file` - Upload and ingest a file
- `POST /ingest/directory` - Ingest from directory
- `POST /query` - Query the system
- `GET /stats` - Get system statistics

**Example API Request:**

```bash
# Ingest a file
curl -X POST "http://localhost:8000/ingest/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "return_diagnostics": false}'
```

## Configuration

Edit `config/settings.py` to customize:

- **Chunking**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_SIZE`
- **Retrieval**: `TOP_K_RETRIEVAL`, `BM25_WEIGHT`, `FAISS_WEIGHT`
- **Re-ranking**: `TOP_K_RERANK`, `RERANK_MODEL`
- **LLM**: `LLM_MODEL`, `MAX_CONTEXT_LENGTH`, `TEMPERATURE`
- **Citations**: `MIN_CITATIONS_REQUIRED`, `CITATION_FORMAT`
- **Hallucination Guard**: `ENTITY_VERIFICATION_THRESHOLD`, `MAX_ENTITY_MISMATCH`

## Components

### 1. Document Ingestion
- Supports PDF, TXT, and MD files
- Extracts text and metadata
- Handles multiple documents efficiently

### 2. Token-aware Chunking
- Splits documents using token counts (not just characters)
- Configurable chunk size and overlap
- Preserves document metadata in chunks

### 3. Hybrid Retrieval
- **BM25**: Sparse keyword-based retrieval
- **FAISS**: Dense semantic search using embeddings
- Weighted combination of both methods

### 4. Re-ranking
- Cross-encoder model for precise relevance scoring
- Improves precision of top results
- Configurable number of final results

### 5. Answer Generation
- Template-based (fast, default)
- LLM-based (optional, slower but better quality)
- Enforces citation rules in prompts

### 6. Hallucination Guard
- Extracts named entities from answers and sources
- Verifies entity overlap
- Detects potential hallucinations

### 7. Evaluation Metrics
- Retrieval metrics: Precision@K, Recall@K, MRR
- Citation metrics: Coverage, accuracy

## Examples

See `example.py` for a complete working example:

```bash
python example.py
```

## Performance

The system is optimized for sub-second latency:

- **Ingestion**: ~1-2 seconds per PDF page
- **Indexing**: ~5-10 seconds for 100 chunks
- **Query**: ~0.3-0.8 seconds (without LLM)
- **Query with LLM**: ~2-5 seconds (depends on model)

Tips for faster performance:
- Use smaller embedding models
- Reduce `TOP_K_RETRIEVAL` and `TOP_K_RERANK`
- Disable LLM generation for fastest results
- Use GPU acceleration if available

## Testing

Run tests:

```bash
pytest tests/
```

## Project Structure

```
pocketRAG/
├── src/
│   └── pocketrag/
│       ├── ingestion/       # Document ingestion
│       ├── chunking/        # Token-aware chunking
│       ├── retrieval/       # BM25 + FAISS hybrid retrieval
│       ├── ranking/         # Cross-encoder re-ranking
│       ├── llm/             # Local LLM integration
│       ├── guards/          # Hallucination detection
│       ├── api/             # FastAPI server
│       ├── evaluation/      # Metrics and evaluation
│       ├── utils/           # Utility functions
│       └── pocketrag.py     # Main system class
├── config/
│   └── settings.py          # Configuration
├── tests/                   # Test files
├── data/                    # Data storage (created at runtime)
├── cli.py                   # Command-line interface
├── example.py               # Example usage
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Dependencies

Core dependencies:
- `fastapi` - Web framework
- `pypdf` - PDF parsing
- `sentence-transformers` - Embeddings and re-ranking
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - BM25 implementation
- `transformers` - LLM support
- `spacy` - Named entity recognition
- `tiktoken` - Token counting

## Limitations

- LLM generation requires significant resources (CPU/GPU)
- Large documents may require more memory
- First query after startup is slower (model loading)
- PDF parsing quality depends on source document

## Future Enhancements

- [ ] Support for more document formats (DOCX, HTML)
- [ ] Query caching for repeated questions
- [ ] Batch processing for multiple queries
- [ ] Advanced citation linking and highlighting
- [ ] Web UI for easier interaction
- [ ] Docker containerization
- [ ] GPU acceleration support
- [ ] Streaming responses

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use PocketRAG in your research or project, please cite:

```
@software{pocketrag2024,
  title={PocketRAG: Offline Document QA System},
  author={PocketRAG Contributors},
  year={2024},
  url={https://github.com/rishisinu/pocketRAG}
}
```
