# PocketRAG Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/rishisinu/pocketRAG.git
cd pocketRAG

# Install dependencies
pip install -r requirements.txt

# (Optional) Download spaCy model for better entity extraction
python -m spacy download en_core_web_sm
```

## 5-Minute Tutorial

### 1. Basic Usage (Python API)

```python
from pathlib import Path
from pocketrag.pocketrag import PocketRAG

# Initialize the system
rag = PocketRAG(use_llm=False)  # Fast mode without LLM

# Ingest documents
rag.ingest_documents(directory=Path("path/to/your/documents"))

# Query
result = rag.query("What is machine learning?")

print(result['answer'])
print(f"Sources: {result['sources']}")
```

### 2. CLI Usage

```bash
# Ingest documents
python cli.py ingest --directory /path/to/docs

# Query
python cli.py query "What is machine learning?"

# Get statistics
python cli.py stats
```

### 3. REST API

```bash
# Start the server
python cli.py server

# In another terminal, query the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

## Configuration

Edit `config/settings.py` to customize:

- **CHUNK_SIZE**: Default 512 tokens
- **TOP_K_RETRIEVAL**: Default 10 chunks
- **TOP_K_RERANK**: Default 5 chunks
- **Use LLM**: Set `use_llm=True` for better answers (slower)

## Performance Tips

1. **Fast Mode** (sub-second): Use `use_llm=False`
2. **Quality Mode** (2-5s): Use `use_llm=True`
3. **GPU Acceleration**: Set `device="cuda"` if available
4. **Smaller Models**: Use smaller embedding models for faster retrieval

## Architecture Flow

```
Documents → Ingestion → Chunking → Indexing
                                       ↓
Query → Hybrid Retrieval (BM25+FAISS) → Re-ranking
                                           ↓
                             Answer Generation → Validation
                                                      ↓
                                              Citation-backed Answer
```

## Key Features

✅ **Fully Offline** - No external API calls
✅ **Citation-backed** - All answers include source references
✅ **Hallucination Guard** - Entity-based verification
✅ **Hybrid Retrieval** - BM25 + semantic search
✅ **Fast** - Sub-second latency possible
✅ **Evaluation Metrics** - Built-in quality assessment

## Troubleshooting

**Issue**: Slow first query
**Solution**: Models are loaded on first use. Subsequent queries are fast.

**Issue**: No citations in answers
**Solution**: Check that documents were properly indexed. Run `rag.get_statistics()`.

**Issue**: Poor answer quality
**Solution**: Try enabling LLM mode with `use_llm=True`.

## Next Steps

1. Try the example: `python example.py`
2. Read the full documentation in README.md
3. Explore API docs at `http://localhost:8000/docs`
4. Customize configuration in `config/settings.py`
