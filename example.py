"""Example usage of PocketRAG."""

from pathlib import Path
import sys

sys.path.insert(0, 'src')

from pocketrag.pocketrag import PocketRAG


def main():
    """Run example."""
    
    # Initialize PocketRAG
    print("Initializing PocketRAG...")
    rag = PocketRAG(
        data_dir=Path("data"),
        use_llm=False  # Set to True to use LLM (requires more resources)
    )
    
    # Create a sample document
    sample_dir = Path("data/sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_dir / "sample.txt"
    sample_file.write_text("""
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses statistical techniques to give computer systems the ability to learn and improve from experience.

There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data. Examples include classification and regression tasks.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Common techniques include clustering and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties.

Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has achieved remarkable success in image recognition, natural language processing, and game playing.

Popular machine learning frameworks include TensorFlow, PyTorch, and Scikit-learn. These tools make it easier for developers to build and deploy machine learning models.
""")
    
    # Ingest documents
    print("\nIngesting sample document...")
    documents = rag.ingest_documents(directory=sample_dir)
    print(f"Ingested {len(documents)} documents")
    
    # Query the system
    queries = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What are popular machine learning frameworks?"
    ]
    
    for query in queries:
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        result = rag.query(query, return_diagnostics=True)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nCitation Valid: {result['citation_valid']}")
        print(f"Entity Verified: {result['entity_verified']}")
        print(f"Sources: {', '.join(result['sources'])}")
        
        if 'diagnostics' in result:
            print(f"\nRetrieved chunks: {result['diagnostics']['num_retrieved']}")
            print(f"Re-ranked chunks: {result['diagnostics']['num_reranked']}")
    
    # Show statistics
    print("\n" + "="*80)
    print("System Statistics")
    print("="*80)
    stats = rag.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
