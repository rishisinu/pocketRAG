#!/usr/bin/env python3
"""Command-line interface for PocketRAG."""

import argparse
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pocketrag.pocketrag import PocketRAG


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PocketRAG - Offline Document QA System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file to ingest"
    )
    ingest_parser.add_argument(
        "--directory",
        type=str,
        help="Path to directory to ingest"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument(
        "query",
        type=str,
        help="Query string"
    )
    query_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show diagnostic information"
    )
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    
    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize PocketRAG
    rag = PocketRAG(
        data_dir=Path("data"),
        use_llm=False  # Set to True for LLM-based generation
    )
    
    if args.command == "ingest":
        if args.file:
            documents = rag.ingest_documents(file_paths=[Path(args.file)])
            print(f"Ingested {len(documents)} document(s)")
        elif args.directory:
            documents = rag.ingest_documents(directory=Path(args.directory))
            print(f"Ingested {len(documents)} document(s)")
        else:
            print("Error: Please specify --file or --directory")
            return
    
    elif args.command == "query":
        result = rag.query(args.query, return_diagnostics=args.diagnostics)
        
        print("\n" + "="*80)
        print(f"Query: {result['query']}")
        print("="*80)
        print(f"\nAnswer: {result['answer']}\n")
        print(f"Citation Valid: {result['citation_valid']}")
        print(f"Entity Verified: {result['entity_verified']}")
        print(f"\nSources: {', '.join(result['sources'])}")
        
        if args.diagnostics and 'diagnostics' in result:
            print("\n--- Diagnostics ---")
            for key, value in result['diagnostics'].items():
                print(f"{key}: {value}")
    
    elif args.command == "stats":
        stats = rag.get_statistics()
        print("\n--- System Statistics ---")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    elif args.command == "server":
        import uvicorn
        from pocketrag.api.server import app
        
        print(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
