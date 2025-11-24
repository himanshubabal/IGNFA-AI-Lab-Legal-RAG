"""
Basic example of using RAG-Anything for document processing and querying.

This example demonstrates:
1. Processing a document
2. Querying the knowledge base
3. Getting search results
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything import RAGAnything


def main():
    """Main example function."""
    # Initialize RAG-Anything
    print("Initializing RAG-Anything...")
    rag = RAGAnything(
        parser="mineru",  # or "docling"
        parse_method="auto",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Example: Process a document
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nProcessing document: {file_path}")

        try:
            result = rag.process_document_complete(
                file_path=file_path,
                display_stats=True,
            )

            print(f"\nDocument processed successfully!")
            print(f"Number of chunks: {result.get('num_chunks', 0)}")
            print(f"Output file: {result.get('output_file', 'N/A')}")

            # Example: Query the knowledge base
            print("\n" + "=" * 50)
            print("Querying knowledge base...")
            print("=" * 50)

            queries = [
                "What is this document about?",
                "Summarize the main points.",
            ]

            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.query(query, n_results=3)
                print(f"Answer: {answer['answer']}")
                if answer.get('sources'):
                    print(f"Sources: {', '.join(answer['sources'])}")

        except Exception as e:
            print(f"Error: {str(e)}")
            return 1
    else:
        print("Usage: python raganything_example.py <document_path>")
        print("\nExample:")
        print("  python raganything_example.py document.pdf")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

