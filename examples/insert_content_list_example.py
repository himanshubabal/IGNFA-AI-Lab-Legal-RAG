"""
Example demonstrating content insertion from pre-parsed content.

This example shows how to insert content that has already been
parsed by external tools into the RAG-Anything knowledge base.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything import RAGAnything


def main():
    """Demonstrate content list insertion."""
    print("=" * 60)
    print("Insert Content List Example")
    print("=" * 60)

    # Initialize RAG-Anything
    print("\nInitializing RAG-Anything...")
    rag = RAGAnything()

    # Sample pre-parsed content
    # This could come from external parsers, APIs, or cached results
    content_list = [
        {
            "content": """
# Chapter 1: Introduction

This chapter introduces the main concepts and provides an overview
of the document. It covers the background and motivation for the work.
""",
            "metadata": {
                "source": "document1.pdf",
                "chapter": 1,
                "page": 1,
            },
        },
        {
            "content": """
# Chapter 2: Methodology

This chapter describes the methodology used in the research.
It includes detailed explanations of the experimental setup.
""",
            "metadata": {
                "source": "document1.pdf",
                "chapter": 2,
                "page": 5,
            },
        },
        {
            "content": """
# Chapter 3: Results

The results section presents the findings from the experiments.
Key observations and data are discussed in detail.
""",
            "metadata": {
                "source": "document1.pdf",
                "chapter": 3,
                "page": 10,
            },
        },
    ]

    print(f"\nInserting {len(content_list)} content items...")

    try:
        # Insert content list
        chunk_ids = rag.insert_content_list(
            content_list=content_list,
            doc_id="document1",
        )

        print(f"Successfully inserted {len(chunk_ids)} chunks")
        print(f"Chunk IDs: {chunk_ids[:5]}...")  # Show first 5

        # Now query the inserted content
        print("\n" + "=" * 60)
        print("Querying inserted content:")
        print("=" * 60)

        queries = [
            "What is the methodology?",
            "What are the main results?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            result = rag.query(query, n_results=3)
            print(f"Answer: {result['answer']}")
            if result.get("sources"):
                print(f"Sources: {', '.join(result['sources'])}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

