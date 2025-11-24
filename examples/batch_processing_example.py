"""
Example demonstrating batch processing of multiple documents.

This example shows how to process multiple documents efficiently
using the BatchProcessor class.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.batch import BatchProcessor
from raganything import RAGAnything


def main():
    """Demonstrate batch processing."""
    print("=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    # Initialize batch processor
    print("\nInitializing batch processor...")
    batch_processor = BatchProcessor(
        raganything=RAGAnything(parser="mineru", parse_method="auto")
    )

    # Example file paths (replace with actual paths)
    file_paths = [
        "document1.pdf",
        "document2.pdf",
        "document3.docx",
    ]

    print(f"\nProcessing {len(file_paths)} documents...")
    print("File paths:")
    for path in file_paths:
        print(f"  - {path}")

    # Process batch
    try:
        results = batch_processor.process_batch(
            file_paths=file_paths,
            display_progress=True,
            continue_on_error=True,
        )

        # Display summary
        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Total chunks created: {results['total_chunks']}")

        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  - {error['file_path']}: {error['error']}")

    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return 1

    # Example: Process directory
    print("\n" + "=" * 60)
    print("Directory Processing Example")
    print("=" * 60)
    print("\nTo process all files in a directory:")
    print("  results = batch_processor.process_directory(")
    print("      directory='./documents',")
    print("      pattern='*.pdf',")
    print("      recursive=True,")
    print("  )")

    return 0


if __name__ == "__main__":
    sys.exit(main())

