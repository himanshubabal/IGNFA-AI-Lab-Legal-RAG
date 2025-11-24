"""
Command-line interface for RAG-Anything.
"""

import argparse
import sys
from pathlib import Path

from raganything import RAGAnything
from raganything.smart_processor import SmartProcessor


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG-Anything: All-in-One RAG Framework"
    )
    parser.add_argument(
        "command",
        choices=["process", "process-all", "query", "batch", "ui", "status"],
        help="Command to execute",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="File path to process",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query string",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["mineru", "docling"],
        help="Parser to use",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--documents-dir",
        "-d",
        type=str,
        default="documents",
        help="Documents directory (default: documents)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all documents",
    )

    args = parser.parse_args()

    if args.command == "process":
        if not args.file:
            print("Error: --file is required for 'process' command")
            sys.exit(1)

        rag = RAGAnything(parser=args.parser)
        result = rag.process_document_complete(file_path=args.file, output_dir=args.output_dir)
        print(f"Processed: {result.get('num_chunks', 0)} chunks created")

    elif args.command == "process-all":
        print(f"Processing all documents in: {args.documents_dir}")
        processor = SmartProcessor(documents_dir=args.documents_dir)
        results = processor.process_all(force_reprocess=args.force)

        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"New documents: {len(results['new'])}")
        for doc in results["new"]:
            print(f"  ✓ {Path(doc['path']).name} ({doc['chunks']} chunks)")

        print(f"\nUpdated documents: {len(results['updated'])}")
        for doc in results["updated"]:
            print(f"  ↻ {Path(doc['path']).name} ({doc['chunks']} chunks)")

        print(f"\nRemoved documents: {len(results['removed'])}")
        for doc_path in results["removed"]:
            print(f"  ✗ {Path(doc_path).name}")

        print(f"\nUnchanged documents: {len(results['unchanged'])}")

        if results["errors"]:
            print(f"\nErrors: {len(results['errors'])}")
            for error in results["errors"]:
                print(f"  ✗ {Path(error['path']).name}: {error['error']}")

        print("=" * 60)

    elif args.command == "status":
        processor = SmartProcessor(documents_dir=args.documents_dir)
        status = processor.get_document_status()

        print("\n" + "=" * 60)
        print("Document Status")
        print("=" * 60)
        print(f"Total files: {status['total_files']}")
        print(f"Processed: {len(status['processed'])}")
        print(f"Unprocessed: {len(status['unprocessed'])}")
        print(f"Removed (tracked but missing): {len(status['removed'])}")

        if status["processed"]:
            print("\nProcessed Documents:")
            for doc in status["processed"]:
                print(f"  ✓ {Path(doc['path']).name}")
                print(f"    ID: {doc['doc_id']}, Chunks: {doc['chunks']}")

        if status["unprocessed"]:
            print("\nUnprocessed Documents:")
            for doc in status["unprocessed"]:
                print(f"  ○ {Path(doc['path']).name}")

        print("=" * 60)

    elif args.command == "query":
        if not args.query:
            print("Error: --query is required for 'query' command")
            sys.exit(1)

        rag = RAGAnything(parser=args.parser)
        result = rag.query(args.query)
        print(f"Answer: {result['answer']}")

    elif args.command == "ui":
        try:
            import streamlit.web.cli as stcli
            app_path = Path(__file__).parent / "webui" / "streamlit_app.py"
            sys.argv = ["streamlit", "run", str(app_path)]
            stcli.main()
        except ImportError:
            print("Error: Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

