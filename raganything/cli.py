"""
Command-line interface for RAG-Anything.
"""

import argparse
import sys
from pathlib import Path

from raganything import RAGAnything


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG-Anything: All-in-One RAG Framework"
    )
    parser.add_argument(
        "command",
        choices=["process", "query", "batch", "ui"],
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

    args = parser.parse_args()

    if args.command == "process":
        if not args.file:
            print("Error: --file is required for 'process' command")
            sys.exit(1)

        rag = RAGAnything(parser=args.parser)
        result = rag.process_document_complete(file_path=args.file, output_dir=args.output_dir)
        print(f"Processed: {result.get('num_chunks', 0)} chunks created")

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

