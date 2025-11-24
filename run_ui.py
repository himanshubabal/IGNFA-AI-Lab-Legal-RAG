#!/usr/bin/env python3
"""
Launch script for RAG-Anything Web UI.

This script launches the Streamlit web interface with automatic
document processing on startup.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from raganything.smart_processor import SmartProcessor
from raganything import RAGAnything


def main():
    """Launch web UI with automatic processing."""
    print("=" * 60)
    print("RAG-Anything Web UI")
    print("=" * 60)
    print()

    # Initialize and process documents
    print("Initializing RAG-Anything...")
    rag = RAGAnything()
    processor = SmartProcessor(documents_dir="documents", raganything=rag)

    print("Scanning documents directory...")
    results = processor.process_all()

    if results["new"] or results["updated"] or results["removed"]:
        print(f"\nProcessing Summary:")
        print(f"  - New documents: {len(results['new'])}")
        print(f"  - Updated documents: {len(results['updated'])}")
        print(f"  - Removed documents: {len(results['removed'])}")
    else:
        print("No changes detected. All documents are up to date.")

    print("\n" + "=" * 60)
    print("Launching Web UI...")
    print("=" * 60)
    print()

    # Launch Streamlit
    app_path = project_root / "raganything" / "webui" / "streamlit_app.py"
    subprocess.run(
        ["streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

