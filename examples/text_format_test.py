"""
Test script for text format parsing with MinerU.

This script tests parsing of text files (.txt, .md) using the MinerU parser.
No API key required for parsing only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.parser import MinerUParser


def check_reportlab():
    """Check if ReportLab is installed."""
    try:
        import reportlab

        print(f"ReportLab found: {reportlab.Version}")
        return True
    except ImportError:
        print("Warning: ReportLab not installed. Text processing features may be limited.")
        print("Install with: pip install ReportLab")
        return False


def main():
    """Test text format parsing."""
    parser = argparse.ArgumentParser(description="Test text format parsing")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to text file",
    )
    parser.add_argument(
        "--check-reportlab",
        action="store_true",
        help="Check if ReportLab is installed",
    )

    args = parser.parse_args()

    if args.check_reportlab:
        check_reportlab()
        return 0

    # Check ReportLab
    check_reportlab()

    # Test parsing
    print(f"\nTesting text format parsing: {args.file}")
    print("=" * 60)

    try:
        mineru_parser = MinerUParser(parse_method="auto")

        if not mineru_parser.is_supported(args.file):
            print(f"Error: File type not supported by MinerU: {args.file}")
            return 1

        print("Parsing text file...")
        result = mineru_parser.parse(file_path=args.file)

        print("\nParsing successful!")
        print(f"Parser: {result.get('parser')}")
        print(f"Output file: {result.get('output_file', 'N/A')}")
        print(f"Content length: {len(result.get('content', ''))} characters")

        # Show preview
        content = result.get("content", "")
        if content:
            preview = content[:500] if len(content) > 500 else content
            print(f"\nContent preview:\n{preview}...")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

