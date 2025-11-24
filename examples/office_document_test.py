"""
Test script for Office document parsing with MinerU.

This script tests parsing of Office documents (.doc, .docx, .ppt, .pptx, .xls, .xlsx)
using the MinerU parser. No API key required for parsing only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.parser import MinerUParser


def check_libreoffice():
    """Check if LibreOffice is installed."""
    import subprocess

    try:
        result = subprocess.run(
            ["soffice", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"LibreOffice found: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("Warning: LibreOffice not found. Office document parsing may not work.")
    print("Install LibreOffice: https://www.libreoffice.org/download/")
    return False


def main():
    """Test Office document parsing."""
    parser = argparse.ArgumentParser(description="Test Office document parsing")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to Office document",
    )
    parser.add_argument(
        "--check-libreoffice",
        action="store_true",
        help="Check if LibreOffice is installed",
    )

    args = parser.parse_args()

    if args.check_libreoffice:
        check_libreoffice()
        return 0

    # Check LibreOffice
    if not check_libreoffice():
        print("\nContinuing anyway...")

    # Test parsing
    print(f"\nTesting Office document parsing: {args.file}")
    print("=" * 60)

    try:
        mineru_parser = MinerUParser(parse_method="auto")

        if not mineru_parser.is_supported(args.file):
            print(f"Error: File type not supported by MinerU: {args.file}")
            return 1

        print("Parsing document...")
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

