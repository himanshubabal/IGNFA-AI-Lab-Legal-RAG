"""
Test script for image format parsing with MinerU.

This script tests parsing of image files (.jpg, .png, .bmp, .tiff, .gif, .webp)
using the MinerU parser. No API key required for parsing only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.parser import MinerUParser


def check_pillow():
    """Check if PIL/Pillow is installed."""
    try:
        from PIL import Image

        print(f"Pillow found: {Image.__version__}")
        return True
    except ImportError:
        print("Warning: Pillow not installed. Extended image format support may be limited.")
        print("Install with: pip install Pillow")
        return False


def main():
    """Test image format parsing."""
    parser = argparse.ArgumentParser(description="Test image format parsing")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--check-pillow",
        action="store_true",
        help="Check if Pillow is installed",
    )

    args = parser.parse_args()

    if args.check_pillow:
        check_pillow()
        return 0

    # Check Pillow
    check_pillow()

    # Test parsing
    print(f"\nTesting image format parsing: {args.file}")
    print("=" * 60)

    try:
        mineru_parser = MinerUParser(parse_method="auto")

        if not mineru_parser.is_supported(args.file):
            print(f"Error: File type not supported by MinerU: {args.file}")
            return 1

        print("Parsing image...")
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

