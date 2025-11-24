"""
Example demonstrating modal processors for different content types.

This example shows how to use ImageProcessor, TableProcessor,
and EquationProcessor to process multimodal content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.modalprocessors import (
    ImageProcessor,
    TableProcessor,
    EquationProcessor,
    get_processor,
)


def main():
    """Demonstrate modal processors."""
    print("=" * 60)
    print("Modal Processors Example")
    print("=" * 60)

    # Image Processor
    print("\n1. Image Processor")
    print("-" * 60)
    image_processor = ImageProcessor()
    image_result = image_processor.process(
        content=None,
        content_type="image",
        image_path="example_image.jpg",  # Replace with actual image path
        encode_base64=False,
    )
    print(f"Image processing result: {image_result}")

    # Table Processor
    print("\n2. Table Processor")
    print("-" * 60)
    table_processor = TableProcessor()
    sample_table = [
        {"Name": "Alice", "Age": 30, "City": "New York"},
        {"Name": "Bob", "Age": 25, "City": "London"},
        {"Name": "Charlie", "Age": 35, "City": "Tokyo"},
    ]
    table_result = table_processor.process(
        content=sample_table,
        content_type="table",
        format="markdown",
    )
    print("Table processing result:")
    print(table_result["content"])

    # Equation Processor
    print("\n3. Equation Processor")
    print("-" * 60)
    equation_processor = EquationProcessor()
    equation_result = equation_processor.process(
        content="E = mc^2",
        content_type="equation",
        format="latex",
    )
    print(f"Equation processing result: {equation_result}")

    # Using get_processor utility
    print("\n4. Using get_processor utility")
    print("-" * 60)
    processor = get_processor("table")
    result = processor.process(
        content=sample_table,
        content_type="table",
    )
    print("Result from get_processor:")
    print(result["content"])


if __name__ == "__main__":
    main()

