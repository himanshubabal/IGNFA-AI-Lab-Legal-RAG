"""
Example demonstrating enhanced markdown generation.

This example shows how to generate structured markdown
with multimodal elements (images, tables, equations).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.enhanced_markdown import (
    EnhancedMarkdownGenerator,
    generate_enhanced_markdown,
)


def main():
    """Demonstrate enhanced markdown generation."""
    print("=" * 60)
    print("Enhanced Markdown Example")
    print("=" * 60)

    # Sample content
    content = """
# Document Title

This is the main content of the document. It contains important information
about various topics including images, tables, and equations.
"""

    # Sample metadata
    metadata = {
        "author": "John Doe",
        "date": "2025-01-01",
        "version": "1.0",
    }

    # Sample images
    images = [
        {
            "content": "path/to/image1.jpg",
            "metadata": {"path": "path/to/image1.jpg", "size": 102400},
        },
        {
            "content": "path/to/image2.png",
            "metadata": {"path": "path/to/image2.png", "size": 204800},
        },
    ]

    # Sample tables
    tables = [
        {
            "content": """| Name | Age | City |
|------|-----|------|
| Alice | 30 | New York |
| Bob | 25 | London |""",
            "metadata": {"rows": 2},
        }
    ]

    # Sample equations
    equations = [
        {
            "content": "E = mc^2",
            "format": "latex",
        },
        {
            "content": "\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
            "format": "latex",
        },
    ]

    # Generate enhanced markdown
    print("\nGenerating enhanced markdown...")
    markdown = generate_enhanced_markdown(
        content=content,
        metadata=metadata,
        images=images,
        tables=tables,
        equations=equations,
        include_metadata=True,
    )

    print("\n" + "=" * 60)
    print("Generated Markdown:")
    print("=" * 60)
    print(markdown)

    # Using EnhancedMarkdownGenerator class directly
    print("\n" + "=" * 60)
    print("Using EnhancedMarkdownGenerator class:")
    print("=" * 60)
    generator = EnhancedMarkdownGenerator(include_metadata=True)

    # Add elements individually
    base_markdown = "# My Document\n\nSome content here."
    markdown_with_image = generator.add_image_reference(
        base_markdown, "image.jpg", alt_text="Example Image"
    )
    markdown_with_table = generator.add_table(
        markdown_with_image,
        """| Col1 | Col2 |
|------|------|
| A | B |""",
        caption="Sample Table",
    )
    markdown_with_equation = generator.add_equation(
        markdown_with_table, "x^2 + y^2 = r^2", format="latex"
    )

    print(markdown_with_equation)


if __name__ == "__main__":
    main()

