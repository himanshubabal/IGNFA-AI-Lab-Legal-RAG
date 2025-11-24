"""Tests for enhanced markdown module."""

from raganything.enhanced_markdown import (
    EnhancedMarkdownGenerator,
    generate_enhanced_markdown,
)


def test_generate_enhanced_markdown():
    """Test generate_enhanced_markdown function."""
    content = "# Test Document\n\nContent here."
    metadata = {"author": "Test Author"}

    markdown = generate_enhanced_markdown(
        content=content,
        metadata=metadata,
        include_metadata=True,
    )

    assert "# Test Document" in markdown
    assert "Test Author" in markdown


def test_enhanced_markdown_generator():
    """Test EnhancedMarkdownGenerator class."""
    generator = EnhancedMarkdownGenerator(include_metadata=True)

    content = "# Test"
    markdown = generator.generate(content=content)

    assert "# Test" in markdown


def test_add_image_reference():
    """Test adding image reference."""
    generator = EnhancedMarkdownGenerator()
    markdown = "# Document"
    result = generator.add_image_reference(markdown, "image.jpg", "Alt text")

    assert "image.jpg" in result
    assert "Alt text" in result


def test_add_table():
    """Test adding table."""
    generator = EnhancedMarkdownGenerator()
    markdown = "# Document"
    table = "| Col1 | Col2 |\n|------|------|\n| A | B |"
    result = generator.add_table(markdown, table, "Table Caption")

    assert "Col1" in result
    assert "Table Caption" in result


def test_add_equation():
    """Test adding equation."""
    generator = EnhancedMarkdownGenerator()
    markdown = "# Document"
    result = generator.add_equation(markdown, "E = mc^2", format="latex")

    assert "E = mc^2" in result or "mc^2" in result

