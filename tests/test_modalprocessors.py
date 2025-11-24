"""Tests for modal processors."""

import pytest

from raganything.modalprocessors import (
    EquationProcessor,
    GenericProcessor,
    ImageProcessor,
    TableProcessor,
    get_processor,
)


def test_image_processor_supports():
    """Test ImageProcessor supports method."""
    processor = ImageProcessor()
    assert processor.supports("image") is True
    assert processor.supports("img") is True
    assert processor.supports("table") is False


def test_table_processor_supports():
    """Test TableProcessor supports method."""
    processor = TableProcessor()
    assert processor.supports("table") is True
    assert processor.supports("tbl") is True
    assert processor.supports("image") is False


def test_equation_processor_supports():
    """Test EquationProcessor supports method."""
    processor = EquationProcessor()
    assert processor.supports("equation") is True
    assert processor.supports("formula") is True
    assert processor.supports("table") is False


def test_table_processor_process():
    """Test TableProcessor processing."""
    processor = TableProcessor()
    table_data = [
        {"Name": "Alice", "Age": 30},
        {"Name": "Bob", "Age": 25},
    ]

    result = processor.process(
        content=table_data,
        content_type="table",
        format="markdown",
    )

    assert result["type"] == "table"
    assert "content" in result
    assert "Alice" in result["content"]


def test_equation_processor_process():
    """Test EquationProcessor processing."""
    processor = EquationProcessor()
    result = processor.process(
        content="E = mc^2",
        content_type="equation",
        format="latex",
    )

    assert result["type"] == "equation"
    assert "E = mc^2" in result["content"]


def test_generic_processor():
    """Test GenericProcessor."""
    processor = GenericProcessor()
    assert processor.supports("anything") is True

    result = processor.process(
        content="test content",
        content_type="custom",
    )

    assert result["type"] == "custom"
    assert result["content"] == "test content"


def test_get_processor():
    """Test get_processor utility."""
    processor = get_processor("table")
    assert isinstance(processor, TableProcessor)

    processor = get_processor("image")
    assert isinstance(processor, ImageProcessor)

    processor = get_processor("unknown")
    assert isinstance(processor, GenericProcessor)

