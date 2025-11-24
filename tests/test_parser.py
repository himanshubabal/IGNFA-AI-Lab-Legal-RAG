"""Tests for parser module."""

import pytest

from raganything.parser import DoclingParser, MinerUParser, ParserFactory


def test_parser_factory_mineru():
    """Test creating MinerU parser."""
    parser = ParserFactory.create_parser("mineru", parse_method="auto")
    assert isinstance(parser, MinerUParser)
    assert parser.parse_method == "auto"


def test_parser_factory_docling():
    """Test creating Docling parser."""
    parser = ParserFactory.create_parser("docling")
    assert isinstance(parser, DoclingParser)


def test_parser_factory_invalid():
    """Test invalid parser type."""
    with pytest.raises(ValueError):
        ParserFactory.create_parser("invalid")


def test_mineru_parser_supported():
    """Test MinerU parser file support."""
    parser = MinerUParser()
    assert parser.is_supported("test.pdf") is True
    assert parser.is_supported("test.jpg") is True
    assert parser.is_supported("test.docx") is True
    assert parser.is_supported("test.unknown") is False


def test_docling_parser_supported():
    """Test Docling parser file support."""
    parser = DoclingParser()
    assert parser.is_supported("test.docx") is True
    assert parser.is_supported("test.html") is True
    assert parser.is_supported("test.pdf") is False


def test_get_parser_for_file():
    """Test getting parser for file."""
    # PDF should use MinerU
    parser = ParserFactory.get_parser_for_file("test.pdf")
    assert isinstance(parser, MinerUParser)

    # DOCX can use either, but should get one
    parser = ParserFactory.get_parser_for_file("test.docx")
    assert parser is not None

