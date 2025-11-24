"""Tests for processor module."""

import pytest

from raganything.processor import ContentProcessor, TextChunker


def test_text_chunker_character():
    """Test character-based chunking."""
    text = "a" * 5000
    chunks = TextChunker.chunk_by_character(text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) > 0
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_text_chunker_sentence():
    """Test sentence-based chunking."""
    text = "Sentence one. Sentence two. Sentence three. " * 10
    chunks = TextChunker.chunk_by_sentence(text, chunk_size=100)

    assert len(chunks) > 0


def test_text_chunker_paragraph():
    """Test paragraph-based chunking."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = TextChunker.chunk_by_paragraph(text, chunk_size=100)

    assert len(chunks) > 0


def test_content_processor_chunk_text():
    """Test ContentProcessor chunking."""
    processor = ContentProcessor(chunk_strategy="character")
    text = "a" * 2000
    chunks = processor.chunk_text(text)

    assert len(chunks) > 0


def test_content_processor_get_statistics():
    """Test ContentProcessor statistics."""
    processor = ContentProcessor()
    text = "This is a test document with multiple words."
    stats = processor.get_statistics(text)

    assert "total_chars" in stats
    assert "total_words" in stats
    assert "total_chunks" in stats
    assert stats["total_chars"] == len(text)

