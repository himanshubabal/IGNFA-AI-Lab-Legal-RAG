"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pytest

from raganything.utils import (
    detect_file_type,
    ensure_directory,
    format_file_size,
    is_image,
    is_office_document,
    is_pdf,
    is_text_file,
    sanitize_filename,
    setup_logging,
    validate_file_path,
)


def test_detect_file_type():
    """Test file type detection."""
    extension, mime_type = detect_file_type("test.pdf")
    assert extension == ".pdf"
    assert mime_type == "application/pdf"


def test_is_pdf():
    """Test PDF detection."""
    assert is_pdf("test.pdf") is True
    assert is_pdf("test.txt") is False


def test_is_image():
    """Test image detection."""
    assert is_image("test.jpg") is True
    assert is_image("test.png") is True
    assert is_image("test.pdf") is False


def test_is_office_document():
    """Test Office document detection."""
    assert is_office_document("test.docx") is True
    assert is_office_document("test.xlsx") is True
    assert is_office_document("test.pdf") is False


def test_is_text_file():
    """Test text file detection."""
    assert is_text_file("test.txt") is True
    assert is_text_file("test.md") is True
    assert is_text_file("test.pdf") is False


def test_validate_file_path(tmp_path):
    """Test file path validation."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    path = validate_file_path(str(test_file))
    assert path == test_file.resolve()

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        validate_file_path(str(tmp_path / "nonexistent.txt"))


def test_ensure_directory(tmp_path):
    """Test directory creation."""
    new_dir = tmp_path / "new_dir" / "sub_dir"
    result = ensure_directory(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_sanitize_filename():
    """Test filename sanitization."""
    assert sanitize_filename("test<file>.txt") == "test_file_.txt"
    assert sanitize_filename("  test.txt  ") == "test.txt"
    assert sanitize_filename("test/file.txt") == "test_file.txt"


def test_format_file_size():
    """Test file size formatting."""
    assert "B" in format_file_size(500)
    assert "KB" in format_file_size(1024)
    assert "MB" in format_file_size(1024 * 1024)


def test_setup_logging():
    """Test logging setup."""
    logger = setup_logging()
    assert logger is not None

