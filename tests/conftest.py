"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences and paragraphs.
    
    This is the second paragraph with more content.
    It helps test chunking and processing functionality.
    """


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "author": "Test Author",
        "date": "2025-01-01",
        "source": "test_document.pdf",
    }

