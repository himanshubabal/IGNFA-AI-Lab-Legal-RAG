"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest

from raganything.config import Config, get_config


def test_config_singleton():
    """Test that Config is a singleton."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


def test_get_config():
    """Test get_config function."""
    config = get_config()
    assert isinstance(config, Config)


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    assert config.parser in ["mineru", "docling"]
    assert config.parse_method in ["auto", "ocr", "txt"]
    assert config.output_dir.exists()


def test_config_env_vars(monkeypatch):
    """Test configuration from environment variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("OUTPUT_DIR", tmpdir)
        monkeypatch.setenv("PARSER", "docling")
        monkeypatch.setenv("PARSE_METHOD", "ocr")

        # Reset singleton
        Config._instance = None
        Config._initialized = False

        config = Config()
        assert config.parser == "docling"
        assert config.parse_method == "ocr"
        assert str(config.output_dir) == tmpdir


def test_get_output_path():
    """Test get_output_path method."""
    config = Config()
    output_path = config.get_output_path("test.txt")
    assert output_path.name == "test.txt"
    assert output_path.parent == config.output_dir

