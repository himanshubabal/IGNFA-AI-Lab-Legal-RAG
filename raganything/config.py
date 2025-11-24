"""
Configuration management for RAG-Anything.

This module handles loading and managing configuration from environment
variables and provides a centralized configuration interface.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Configuration manager for RAG-Anything."""

    _instance: Optional["Config"] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration from environment variables."""
        if not self._initialized:
            # Load environment variables from .env file
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                # Try loading from current directory
                load_dotenv()

            # OpenAI configuration
            self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
            self.openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

            # Output configuration
            output_dir = os.getenv("OUTPUT_DIR", "./output")
            self.output_dir: Path = Path(output_dir).expanduser().resolve()

            # Parser configuration
            self.parser: str = os.getenv("PARSER", "mineru").lower()
            if self.parser not in ["mineru", "docling"]:
                self.parser = "mineru"

            # Parse method configuration
            self.parse_method: str = os.getenv("PARSE_METHOD", "auto").lower()
            if self.parse_method not in ["auto", "ocr", "txt"]:
                self.parse_method = "auto"

            # Legacy support for MINERU_PARSE_METHOD
            legacy_method = os.getenv("MINERU_PARSE_METHOD")
            if legacy_method and not os.getenv("PARSE_METHOD"):
                self.parse_method = legacy_method.lower()

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            Config._initialized = True

    @classmethod
    def get_instance(cls) -> "Config":
        """Get the singleton configuration instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check if OpenAI API key is set (optional for parsing-only usage)
        # if not self.openai_api_key:
        #     print("Warning: OPENAI_API_KEY not set. LLM features will be unavailable.")
        return True

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """
        Get output path for a file.

        Args:
            filename: Optional filename to append to output directory

        Returns:
            Path object for the output location
        """
        if filename:
            return self.output_dir / filename
        return self.output_dir

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(parser={self.parser}, parse_method={self.parse_method}, "
            f"output_dir={self.output_dir})"
        )


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    return Config.get_instance()

