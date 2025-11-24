"""
Utility functions for RAG-Anything.

This module provides helper functions for file operations, format detection,
validation, and logging setup.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def detect_file_type(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Detect file type and MIME type from file path.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (file_extension, mime_type)
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(path))

    return extension, mime_type


def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF."""
    extension, mime_type = detect_file_type(file_path)
    return extension == ".pdf" or mime_type == "application/pdf"


def is_image(file_path: str) -> bool:
    """Check if file is an image."""
    extension, mime_type = detect_file_type(file_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
    image_mimes = {
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/tiff",
        "image/gif",
        "image/webp",
    }
    return extension in image_extensions or (mime_type and mime_type in image_mimes)


def is_office_document(file_path: str) -> bool:
    """Check if file is an Office document."""
    extension, mime_type = detect_file_type(file_path)
    office_extensions = {
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
    }
    office_mimes = {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return extension in office_extensions or (mime_type and mime_type in office_mimes)


def is_text_file(file_path: str) -> bool:
    """Check if file is a text file."""
    extension, mime_type = detect_file_type(file_path)
    text_extensions = {".txt", ".md", ".markdown"}
    text_mimes = {"text/plain", "text/markdown"}
    return extension in text_extensions or (mime_type and mime_type in text_mimes)


def validate_file_path(file_path: str) -> Path:
    """
    Validate and return Path object for file.

    Args:
        file_path: Path to validate

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If path is invalid
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return path


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

