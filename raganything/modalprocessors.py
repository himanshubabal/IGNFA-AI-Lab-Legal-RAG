"""
Multimodal content processors for RAG-Anything.

This module provides processors for different content types:
images, tables, equations, and generic content.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from raganything.base import BaseModalProcessor

logger = logging.getLogger(__name__)


class ImageProcessor(BaseModalProcessor):
    """Processor for image content."""

    def supports(self, content_type: str) -> bool:
        """Check if processor supports the content type."""
        return content_type.lower() in ["image", "img", "picture", "photo"]

    def process(
        self,
        content: Any,
        content_type: str,
        image_path: Optional[str] = None,
        encode_base64: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process image content.

        Args:
            content: Image content (path, bytes, or base64)
            content_type: Type of content (should be 'image')
            image_path: Optional path to image file
            encode_base64: Whether to encode image as base64
            **kwargs: Additional parameters

        Returns:
            Dictionary with processed image data
        """
        result: Dict[str, Any] = {
            "type": "image",
            "content": None,
            "metadata": {},
        }

        try:
            # Handle different input types
            if image_path:
                path = Path(image_path)
                if path.exists():
                    with open(path, "rb") as f:
                        image_data = f.read()
                    result["content"] = base64.b64encode(image_data).decode("utf-8") if encode_base64 else str(path)
                    result["metadata"]["path"] = str(path)
                    result["metadata"]["size"] = len(image_data)
                else:
                    logger.warning(f"Image file not found: {image_path}")
            elif isinstance(content, bytes):
                result["content"] = base64.b64encode(content).decode("utf-8") if encode_base64 else "bytes"
                result["metadata"]["size"] = len(content)
            elif isinstance(content, str):
                # Assume it's a path or base64
                if Path(content).exists():
                    with open(content, "rb") as f:
                        image_data = f.read()
                    result["content"] = base64.b64encode(image_data).decode("utf-8") if encode_base64 else content
                    result["metadata"]["path"] = content
                    result["metadata"]["size"] = len(image_data)
                else:
                    # Assume base64
                    result["content"] = content
            else:
                logger.warning(f"Unsupported image content type: {type(content)}")

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            result["error"] = str(e)

        return result


class TableProcessor(BaseModalProcessor):
    """Processor for table content."""

    def supports(self, content_type: str) -> bool:
        """Check if processor supports the content type."""
        return content_type.lower() in ["table", "tbl", "data-table"]

    def process(
        self,
        content: Any,
        content_type: str,
        format: str = "markdown",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process table content.

        Args:
            content: Table content (dict, list, or markdown string)
            content_type: Type of content (should be 'table')
            format: Output format ('markdown', 'html', 'json')
            **kwargs: Additional parameters

        Returns:
            Dictionary with processed table data
        """
        result: Dict[str, Any] = {
            "type": "table",
            "content": None,
            "format": format,
            "metadata": {},
        }

        try:
            if isinstance(content, str):
                # Assume markdown table
                result["content"] = content
            elif isinstance(content, (list, dict)):
                # Convert to markdown table
                if format == "markdown":
                    result["content"] = self._to_markdown_table(content)
                elif format == "json":
                    import json
                    result["content"] = json.dumps(content, indent=2)
                else:
                    result["content"] = str(content)
            else:
                result["content"] = str(content)

            result["metadata"]["rows"] = self._count_rows(content)

        except Exception as e:
            logger.error(f"Error processing table: {str(e)}")
            result["error"] = str(e)

        return result

    def _to_markdown_table(self, data: Any) -> str:
        """Convert data to markdown table format."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # List of dictionaries
                headers = list(data[0].keys())
                rows = [[str(item.get(h, "")) for h in headers] for item in data]
            else:
                # List of lists
                headers = [f"Column {i+1}" for i in range(len(data[0]))]
                rows = data
        elif isinstance(data, dict):
            # Single dictionary
            headers = list(data.keys())
            rows = [[str(v) for v in data.values()]]
        else:
            return str(data)

        # Build markdown table
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def _count_rows(self, data: Any) -> int:
        """Count rows in table data."""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return 1
        return 0


class EquationProcessor(BaseModalProcessor):
    """Processor for mathematical equation content."""

    def supports(self, content_type: str) -> bool:
        """Check if processor supports the content type."""
        return content_type.lower() in ["equation", "formula", "math", "latex"]

    def process(
        self,
        content: Any,
        content_type: str,
        format: str = "latex",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process equation content.

        Args:
            content: Equation content (LaTeX string or math expression)
            content_type: Type of content (should be 'equation')
            format: Output format ('latex', 'mathml', 'text')
            **kwargs: Additional parameters

        Returns:
            Dictionary with processed equation data
        """
        result: Dict[str, Any] = {
            "type": "equation",
            "content": None,
            "format": format,
            "metadata": {},
        }

        try:
            if isinstance(content, str):
                result["content"] = content
                # Detect if it's LaTeX
                if content.startswith("$") or content.startswith("\\"):
                    result["format"] = "latex"
                    result["metadata"]["is_latex"] = True
            else:
                result["content"] = str(content)

        except Exception as e:
            logger.error(f"Error processing equation: {str(e)}")
            result["error"] = str(e)

        return result


class GenericProcessor(BaseModalProcessor):
    """Generic processor for custom content types."""

    def supports(self, content_type: str) -> bool:
        """Generic processor supports all content types."""
        return True

    def process(
        self,
        content: Any,
        content_type: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process generic content.

        Args:
            content: Content to process
            content_type: Type of content
            **kwargs: Additional parameters

        Returns:
            Dictionary with processed content data
        """
        return {
            "type": content_type,
            "content": str(content),
            "metadata": {
                "content_type": type(content).__name__,
            },
        }


class ModalProcessorRegistry:
    """Registry for managing modal processors."""

    def __init__(self):
        """Initialize processor registry."""
        self.processors: List[BaseModalProcessor] = [
            ImageProcessor(),
            TableProcessor(),
            EquationProcessor(),
            GenericProcessor(),  # Fallback processor
        ]

    def get_processor(self, content_type: str) -> BaseModalProcessor:
        """
        Get processor for content type.

        Args:
            content_type: Type of content

        Returns:
            Processor instance
        """
        for processor in self.processors:
            if processor.supports(content_type):
                return processor

        # Fallback to generic processor
        return GenericProcessor()

    def register_processor(self, processor: BaseModalProcessor, priority: int = 0):
        """
        Register a custom processor.

        Args:
            processor: Processor instance
            priority: Priority (higher = checked first)
        """
        # Insert at position based on priority
        insert_pos = len(self.processors) - 1  # Before GenericProcessor
        for i, p in enumerate(self.processors):
            if isinstance(p, GenericProcessor):
                insert_pos = i
                break

        self.processors.insert(insert_pos, processor)


# Global registry instance
_processor_registry = ModalProcessorRegistry()


def get_processor(content_type: str) -> BaseModalProcessor:
    """
    Get processor for content type from global registry.

    Args:
        content_type: Type of content

    Returns:
        Processor instance
    """
    return _processor_registry.get_processor(content_type)

