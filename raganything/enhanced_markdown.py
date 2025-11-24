"""
Enhanced markdown generation for RAG-Anything.

This module provides functionality to generate structured markdown
with multimodal elements (images, tables, equations) and metadata.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnhancedMarkdownGenerator:
    """Generator for enhanced markdown with multimodal elements."""

    def __init__(self, include_metadata: bool = True):
        """
        Initialize markdown generator.

        Args:
            include_metadata: Whether to include metadata in markdown
        """
        self.include_metadata = include_metadata

    def generate(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        equations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate enhanced markdown from content and multimodal elements.

        Args:
            content: Main text content
            metadata: Document metadata
            images: List of image data dictionaries
            tables: List of table data dictionaries
            equations: List of equation data dictionaries

        Returns:
            Enhanced markdown string
        """
        lines: List[str] = []

        # Add metadata section if enabled
        if self.include_metadata and metadata:
            lines.append("---")
            lines.append("## Metadata")
            for key, value in metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("---")
            lines.append("")

        # Add main content
        if content:
            lines.append(content)
            lines.append("")

        # Add images
        if images:
            lines.append("## Images")
            for i, image in enumerate(images, 1):
                image_md = self._format_image(image, i)
                lines.append(image_md)
            lines.append("")

        # Add tables
        if tables:
            lines.append("## Tables")
            for i, table in enumerate(tables, 1):
                table_md = self._format_table(table, i)
                lines.append(table_md)
            lines.append("")

        # Add equations
        if equations:
            lines.append("## Equations")
            for i, equation in enumerate(equations, 1):
                equation_md = self._format_equation(equation, i)
                lines.append(equation_md)
            lines.append("")

        return "\n".join(lines)

    def _format_image(self, image: Dict[str, Any], index: int) -> str:
        """Format image in markdown."""
        lines = [f"### Image {index}"]

        # Image path or base64
        content = image.get("content", "")
        if content:
            if content.startswith("data:image") or len(content) > 100:
                # Base64 encoded image
                lines.append(f"![Image {index}](data:image/png;base64,{content})")
            else:
                # Image path
                lines.append(f"![Image {index}]({content})")

        # Image metadata
        if "metadata" in image:
            meta = image["metadata"]
            if "path" in meta:
                lines.append(f"*Path: {meta['path']}*")
            if "size" in meta:
                lines.append(f"*Size: {meta['size']} bytes*")

        return "\n".join(lines)

    def _format_table(self, table: Dict[str, Any], index: int) -> str:
        """Format table in markdown."""
        lines = [f"### Table {index}"]

        content = table.get("content", "")
        if content:
            lines.append(content)
        else:
            lines.append("*No table content*")

        # Table metadata
        if "metadata" in table:
            meta = table["metadata"]
            if "rows" in meta:
                lines.append(f"*Rows: {meta['rows']}*")

        return "\n".join(lines)

    def _format_equation(self, equation: Dict[str, Any], index: int) -> str:
        """Format equation in markdown."""
        lines = [f"### Equation {index}"]

        content = equation.get("content", "")
        format_type = equation.get("format", "latex")

        if content:
            if format_type == "latex":
                # LaTeX equation (inline and block)
                if not content.startswith("$"):
                    content = f"${content}$"
                lines.append(content)
            else:
                lines.append(f"```{format_type}")
                lines.append(content)
                lines.append("```")
        else:
            lines.append("*No equation content*")

        return "\n".join(lines)

    def add_image_reference(
        self,
        markdown: str,
        image_path: str,
        alt_text: Optional[str] = None,
        position: Optional[int] = None,
    ) -> str:
        """
        Add image reference to markdown.

        Args:
            markdown: Existing markdown content
            image_path: Path to image file
            alt_text: Alternative text for image
            position: Position to insert (None = append)

        Returns:
            Updated markdown with image reference
        """
        alt = alt_text or Path(image_path).stem
        image_ref = f"\n![{alt}]({image_path})\n"

        if position is None:
            return markdown + image_ref
        else:
            lines = markdown.split("\n")
            lines.insert(position, image_ref)
            return "\n".join(lines)

    def add_table(
        self,
        markdown: str,
        table_content: str,
        caption: Optional[str] = None,
        position: Optional[int] = None,
    ) -> str:
        """
        Add table to markdown.

        Args:
            markdown: Existing markdown content
            table_content: Table content (markdown format)
            caption: Optional table caption
            position: Position to insert (None = append)

        Returns:
            Updated markdown with table
        """
        table_md = "\n"
        if caption:
            table_md += f"**{caption}**\n\n"
        table_md += table_content + "\n"

        if position is None:
            return markdown + table_md
        else:
            lines = markdown.split("\n")
            lines.insert(position, table_md)
            return "\n".join(lines)

    def add_equation(
        self,
        markdown: str,
        equation: str,
        format: str = "latex",
        position: Optional[int] = None,
    ) -> str:
        """
        Add equation to markdown.

        Args:
            markdown: Existing markdown content
            equation: Equation content (LaTeX or other format)
            format: Equation format ('latex', 'mathml', etc.)
            position: Position to insert (None = append)

        Returns:
            Updated markdown with equation
        """
        if format == "latex":
            if not equation.startswith("$"):
                equation = f"${equation}$"
            equation_md = f"\n{equation}\n"
        else:
            equation_md = f"\n```{format}\n{equation}\n```\n"

        if position is None:
            return markdown + equation_md
        else:
            lines = markdown.split("\n")
            lines.insert(position, equation_md)
            return "\n".join(lines)


def generate_enhanced_markdown(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    tables: Optional[List[Dict[str, Any]]] = None,
    equations: Optional[List[Dict[str, Any]]] = None,
    include_metadata: bool = True,
) -> str:
    """
    Convenience function to generate enhanced markdown.

    Args:
        content: Main text content
        metadata: Document metadata
        images: List of image data
        tables: List of table data
        equations: List of equation data
        include_metadata: Whether to include metadata

    Returns:
        Enhanced markdown string
    """
    generator = EnhancedMarkdownGenerator(include_metadata=include_metadata)
    return generator.generate(
        content=content,
        metadata=metadata,
        images=images,
        tables=tables,
        equations=equations,
    )

