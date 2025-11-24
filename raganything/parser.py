"""
Document parser abstraction layer for RAG-Anything.

This module provides a unified interface for different document parsers
(MinerU, Docling) and handles parser selection and output standardization.
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from raganything.base import BaseParser
from raganything.config import get_config
from raganything.utils import validate_file_path, ensure_directory

logger = logging.getLogger(__name__)


class MinerUParser(BaseParser):
    """Parser wrapper for MinerU document parser."""

    def __init__(self, parse_method: str = "auto"):
        """
        Initialize MinerU parser.

        Args:
            parse_method: Parse method ('auto', 'ocr', 'txt')
        """
        self.parse_method = parse_method
        self.supported_formats = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
        }

    def is_supported(self, file_path: str) -> bool:
        """Check if MinerU supports the file type."""
        from raganything.utils import detect_file_type

        extension, _ = detect_file_type(file_path)
        return extension.lower() in self.supported_formats

    def parse(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        device: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        backend: str = "pipeline",
        source: str = "huggingface",
        vlm_url: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Parse document using MinerU.

        Args:
            file_path: Path to document
            output_dir: Output directory for parsed results
            lang: Document language for OCR (e.g., 'ch', 'en', 'ja')
            device: Inference device ('cpu', 'cuda', 'cuda:0', 'npu', 'mps')
            start_page: Starting page number (0-based, for PDF)
            end_page: Ending page number (0-based, for PDF)
            formula: Enable formula parsing
            table: Enable table parsing
            backend: Parsing backend ('pipeline', 'vlm-transformers', etc.)
            source: Model source ('huggingface', 'modelscope', 'local')
            vlm_url: Service address when using backend='vlm-sglang-client'
            **kwargs: Additional parameters

        Returns:
            Dictionary with parsed content and metadata
        """
        file_path = validate_file_path(file_path)
        config = get_config()

        # Determine output directory
        if output_dir is None:
            output_dir = config.output_dir
        else:
            output_dir = ensure_directory(Path(output_dir))

        # Build MinerU command (magic-pdf is the CLI command)
        # Try 'magic-pdf' first, fallback to 'mineru' for compatibility
        cmd_base = "magic-pdf"
        is_magic_pdf = True
        try:
            subprocess.run([cmd_base, "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            cmd_base = "mineru"  # Fallback to mineru if magic-pdf not found
            is_magic_pdf = False
        
        # Build base command
        if is_magic_pdf:
            # magic-pdf uses: -p (path), -o (output-dir), -m (method)
            cmd = [cmd_base, "-p", str(file_path), "-o", str(output_dir), "-m", self.parse_method]
            
            # Add optional parameters (magic-pdf specific)
            if lang:
                cmd.extend(["-l", lang])  # magic-pdf uses -l for lang
            if start_page is not None:
                cmd.extend(["-s", str(start_page)])  # magic-pdf uses -s for start
            if end_page is not None:
                cmd.extend(["-e", str(end_page)])  # magic-pdf uses -e for end
            # Note: magic-pdf doesn't support: device, formula, table, backend, source, vlm_url
        else:
            # Original mineru command format (for compatibility)
            cmd = [cmd_base, "-p", str(file_path), "-o", str(output_dir), "-m", self.parse_method]
            
            # Add optional parameters (mineru format)
            if lang:
                cmd.extend(["--lang", lang])
            if device:
                cmd.extend(["--device", device])
            if start_page is not None:
                cmd.extend(["--start-page", str(start_page)])
            if end_page is not None:
                cmd.extend(["--end-page", str(end_page)])
            if not formula:
                cmd.append("--no-formula")
            if not table:
                cmd.append("--no-table")
            if backend:
                cmd.extend(["-b", backend])
            if source:
                cmd.extend(["--source", source])
            if vlm_url:
                cmd.extend(["--vlm-url", vlm_url])

        logger.info(f"Running MinerU parser: {' '.join(cmd)}")

        try:
            # Execute MinerU command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit, we'll check manually
                timeout=3600,  # 1 hour timeout
            )
            
            # Check if command failed
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"MinerU command failed with exit code {result.returncode}")
                logger.error(f"Error output: {error_msg[:500]}")
                
                # Check if it's a config file issue
                if "magic-pdf.json" in error_msg or "config" in error_msg.lower():
                    logger.error(
                        "magic-pdf requires a config file at ~/magic-pdf.json. "
                        "Please create one or check magic-pdf documentation."
                    )
                
                raise RuntimeError(
                    f"MinerU parsing failed (exit code {result.returncode}): {error_msg[:200]}"
                )

            # Parse output - magic-pdf creates files in subdirectories
            # Look for markdown files in output directory and subdirectories
            output_files = list(output_dir.rglob("*.md"))
            
            # Filter out README.md files (these are not the parsed content)
            output_files = [f for f in output_files if f.name.upper() != "README.MD"]
            
            # Also check for files in the output directory itself (excluding README)
            if not output_files:
                output_files = [f for f in output_dir.glob("*.md") if f.name.upper() != "README.MD"]
            
            # magic-pdf might create files with the document name as subdirectory
            # Try looking in a subdirectory matching the file stem
            if not output_files:
                file_stem = Path(file_path).stem
                potential_dirs = [
                    output_dir / file_stem,
                    output_dir / Path(file_path).name.replace('.', '_'),
                ]
                for potential_dir in potential_dirs:
                    if potential_dir.exists():
                        found_files = [f for f in potential_dir.rglob("*.md") if f.name.upper() != "README.MD"]
                        if found_files:
                            output_files = found_files
                            break
            
            # Log search results for debugging
            logger.debug(f"Searching for markdown files in: {output_dir}")
            logger.debug(f"Found {len(output_files)} markdown file(s) (excluding README)")
            
            # If still no files, list what was actually created
            if not output_files:
                all_files = list(output_dir.rglob("*"))
                file_list = [f.name for f in all_files[:10] if f.is_file()]
                logger.warning(
                    f"No markdown output files found from MinerU. "
                    f"Output directory contains: {file_list}"
                )

            if output_files:
                # Read the first markdown file found
                markdown_file = output_files[0]
                with open(markdown_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Log content extraction
                logger.info(
                    f"MinerU extracted {len(content)} characters from {Path(file_path).name}"
                )
                if len(content) < 100:
                    logger.warning(
                        f"Very short content extracted ({len(content)} chars). "
                        f"Check if parsing was successful."
                    )

                return {
                    "content": content,
                    "format": "markdown",
                    "output_file": str(markdown_file),
                    "parser": "mineru",
                    "metadata": {
                        "file_path": str(file_path),
                        "parse_method": self.parse_method,
                        "lang": lang,
                        "device": device,
                    },
                }
            else:
                logger.warning("No markdown output files found from MinerU")
                return {
                    "content": "",
                    "format": "markdown",
                    "output_file": None,
                    "parser": "mineru",
                    "metadata": {
                        "file_path": str(file_path),
                        "parse_method": self.parse_method,
                        "error": "No output files generated",
                    },
                }

        except subprocess.CalledProcessError as e:
            logger.error(f"MinerU parsing failed: {e.stderr}")
            raise RuntimeError(f"MinerU parsing failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            logger.error("MinerU parsing timed out")
            raise RuntimeError("MinerU parsing timed out after 1 hour")
        except FileNotFoundError:
            error_msg = (
                "MinerU not found. Please install MinerU:\n"
                "  pip install magic-pdf\n"
                "  Or visit: https://github.com/HKUDS/MinerU\n\n"
                "Alternatively, you can use Docling parser by setting PARSER=docling in .env"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from None


class DoclingParser(BaseParser):
    """Parser wrapper for Docling document parser."""

    def __init__(self):
        """Initialize Docling parser."""
        self.supported_formats = {
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".html",
            ".htm",
        }

    def is_supported(self, file_path: str) -> bool:
        """Check if Docling supports the file type."""
        from raganything.utils import detect_file_type

        extension, _ = detect_file_type(file_path)
        return extension.lower() in self.supported_formats

    def parse(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Parse document using Docling.

        Args:
            file_path: Path to document
            output_dir: Output directory for parsed results
            **kwargs: Additional parameters (ignored for Docling)

        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
        except ImportError:
            raise RuntimeError(
                "Docling not installed. Please install: pip install docling"
            ) from None

        file_path = validate_file_path(file_path)
        config = get_config()

        # Determine output directory
        if output_dir is None:
            output_dir = config.output_dir
        else:
            output_dir = ensure_directory(Path(output_dir))

        logger.info(f"Running Docling parser on: {file_path}")

        try:
            # Initialize Docling converter
            converter = DocumentConverter()

            # Convert document
            result = converter.convert(str(file_path))

            # Get markdown content
            content = result.document.export_to_markdown()

            # Save to output directory
            output_file = output_dir / f"{file_path.stem}_docling.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "content": content,
                "format": "markdown",
                "output_file": str(output_file),
                "parser": "docling",
                "metadata": {
                    "file_path": str(file_path),
                },
            }

        except Exception as e:
            logger.error(f"Docling parsing failed: {str(e)}")
            raise RuntimeError(f"Docling parsing failed: {str(e)}") from e


class ParserFactory:
    """Factory for creating parser instances."""

    @staticmethod
    def check_mineru_available() -> bool:
        """Check if MinerU is available in the system."""
        # First, try to check if magic-pdf Python package is installed
        try:
            import magic_pdf
            # If we can import it, it's available
            return True
        except ImportError:
            pass
        
        # Also check for CLI command (mineru or magic-pdf)
        try:
            import subprocess
            # Try 'mineru' command
            result = subprocess.run(
                ["mineru", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try 'magic-pdf' command
        try:
            import subprocess
            result = subprocess.run(
                ["magic-pdf", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try python -m magic_pdf
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "magic_pdf", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return False

    @staticmethod
    def create_parser(parser_type: Optional[str] = None, parse_method: str = "auto") -> BaseParser:
        """
        Create a parser instance.

        Args:
            parser_type: Parser type ('mineru' or 'docling'). If None, uses config.
            parse_method: Parse method for MinerU ('auto', 'ocr', 'txt')

        Returns:
            Parser instance

        Raises:
            ValueError: If parser type is invalid
        """
        if parser_type is None:
            config = get_config()
            parser_type = config.parser

        parser_type = parser_type.lower()

        if parser_type == "mineru":
            # Check if MinerU is available
            if not ParserFactory.check_mineru_available():
                logger.warning(
                    "MinerU not found. Falling back to Docling parser. "
                    "Install MinerU with: pip install magic-pdf"
                )
                # Fall back to Docling if MinerU is not available
                return DoclingParser()
            return MinerUParser(parse_method=parse_method)
        elif parser_type == "docling":
            return DoclingParser(parse_method=parse_method)
        else:
            raise ValueError(f"Unknown parser type: {parser_type}. Use 'mineru' or 'docling'")

    @staticmethod
    def get_parser_for_file(file_path: str, parser_type: Optional[str] = None) -> BaseParser:
        """
        Get appropriate parser for a file.

        Args:
            file_path: Path to file
            parser_type: Preferred parser type (optional)

        Returns:
            Parser instance that supports the file type
        """
        if parser_type:
            parser = ParserFactory.create_parser(parser_type)
            if parser.is_supported(file_path):
                return parser
            else:
                logger.warning(
                    f"Parser {parser_type} does not support {file_path}, "
                    "trying alternative parser"
                )

        # Try MinerU first (supports more formats)
        mineru_parser = MinerUParser()
        if mineru_parser.is_supported(file_path):
            return mineru_parser

        # Try Docling
        docling_parser = DoclingParser()
        if docling_parser.is_supported(file_path):
            return docling_parser

        raise ValueError(f"No parser found that supports file: {file_path}")

