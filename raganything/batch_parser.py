"""
Batch parser coordination for RAG-Anything.

This module provides functionality for coordinating batch parsing
operations with parallel processing support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from raganything.parser import ParserFactory
from raganything.utils import validate_file_path

logger = logging.getLogger(__name__)


class BatchParser:
    """Batch parser coordinator."""

    def __init__(
        self,
        parser: Optional[str] = None,
        parse_method: str = "auto",
        max_workers: int = 4,
    ):
        """
        Initialize batch parser.

        Args:
            parser: Parser type ('mineru' or 'docling')
            parse_method: Parse method ('auto', 'ocr', 'txt')
            max_workers: Maximum number of parallel workers
        """
        self.parser_type = parser
        self.parse_method = parse_method
        self.max_workers = max_workers

    def parse_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        continue_on_error: bool = True,
        **parser_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Parse a batch of documents.

        Args:
            file_paths: List of file paths
            output_dir: Optional output directory
            continue_on_error: Whether to continue on errors
            **parser_kwargs: Additional parser parameters

        Returns:
            Dictionary with parsing results
        """
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {}
            for file_path in file_paths:
                parser = ParserFactory.create_parser(
                    parser_type=self.parser_type,
                    parse_method=self.parse_method,
                )
                future = executor.submit(
                    self._parse_single,
                    parser,
                    file_path,
                    output_dir,
                    **parser_kwargs
                )
                future_to_file[future] = file_path

            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    result["file_path"] = file_path
                    results.append(result)
                except Exception as e:
                    error_info = {
                        "file_path": file_path,
                        "error": str(e),
                    }
                    errors.append(error_info)
                    logger.error(f"Error parsing {file_path}: {str(e)}")

                    if not continue_on_error:
                        raise

        return {
            "total": len(file_paths),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }

    def _parse_single(
        self,
        parser: Any,
        file_path: str,
        output_dir: Optional[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Parse a single document (internal method)."""
        return parser.parse(file_path=file_path, output_dir=output_dir, **kwargs)

    def parse_directory(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = True,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Parse all files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern
            recursive: Whether to search recursively
            output_dir: Optional output directory
            **kwargs: Additional arguments

        Returns:
            Dictionary with parsing results
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            file_paths = list(dir_path.rglob(pattern))
        else:
            file_paths = list(dir_path.glob(pattern))

        file_paths = [str(f) for f in file_paths if f.is_file()]

        logger.info(f"Found {len(file_paths)} files in {directory}")

        return self.parse_batch(file_paths, output_dir=output_dir, **kwargs)

