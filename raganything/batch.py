"""
Batch processing utilities for RAG-Anything.

This module provides functionality for processing multiple documents
efficiently with progress tracking and error handling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from raganything.raganything import RAGAnything
from raganything.utils import validate_file_path

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch document processor."""

    def __init__(self, raganything: Optional[RAGAnything] = None, **kwargs: Any):
        """
        Initialize batch processor.

        Args:
            raganything: RAGAnything instance (creates new if None)
            **kwargs: Arguments to pass to RAGAnything constructor
        """
        self.raganything = raganything or RAGAnything(**kwargs)
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

    def process_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        display_progress: bool = True,
        continue_on_error: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process a batch of documents.

        Args:
            file_paths: List of file paths to process
            output_dir: Optional output directory
            display_progress: Whether to display progress
            continue_on_error: Whether to continue processing on errors
            **kwargs: Additional arguments for process_document_complete

        Returns:
            Dictionary with batch processing results
        """
        self.results = []
        self.errors = []

        total = len(file_paths)
        logger.info(f"Starting batch processing of {total} documents")

        for i, file_path in enumerate(file_paths, 1):
            if display_progress:
                logger.info(f"Processing {i}/{total}: {file_path}")

            try:
                result = self.raganything.process_document_complete(
                    file_path=file_path,
                    output_dir=output_dir,
                    **kwargs
                )
                result["batch_index"] = i
                result["batch_total"] = total
                self.results.append(result)

            except Exception as e:
                error_info = {
                    "file_path": file_path,
                    "error": str(e),
                    "batch_index": i,
                    "batch_total": total,
                }
                self.errors.append(error_info)
                logger.error(f"Error processing {file_path}: {str(e)}")

                if not continue_on_error:
                    raise

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get batch processing summary.

        Returns:
            Dictionary with summary statistics
        """
        total_processed = len(self.results)
        total_errors = len(self.errors)
        total_chunks = sum(r.get("num_chunks", 0) for r in self.results)

        return {
            "total_files": total_processed + total_errors,
            "successful": total_processed,
            "failed": total_errors,
            "total_chunks": total_chunks,
            "results": self.results,
            "errors": self.errors,
        }

    def process_directory(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = True,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process all files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.pdf")
            recursive: Whether to search recursively
            output_dir: Optional output directory
            **kwargs: Additional arguments for process_batch

        Returns:
            Dictionary with batch processing results
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            file_paths = list(dir_path.rglob(pattern))
        else:
            file_paths = list(dir_path.glob(pattern))

        # Filter to only files
        file_paths = [str(f) for f in file_paths if f.is_file()]

        logger.info(f"Found {len(file_paths)} files in {directory}")

        return self.process_batch(file_paths, output_dir=output_dir, **kwargs)


def process_batch(
    file_paths: List[str],
    output_dir: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Convenience function for batch processing.

    Args:
        file_paths: List of file paths
        output_dir: Optional output directory
        **kwargs: Additional arguments

    Returns:
        Batch processing results
    """
    processor = BatchProcessor(**kwargs)
    return processor.process_batch(file_paths, output_dir=output_dir)

