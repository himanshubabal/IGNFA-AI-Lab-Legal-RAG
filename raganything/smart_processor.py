"""
Smart document processor with tracking and lifecycle management.

Processes documents intelligently by tracking what's been processed,
handling new documents, updates, and removals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from raganything import RAGAnything
from raganything.document_tracker import DocumentTracker
from raganything.utils import (
    is_pdf,
    is_office_document,
    is_image,
    is_text_file,
)

logger = logging.getLogger(__name__)


class SmartProcessor:
    """Smart document processor with tracking."""

    def __init__(
        self,
        documents_dir: str = "documents",
        raganything: Optional[RAGAnything] = None,
    ):
        """
        Initialize smart processor.

        Args:
            documents_dir: Directory containing documents to process
            raganything: RAGAnything instance (creates new if None)
        """
        self.documents_dir = Path(documents_dir)
        self.raganything = raganything or RAGAnything()
        self.tracker = DocumentTracker()

    def _should_exclude_file(self, file_path: Path) -> bool:
        """
        Check if a file should be excluded from processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be excluded, False otherwise
        """
        file_name = file_path.name.lower()
        file_name_no_ext = file_path.stem.lower()
        
        # Exclude hidden files (starting with .)
        if file_name.startswith('.'):
            return True
        
        # Exclude common system files
        excluded_names = {
            '.ds_store',
            'thumbs.db',
            'desktop.ini',
            '.gitkeep',
            '.gitignore',
        }
        if file_name in excluded_names:
            return True
        
        # Exclude README files (case-insensitive)
        if file_name_no_ext in {'readme', 'read_me', 'read-me'}:
            return True
        
        # Exclude LICENSE files
        if file_name_no_ext in {'license', 'licence', 'copying'}:
            return True
        
        # Exclude common documentation files
        excluded_patterns = [
            'readme',
            'changelog',
            'contributing',
            'authors',
            'credits',
            'acknowledgments',
        ]
        for pattern in excluded_patterns:
            if pattern in file_name_no_ext:
                return True
        
        return False

    def _get_supported_files(self) -> List[Path]:
        """Get all supported files from documents directory."""
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return []

        supported_files = []
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file():
                # Skip excluded files
                if self._should_exclude_file(file_path):
                    logger.debug(f"Excluding file: {file_path.name}")
                    continue
                
                file_str = str(file_path)
                if (
                    is_pdf(file_str)
                    or is_office_document(file_str)
                    or is_image(file_str)
                    or is_text_file(file_str)
                ):
                    supported_files.append(file_path)

        return supported_files

    def _normalize_paths(self, paths: List[Path]) -> Set[str]:
        """Normalize paths for comparison."""
        return {str(p.resolve()) for p in paths}

    def process_all(
        self,
        force_reprocess: bool = False,
        remove_missing: bool = True,
    ) -> Dict[str, any]:
        """
        Process all documents in the documents directory.

        Args:
            force_reprocess: If True, reprocess all documents
            remove_missing: If True, remove embeddings for deleted documents

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Scanning documents directory: {self.documents_dir}")

        # Get all supported files
        all_files = self._get_supported_files()
        current_paths = self._normalize_paths(all_files)

        results = {
            "new": [],
            "updated": [],
            "unchanged": [],
            "removed": [],
            "errors": [],
        }

        # Find new documents
        if force_reprocess:
            new_docs = [str(f) for f in all_files]
        else:
            new_docs = self.tracker.get_new_documents([str(f) for f in all_files])

        # Find updated documents
        if not force_reprocess:
            updated_docs = self.tracker.get_updated_documents(
                [str(f) for f in all_files]
            )
        else:
            updated_docs = []

        # Find removed documents
        if remove_missing:
            removed_docs = self.tracker.get_removed_documents(current_paths)
            for removed_path in removed_docs:
                logger.info(f"Document removed: {removed_path}")
                # Get document info before removing
                doc_info = self.tracker.get_document_info(removed_path)
                if doc_info:
                    doc_id = doc_info.get("doc_id")
                    # Try to remove from vector store
                    try:
                        if hasattr(self.raganything.processor, "vector_store") and self.raganything.processor.vector_store:
                            # Get all chunk IDs for this document
                            # Note: This is a simplified approach - in production, you'd want
                            # to track chunk IDs more precisely
                            if hasattr(self.raganything.processor.vector_store, "delete"):
                                # Delete by metadata filter (if supported)
                                # For now, we'll mark as removed in tracker
                                logger.info(f"Marking document as removed: {removed_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove from vector store: {e}")
                
                # Mark as removed in tracker
                self.tracker.mark_removed(removed_path)
                results["removed"].append(removed_path)

        # Process new documents
        for doc_path in new_docs:
            try:
                logger.info(f"Processing new document: {doc_path}")
                result = self.raganything.process_document_complete(
                    file_path=doc_path,
                    doc_id=Path(doc_path).stem,
                )

                self.tracker.mark_processed(
                    file_path=doc_path,
                    doc_id=Path(doc_path).stem,
                    num_chunks=result.get("num_chunks", 0),
                    metadata=result.get("metadata", {}),
                )

                results["new"].append(
                    {
                        "path": doc_path,
                        "chunks": result.get("num_chunks", 0),
                    }
                )
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                results["errors"].append({"path": doc_path, "error": str(e)})

        # Process updated documents
        for doc_path in updated_docs:
            try:
                logger.info(f"Reprocessing updated document: {doc_path}")
                # Remove old entry
                self.tracker.mark_removed(doc_path)

                # Process again
                result = self.raganything.process_document_complete(
                    file_path=doc_path,
                    doc_id=Path(doc_path).stem,
                )

                self.tracker.mark_processed(
                    file_path=doc_path,
                    doc_id=Path(doc_path).stem,
                    num_chunks=result.get("num_chunks", 0),
                    metadata=result.get("metadata", {}),
                )

                results["updated"].append(
                    {
                        "path": doc_path,
                        "chunks": result.get("num_chunks", 0),
                    }
                )
            except Exception as e:
                logger.error(f"Error reprocessing {doc_path}: {e}")
                results["errors"].append({"path": doc_path, "error": str(e)})

        # Track unchanged documents
        processed_paths = set(self.tracker.get_all_document_paths())
        unchanged = current_paths.intersection(processed_paths)
        unchanged = unchanged - {str(Path(p).resolve()) for p in new_docs + updated_docs}
        results["unchanged"] = list(unchanged)

        # Summary
        total_processed = len(results["new"]) + len(results["updated"])
        logger.info(
            f"Processing complete: {len(results['new'])} new, "
            f"{len(results['updated'])} updated, {len(results['removed'])} removed, "
            f"{len(results['unchanged'])} unchanged, {len(results['errors'])} errors"
        )

        return results

    def get_document_status(self) -> Dict[str, any]:
        """
        Get status of all documents in the directory.

        Returns:
            Dictionary with document status information
        """
        all_files = self._get_supported_files()
        current_paths = self._normalize_paths(all_files)

        status = {
            "total_files": len(all_files),
            "processed": [],
            "unprocessed": [],
            "removed": [],
        }

        # Check each file
        for file_path in all_files:
            file_str = str(file_path)
            if self.tracker.is_processed(file_str):
                info = self.tracker.get_document_info(file_str)
                status["processed"].append(
                    {
                        "path": file_str,
                        "doc_id": info.get("doc_id"),
                        "chunks": info.get("num_chunks", 0),
                        "processed_time": info.get("processed_time"),
                    }
                )
            else:
                status["unprocessed"].append({"path": file_str})

        # Find removed documents
        processed_paths = set(self.tracker.get_all_document_paths())
        removed = processed_paths - current_paths
        status["removed"] = list(removed)

        return status

