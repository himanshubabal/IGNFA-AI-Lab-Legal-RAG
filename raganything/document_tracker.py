"""
Document tracking system for AI Lab IGNFA - Legal RAG System.

Tracks which documents have been processed, their metadata,
and manages document lifecycle (add, update, remove).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentTracker:
    """Tracks processed documents and their metadata."""

    def __init__(self, tracker_file: Optional[Path] = None):
        """
        Initialize document tracker.

        Args:
            tracker_file: Path to tracker JSON file (default: output/.document_tracker.json)
        """
        if tracker_file is None:
            from raganything.config import get_config
            config = get_config()
            tracker_file = config.output_dir / ".document_tracker.json"

        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.processed_docs: Dict[str, Dict] = self._load_tracker()

    def _load_tracker(self) -> Dict[str, Dict]:
        """Load tracker data from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("processed_docs", {})
            except Exception as e:
                logger.warning(f"Error loading tracker file: {e}")
                return {}
        return {}

    def _save_tracker(self):
        """Save tracker data to file."""
        try:
            # Ensure parent directory exists
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "processed_docs": self.processed_docs,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.tracker_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Tracker file saved: {self.tracker_file}")
        except Exception as e:
            logger.error(f"Error saving tracker file: {e}")
            raise

    def get_processed_docs(self) -> Dict[str, Dict]:
        """Get all processed documents."""
        return self.processed_docs.copy()

    def is_processed(self, file_path: str) -> bool:
        """Check if a document has been processed."""
        normalized_path = str(Path(file_path).resolve())
        return normalized_path in self.processed_docs

    def get_document_info(self, file_path: str) -> Optional[Dict]:
        """Get information about a processed document."""
        normalized_path = str(Path(file_path).resolve())
        return self.processed_docs.get(normalized_path)

    def mark_processed(
        self,
        file_path: str,
        doc_id: str,
        num_chunks: int,
        metadata: Optional[Dict] = None,
    ):
        """
        Mark a document as processed.

        Args:
            file_path: Path to the document
            doc_id: Document ID
            num_chunks: Number of chunks created
            metadata: Additional metadata
        """
        normalized_path = str(Path(file_path).resolve())
        file_stat = Path(file_path).stat()

        self.processed_docs[normalized_path] = {
            "file_path": normalized_path,
            "doc_id": doc_id,
            "num_chunks": num_chunks,
            "file_size": file_stat.st_size,
            "modified_time": file_stat.st_mtime,
            "processed_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_tracker()

    def mark_removed(self, file_path: str):
        """Mark a document as removed (for cleanup)."""
        normalized_path = str(Path(file_path).resolve())
        if normalized_path in self.processed_docs:
            del self.processed_docs[normalized_path]
            self._save_tracker()
            return True
        return False

    def get_new_documents(self, document_paths: List[str]) -> List[str]:
        """
        Get list of documents that haven't been processed yet.

        Args:
            document_paths: List of document file paths

        Returns:
            List of new document paths
        """
        new_docs = []
        for doc_path in document_paths:
            normalized_path = str(Path(doc_path).resolve())
            if normalized_path not in self.processed_docs:
                new_docs.append(doc_path)
        return new_docs

    def get_removed_documents(self, current_paths: Set[str]) -> List[str]:
        """
        Get list of documents that were processed but no longer exist.

        Args:
            current_paths: Set of current document paths (normalized)

        Returns:
            List of removed document paths
        """
        removed = []
        for processed_path in self.processed_docs.keys():
            if processed_path not in current_paths:
                removed.append(processed_path)
        return removed

    def get_updated_documents(self, document_paths: List[str]) -> List[str]:
        """
        Get list of documents that have been modified since processing.

        Args:
            document_paths: List of document file paths

        Returns:
            List of updated document paths
        """
        updated = []
        for doc_path in document_paths:
            normalized_path = str(Path(doc_path).resolve())
            if normalized_path in self.processed_docs:
                doc_info = self.processed_docs[normalized_path]
                file_stat = Path(doc_path).stat()
                if file_stat.st_mtime > doc_info.get("modified_time", 0):
                    updated.append(doc_path)
        return updated

    def get_all_document_paths(self) -> List[str]:
        """Get all tracked document paths."""
        return list(self.processed_docs.keys())

    def clear(self):
        """Clear all tracked documents and ensure tracker file exists."""
        self.processed_docs = {}
        self._save_tracker()
        # Verify file was created
        if not self.tracker_file.exists():
            logger.warning(f"Tracker file not found after clear, attempting to recreate: {self.tracker_file}")
            self._save_tracker()

