"""
Base classes and interfaces for RAG-Anything framework.

This module defines abstract base classes that provide the foundation
for extensible parsers, processors, and modal processors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Parse a document and return structured content.

        Args:
            file_path: Path to the document to parse
            output_dir: Optional output directory for parsed results
            **kwargs: Additional parser-specific parameters

        Returns:
            Dictionary containing parsed content and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement parse method")

    @abstractmethod
    def is_supported(self, file_path: str) -> bool:
        """
        Check if the parser supports the given file type.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file type is supported, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement is_supported method")


class BaseProcessor(ABC):
    """Abstract base class for content processors."""

    @abstractmethod
    def process(
        self,
        content: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process content and return processed results.

        Args:
            content: Content to process
            **kwargs: Additional processing parameters

        Returns:
            Dictionary containing processed content and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process method")


class BaseModalProcessor(ABC):
    """Abstract base class for multimodal content processors."""

    @abstractmethod
    def process(
        self,
        content: Any,
        content_type: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process multimodal content based on content type.

        Args:
            content: Content to process
            content_type: Type of content (e.g., 'image', 'table', 'equation')
            **kwargs: Additional processing parameters

        Returns:
            Dictionary containing processed content and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process method")

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """
        Check if the processor supports the given content type.

        Args:
            content_type: Type of content to check

        Returns:
            True if the content type is supported, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement supports method")


class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document

        Returns:
            List of document IDs

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement add_documents method")

    @abstractmethod
    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            query_embedding: Optional pre-computed query embedding
            n_results: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results with documents and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement search method")

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """
        Delete documents from the vector store.

        Args:
            ids: Optional list of document IDs to delete
            **kwargs: Additional deletion parameters

        Returns:
            True if deletion was successful

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement delete method")

