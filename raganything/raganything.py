"""
Main RAGAnything class - orchestrator for RAG-Anything framework.

This module provides the main interface for document processing,
content management, and querying.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from raganything.config import get_config
from raganything.parser import ParserFactory
from raganything.processor import ContentProcessor
from raganything.query import RAGQuery
from raganything.utils import validate_file_path, ensure_directory

logger = logging.getLogger(__name__)


class RAGAnything:
    """Main RAG-Anything orchestrator class."""

    def __init__(
        self,
        parser: Optional[str] = None,
        parse_method: str = "auto",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "character",
        vector_store_persist_dir: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_top_p: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
    ):
        """
        Initialize RAGAnything instance.

        Args:
            parser: Parser type ('mineru' or 'docling'), uses config if None
            parse_method: Parse method ('auto', 'ocr', 'txt')
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap
            chunk_strategy: Chunking strategy ('character', 'sentence', 'paragraph')
            vector_store_persist_dir: Optional directory for vector store persistence
        """
        self.config = get_config()

        # Initialize parser
        self.parser = ParserFactory.create_parser(
            parser_type=parser or self.config.parser,
            parse_method=parse_method or self.config.parse_method,
        )

        # Initialize processor
        if vector_store_persist_dir:
            from raganything.processor import ChromaVectorStore
            vector_store = ChromaVectorStore(persist_directory=vector_store_persist_dir)
        else:
            vector_store = None

        self.processor = ContentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            vector_store=vector_store,
        )

        # Initialize query handler with LLM configuration
        self.query_handler = RAGQuery(
            vector_store=self.processor.vector_store,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            model=llm_model or self.config.llm_model,
            temperature=llm_temperature if llm_temperature is not None else self.config.llm_temperature,
            top_p=llm_top_p if llm_top_p is not None else self.config.llm_top_p,
            max_tokens=llm_max_tokens if llm_max_tokens is not None else self.config.llm_max_tokens,
        )
        
        # Set custom prompt file if configured
        if self.config.prompt_file_path and self.config.prompt_file_path.exists():
            self.query_handler.custom_prompt_file = str(self.config.prompt_file_path)
        
        # Store config for query method
        self.default_n_results = self.config.query_n_results
        self.default_max_context_length = self.config.query_max_context_length

    def process_document(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        doc_id: Optional[str] = None,
        **parser_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process a document (parse only, no vectorization).

        Args:
            file_path: Path to document
            output_dir: Optional output directory
            doc_id: Optional document ID
            **parser_kwargs: Additional parser parameters

        Returns:
            Dictionary with parsed content and metadata
        """
        file_path = validate_file_path(file_path)

        if output_dir:
            output_dir = str(ensure_directory(Path(output_dir)))
        else:
            output_dir = str(self.config.output_dir)

        logger.info(f"Processing document: {file_path}")

        # Parse document
        result = self.parser.parse(file_path=file_path, output_dir=output_dir, **parser_kwargs)

        # Add document ID if provided
        if doc_id:
            result["metadata"]["doc_id"] = doc_id

        return result

    def process_document_complete(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        doc_id: Optional[str] = None,
        display_stats: bool = True,
        split_by_character: Optional[str] = None,
        **parser_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline (parse + vectorize).

        Args:
            file_path: Path to document
            output_dir: Optional output directory
            doc_id: Optional document ID
            display_stats: Whether to display content statistics
            split_by_character: Optional character to split text by
            **parser_kwargs: Additional parser parameters (lang, device, etc.)

        Returns:
            Dictionary with processing results
        """
        # Parse document
        parse_result = self.process_document(
            file_path=file_path,
            output_dir=output_dir,
            doc_id=doc_id,
            **parser_kwargs
        )

        content = parse_result.get("content", "")
        if not content:
            logger.warning("No content extracted from document")
            return parse_result

        # Split content if requested
        if split_by_character:
            content = content.replace(split_by_character, "\n\n")

        # Process and store in vector database
        metadata = parse_result.get("metadata", {})
        metadata["file_path"] = file_path
        metadata["parser"] = parse_result.get("parser", "unknown")

        chunk_ids = self.processor.process_and_store(
            content=content,
            doc_id=doc_id or Path(file_path).stem,
            metadata=metadata,
        )

        # Get statistics
        stats = self.processor.get_statistics(content)

        result = {
            **parse_result,
            "chunk_ids": chunk_ids,
            "num_chunks": len(chunk_ids),
            "statistics": stats,
        }

        if display_stats:
            logger.info(f"Document processed: {len(chunk_ids)} chunks created")
            logger.info(f"Statistics: {stats}")

        return result

    def query(
        self,
        query: str,
        n_results: Optional[int] = None,
        max_context_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Query the knowledge base.

        Args:
            query: Query string
            n_results: Number of results to retrieve (uses config default if None)
            max_context_length: Maximum context length (uses config default if None)
            temperature: LLM temperature (uses config default if None)
            top_p: LLM top_p (uses config default if None)
            max_tokens: LLM max_tokens (uses config default if None)
            **kwargs: Additional query parameters

        Returns:
            Dictionary with answer and metadata
        """
        return self.query_handler.query(
            query=query,
            n_results=n_results or self.default_n_results,
            max_context_length=max_context_length or self.default_max_context_length,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

    def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> List[str]:
        """
        Insert pre-parsed content into knowledge base.

        Args:
            content_list: List of content dictionaries with 'content' and optional 'metadata'
            doc_id: Optional document ID

        Returns:
            List of chunk IDs
        """
        all_chunk_ids = []

        for i, item in enumerate(content_list):
            content = item.get("content", "")
            if not content:
                continue

            metadata = item.get("metadata", {})
            if doc_id:
                metadata["doc_id"] = doc_id

            chunk_ids = self.processor.process_and_store(
                content=content,
                doc_id=doc_id or f"content_{i}",
                metadata=metadata,
            )

            all_chunk_ids.extend(chunk_ids)

        logger.info(f"Inserted {len(all_chunk_ids)} chunks from content list")
        return all_chunk_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents without generating answer.

        Args:
            query: Search query
            n_results: Number of results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        return self.query_handler.search(
            query=query,
            n_results=n_results,
            min_score=min_score,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics
        """
        # This is a placeholder - actual implementation would query vector store
        return {
            "vector_store_type": type(self.processor.vector_store).__name__,
            "chunk_size": self.processor.chunk_size,
            "chunk_strategy": self.processor.chunk_strategy,
        }

