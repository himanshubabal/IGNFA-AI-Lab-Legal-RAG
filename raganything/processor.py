"""
Content processor for AI Lab IGNFA - Legal RAG System.

This module handles text chunking, vector embedding generation,
and vector store integration.
"""

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from raganything.base import BaseVectorStore
from raganything.config import get_config

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunking strategies."""

    @staticmethod
    def chunk_by_character(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Chunk text by character count.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        return chunks

    @staticmethod
    def chunk_by_sentence(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Chunk text by sentences.

        Args:
            text: Text to chunk
            chunk_size: Approximate size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = text.replace("。", ".").replace("！", "!").replace("？", "?").split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)
            chunk_text = " ".join(current_chunk)

            if len(chunk_text) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                current_chunk = overlap_sentences

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    @staticmethod
    def chunk_by_paragraph(
        text: str,
        chunk_size: int = 1000,
    ) -> List[str]:
        """
        Chunk text by paragraphs.

        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk

        Returns:
            List of text chunks
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            current_chunk.append(para)
            chunk_text = "\n\n".join(current_chunk)

            if len(chunk_text) >= chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize embedding generator.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                client_kwargs = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                # Only set base_url if it's a non-empty string
                if self.base_url and self.base_url.strip():
                    client_kwargs["base_url"] = self.base_url.strip()
                # If no base_url specified, OpenAI client will use default: https://api.openai.com/v1

                self._client = OpenAI(**client_kwargs)
            except ImportError:
                raise RuntimeError("OpenAI library not installed. Install with: pip install openai")

        return self._client

    def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model name

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        import sys
        num_texts = len(texts)
        print(f"🔢 Generating embeddings for {num_texts} chunks (model: {model})...", flush=True)
        logger.info(f"🔢 Generating embeddings for {num_texts} chunks using {model}")

        client = self._get_client()
        
        # Check if client is available
        if client is None:
            error_msg = (
                "OpenAI client not initialized. Please check your API configuration:\n"
                "- Ensure OPENAI_API_KEY is set in environment variables or Streamlit secrets\n"
                "- Verify your API key is valid\n"
                "- Check network connectivity"
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg)
        
        # Check if API key is set (try to get from client if not in self.api_key)
        api_key = self.api_key
        if not api_key and hasattr(client, '_client') and hasattr(client._client, 'api_key'):
            api_key = client._client.api_key
        if not api_key:
            # Try to get from environment as fallback
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            error_msg = (
                "OpenAI API key not found. Please configure it:\n"
                "- Set OPENAI_API_KEY in environment variables\n"
                "- Or configure it in Streamlit secrets (.streamlit/secrets.toml)\n"
                "- Or set it in .env file\n\n"
                "For Streamlit Cloud: Go to Settings → Secrets and add:\n"
                "OPENAI_API_KEY = \"your-api-key-here\""
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg)
        
        # Log API key status (masked for security)
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        logger.debug(f"Using API key: {masked_key}")
        logger.debug(f"API Base URL: {self.base_url or 'https://api.openai.com/v1 (default)'}")

        try:
            from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
            
            response = client.embeddings.create(
                model=model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            print(f"✅ Generated {len(embeddings)} embeddings", flush=True)
            logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except APIConnectionError as e:
            error_msg = (
                f"Connection error when generating embeddings: {str(e)}\n\n"
                "Troubleshooting steps:\n"
                "1. Check your internet connection\n"
                "2. Verify OPENAI_BASE_URL is correct (if using custom endpoint)\n"
                "3. Check if OpenAI API is accessible from your network\n"
                "4. Try again in a few moments (API might be temporarily unavailable)"
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
            
        except APITimeoutError as e:
            error_msg = (
                f"Timeout error when generating embeddings: {str(e)}\n\n"
                "The API request took too long. This might be due to:\n"
                "1. Network latency\n"
                "2. Large batch size\n"
                "3. API server load\n\n"
                "Try processing fewer documents at once or try again later."
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
            
        except AuthenticationError as e:
            error_msg = (
                f"Authentication error when generating embeddings: {str(e)}\n\n"
                "Your API key is invalid or expired. Please:\n"
                "1. Verify your OPENAI_API_KEY is correct\n"
                "2. Check if the API key has necessary permissions\n"
                "3. Generate a new API key from https://platform.openai.com/api-keys"
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
            
        except RateLimitError as e:
            error_msg = (
                f"Rate limit exceeded when generating embeddings: {str(e)}\n\n"
                "You've exceeded your API rate limit. Please:\n"
                "1. Wait a few moments before retrying\n"
                "2. Check your OpenAI usage limits\n"
                "3. Process documents in smaller batches"
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}\n\n"
            error_type = type(e).__name__
            
            # Provide specific guidance based on error type
            if "connection" in str(e).lower() or "ConnectionError" in error_type:
                error_msg += (
                    "Connection issue detected. Please:\n"
                    "1. Check your internet connection\n"
                    "2. Verify API endpoint is accessible\n"
                    "3. Check firewall/proxy settings\n"
                    "4. Try again in a few moments"
                )
            elif "timeout" in str(e).lower():
                error_msg += (
                    "Request timed out. Please:\n"
                    "1. Check your network connection\n"
                    "2. Try processing fewer documents at once\n"
                    "3. Retry the operation"
                )
            else:
                error_msg += (
                    "Unexpected error occurred. Please:\n"
                    "1. Check your API configuration\n"
                    "2. Verify your API key is valid\n"
                    "3. Check OpenAI API status\n"
                    "4. Review error details above"
                )
            
            logger.error(f"❌ {error_msg}")
            print(f"❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(self, collection_name: str = "raganything", persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise RuntimeError("ChromaDB not installed. Install with: pip install chromadb")
        
        # Store chromadb module for later use
        self._chromadb = chromadb

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.debug(f"Retrieved existing collection: {collection_name}")
        except Exception as e:
            error_msg = str(e)
            # Check for specific ChromaDB errors
            if "does not exist" in error_msg or "Collection" in error_msg and "does not exist" in error_msg:
                # Collection doesn't exist - create it
                logger.info(f"Collection '{collection_name}' does not exist. Creating new collection.")
                try:
                    self.collection = self.client.create_collection(name=collection_name)
                    logger.info(f"Created new collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}")
                    raise RuntimeError(
                        f"Failed to create ChromaDB collection '{collection_name}': {create_error}\n"
                        "This might be due to a corrupted database. Try resetting with: "
                        "python -m raganything.cli reset"
                    ) from create_error
            else:
                # Collection might be corrupted - try to recreate
                logger.warning(f"Could not get collection '{collection_name}': {e}. Attempting to recreate.")
                try:
                    # Try to delete corrupted collection if it exists
                    try:
                        self.client.delete_collection(name=collection_name)
                        logger.info(f"Deleted corrupted collection: {collection_name}")
                    except Exception:
                        pass  # Collection might not exist
                    
                    # Create new collection
                    self.collection = self.client.create_collection(name=collection_name)
                    logger.info(f"Created new collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}")
                    raise RuntimeError(
                        f"Failed to create ChromaDB collection '{collection_name}': {create_error}\n"
                        "This might be due to a corrupted database. Try resetting with: "
                        "python -m raganything.cli reset"
                    ) from create_error

    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to ChromaDB."""
        if not documents:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Generate embeddings if not provided
        if embeddings is None:
            config = get_config()
            generator = EmbeddingGenerator(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
            embeddings = generator.generate_embeddings(documents, model=config.embedding_model)

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        # Clean metadatas: Remove None values (ChromaDB doesn't accept None)
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned_meta = {k: v for k, v in meta.items() if v is not None}
            cleaned_metadatas.append(cleaned_meta)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=cleaned_metadatas,
        )

        return ids

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        # Generate query embedding if not provided
        if query_embedding is None:
            config = get_config()
            generator = EmbeddingGenerator(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
            query_embeddings = generator.generate_embeddings([query], model=config.embedding_model)
            query_embedding = query_embeddings[0] if query_embeddings else None

        if query_embedding is None:
            return []

        # Search in collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            # Check for ChromaDB metadata corruption errors
            if "Missing metadata" in error_msg or "metadata segment" in error_msg.lower():
                logger.error(f"ChromaDB metadata corruption detected during search: {e}")
                raise RuntimeError(
                    f"ChromaDB collection metadata is corrupted: {error_msg}\n"
                    "This usually happens after ChromaDB version updates or database corruption.\n"
                    "Please reset the database with: python -m raganything.cli reset\n"
                    "This will clear all embeddings and allow you to reprocess your documents."
                ) from e
            # Re-raise other errors
            raise

        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return formatted_results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete documents from ChromaDB."""
        if ids is None:
            # Delete all documents (recreate collection)
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(name=self.collection_name)
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}")
                return False
        else:
            try:
                self.collection.delete(ids=ids)
            except Exception as e:
                logger.error(f"Error deleting documents: {str(e)}")
                return False

        return True


class ContentProcessor:
    """Main content processor for AI Lab IGNFA - Legal RAG System."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "character",
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        Initialize content processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunk_strategy: Chunking strategy ('character', 'sentence', 'paragraph')
            vector_store: Vector store instance (creates ChromaDB if None)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.chunker = TextChunker()

        # Initialize vector store
        if vector_store is None:
            config = get_config()
            persist_dir = str(config.output_dir / "chroma_db")
            self.vector_store = ChromaVectorStore(persist_directory=persist_dir)
        else:
            self.vector_store = vector_store

        # Initialize embedding generator
        config = get_config()
        self.embedding_generator = EmbeddingGenerator(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using configured strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if self.chunk_strategy == "sentence":
            return self.chunker.chunk_by_sentence(text, self.chunk_size, self.chunk_overlap)
        elif self.chunk_strategy == "paragraph":
            return self.chunker.chunk_by_paragraph(text, self.chunk_size)
        else:  # character
            return self.chunker.chunk_by_character(text, self.chunk_size, self.chunk_overlap)

    def process_and_store(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process content, chunk it, and store in vector database.

        Args:
            content: Content to process
            doc_id: Optional document ID
            metadata: Optional document metadata

        Returns:
            List of chunk IDs
        """
        # Chunk content
        import sys
        content_length = len(content)
        print(f"✂️  Chunking content ({content_length:,} chars)...", flush=True)
        
        chunks = self.chunk_text(content)
        
        # Log chunking info for debugging
        num_chunks = len(chunks)
        logger.info(
            f"✂️  Chunking: {content_length:,} chars -> {num_chunks} chunks "
            f"(strategy={self.chunk_strategy}, size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        print(f"✅ Created {num_chunks} chunks", flush=True)
        
        if num_chunks == 1 and content_length > self.chunk_size:
            logger.warning(
                f"⚠️  Only 1 chunk created for {content_length:,} char document! "
                f"This may indicate a chunking issue."
            )
            print(f"⚠️  Warning: Only 1 chunk for large document ({content_length:,} chars)", flush=True)

        # Generate chunk IDs
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            if doc_id:
                chunk_id = f"{doc_id}_chunk_{i}"
            else:
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                chunk_id = f"chunk_{chunk_hash}_{i}"

            chunk_ids.append(chunk_id)
        
        logger.info(f"Generated {len(chunk_ids)} chunk IDs for document: {doc_id}")

        # Prepare metadatas
        metadatas = []
        for i, chunk_id in enumerate(chunk_ids):
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                "chunk_index": i,
                "chunk_id": chunk_id,
            })
            # Only add doc_id if it's not None
            if doc_id is not None:
                chunk_meta["doc_id"] = doc_id
            # Remove None values (ChromaDB doesn't accept None)
            chunk_meta = {k: v for k, v in chunk_meta.items() if v is not None}
            metadatas.append(chunk_meta)

        # Store in vector database
        import sys
        print(f"💾 Storing {len(chunk_ids)} chunks in vector database...", flush=True)
        logger.info(f"💾 Storing {len(chunk_ids)} chunks in vector database")
        
        self.vector_store.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )
        
        logger.info(f"✅ Successfully stored {len(chunk_ids)} chunks in vector database")
        print(f"✅ Stored {len(chunk_ids)} chunks in vector database", flush=True)
        return chunk_ids

    def get_statistics(self, content: str) -> Dict[str, Any]:
        """
        Get content statistics.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with statistics
        """
        chunks = self.chunk_text(content)

        return {
            "total_chars": len(content),
            "total_words": len(content.split()),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            "chunk_sizes": [len(c) for c in chunks],
        }

