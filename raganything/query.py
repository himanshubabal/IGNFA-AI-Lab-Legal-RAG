"""
Query and retrieval functionality for RAG-Anything.

This module handles semantic search, context retrieval,
and LLM integration for answer generation.
"""

import logging
from typing import Any, Dict, List, Optional

from raganything.base import BaseVectorStore
from raganything.config import get_config
from raganything.prompt import (
    get_query_prompt,
    get_system_prompt,
    get_custom_prompt,
    ContextFormatter,
)

logger = logging.getLogger(__name__)


class RAGQuery:
    """RAG query handler."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize RAG query handler.

        Args:
            vector_store: Vector store instance
            api_key: OpenAI API key
            base_url: Optional base URL for API
            model: LLM model name
            temperature: LLM temperature (0.0-2.0)
            top_p: LLM top_p (0.0-1.0)
            max_tokens: Maximum tokens for response
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = None
        self.custom_prompt_file: Optional[str] = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                client_kwargs = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                self._client = OpenAI(**client_kwargs)
            except ImportError:
                raise RuntimeError("OpenAI library not installed. Install with: pip install openai")

        return self._client

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        results = self.vector_store.search(query=query, n_results=n_results)

        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.get("score", 0.0) >= min_score]

        return results

    def query(
        self,
        query: str,
        n_results: int = 5,
        max_context_length: int = 2000,
        system_prompt: Optional[str] = None,
        query_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG system and generate answer.

        Args:
            query: User query
            n_results: Number of context chunks to retrieve
            max_context_length: Maximum context length
            system_prompt: Optional custom system prompt
            query_template: Optional custom query template
            temperature: LLM temperature
            stream: Whether to stream response

        Returns:
            Dictionary with answer and metadata
        """
        # Search for relevant context
        search_results = self.search(query=query, n_results=n_results)

        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "context": [],
                "sources": [],
            }

        # Format context
        context = ContextFormatter.format_context(
            chunks=search_results,
            max_length=max_context_length,
        )

        # Generate answer using LLM
        try:
            answer = self._generate_answer(
                query=query,
                context=context,
                system_prompt=system_prompt,
                query_template=query_template,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                stream=stream,
            )
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Error generating answer: {str(e)}"

        # Extract sources
        sources = []
        for result in search_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id", "Unknown")
            if source not in sources:
                sources.append(source)

        return {
            "answer": answer,
            "context": search_results,
            "sources": sources,
            "query": query,
        }

    def _generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        query_template: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate answer using LLM.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            query_template: Optional query template
            temperature: LLM temperature
            stream: Whether to stream response

        Returns:
            Generated answer
        """
        client = self._get_client()

        # Get prompts
        if system_prompt is None:
            # Try to load custom prompt from file
            custom_prompt = get_custom_prompt(
                prompt_file=self.custom_prompt_file,
                fallback=None
            )
            if custom_prompt:
                system_prompt = custom_prompt
            else:
                system_prompt = get_system_prompt()

        user_prompt = get_query_prompt(query=query, context=context, template=query_template)

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        if stream:
            # Stream response
            api_params["stream"] = True
            response = client.chat.completions.create(**api_params)

            answer_parts = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    answer_parts.append(chunk.choices[0].delta.content)

            return "".join(answer_parts)
        else:
            # Non-streaming response
            response = client.chat.completions.create(**api_params)

            return response.choices[0].message.content or ""

    def query_with_sources(
        self,
        query: str,
        n_results: int = 5,
        max_context_length: int = 2000,
        include_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Query with detailed source information.

        Args:
            query: User query
            n_results: Number of context chunks to retrieve
            max_context_length: Maximum context length
            include_scores: Whether to include similarity scores

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Search for relevant context
        search_results = self.search(query=query, n_results=n_results)

        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "context": [],
                "sources": [],
                "scores": [],
            }

        # Format context
        if include_scores:
            context = ContextFormatter.format_context_with_scores(
                chunks=search_results,
                max_length=max_context_length,
            )
        else:
            context = ContextFormatter.format_context(
                chunks=search_results,
                max_length=max_context_length,
            )

        # Generate answer
        try:
            answer = self._generate_answer(
                query=query,
                context=context,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Error generating answer: {str(e)}"

        # Extract detailed source information
        sources = []
        scores = []
        for result in search_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id", "Unknown")
            score = result.get("score", 0.0)

            source_info = {
                "source": source,
                "chunk_id": result.get("id"),
                "score": score,
                "metadata": metadata,
            }
            sources.append(source_info)
            scores.append(score)

        return {
            "answer": answer,
            "context": search_results,
            "sources": sources,
            "scores": scores,
            "query": query,
        }

