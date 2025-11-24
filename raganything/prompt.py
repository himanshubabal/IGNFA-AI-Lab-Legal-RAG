"""
Prompt templates for LLM interactions in RAG-Anything.

This module provides prompt templates for query processing,
context formatting, and system prompts.
"""

from typing import List, Optional


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, template: str):
        """
        Initialize prompt template.

        Args:
            template: Template string with placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format template with provided values.

        Args:
            **kwargs: Values to substitute in template

        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)


class QueryPromptTemplate(PromptTemplate):
    """Template for RAG query prompts."""

    DEFAULT_TEMPLATE = """Based on the following context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    def __init__(self, template: Optional[str] = None):
        """
        Initialize query prompt template.

        Args:
            template: Custom template (uses default if None)
        """
        super().__init__(template or self.DEFAULT_TEMPLATE)

    def format_query(self, query: str, context: str) -> str:
        """
        Format query prompt.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return self.format(query=query, context=context)


class SystemPromptTemplate(PromptTemplate):
    """Template for system prompts."""

    DEFAULT_TEMPLATE = """You are a helpful assistant that answers questions based on provided context from documents.
You should:
- Provide accurate answers based on the context
- Cite sources when possible
- Say "I don't know" if the answer is not in the context
- Be concise and clear"""

    def __init__(self, template: Optional[str] = None):
        """
        Initialize system prompt template.

        Args:
            template: Custom template (uses default if None)
        """
        super().__init__(template or self.DEFAULT_TEMPLATE)

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return self.template


class ContextFormatter:
    """Formatter for context chunks."""

    @staticmethod
    def format_context(
        chunks: List[dict],
        max_length: int = 2000,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Format context chunks into a single string.

        Args:
            chunks: List of context chunks with 'content' and optional 'metadata'
            max_length: Maximum total length
            separator: Separator between chunks

        Returns:
            Formatted context string
        """
        formatted_chunks = []
        current_length = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            chunk_text = content
            if "metadata" in chunk and chunk["metadata"]:
                # Add metadata as citation
                metadata = chunk["metadata"]
                source = metadata.get("source", "Unknown")
                chunk_text = f"[Source: {source}]\n{content}"

            if current_length + len(chunk_text) > max_length:
                break

            formatted_chunks.append(chunk_text)
            current_length += len(chunk_text) + len(separator)

        return separator.join(formatted_chunks)

    @staticmethod
    def format_context_with_scores(
        chunks: List[dict],
        max_length: int = 2000,
        show_scores: bool = True,
    ) -> str:
        """
        Format context chunks with similarity scores.

        Args:
            chunks: List of context chunks with 'content', 'score', and 'metadata'
            max_length: Maximum total length
            show_scores: Whether to show similarity scores

        Returns:
            Formatted context string with scores
        """
        formatted_chunks = []
        current_length = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})

            chunk_text = content
            if show_scores:
                chunk_text = f"[Score: {score:.3f}] {chunk_text}"

            if "source" in metadata:
                chunk_text = f"[Source: {metadata['source']}]\n{chunk_text}"

            if current_length + len(chunk_text) > max_length:
                break

            formatted_chunks.append(chunk_text)
            current_length += len(chunk_text) + 20  # Approximate separator length

        return "\n\n---\n\n".join(formatted_chunks)


# Default prompt templates
default_query_template = QueryPromptTemplate()
default_system_template = SystemPromptTemplate()


def get_query_prompt(query: str, context: str, template: Optional[str] = None) -> str:
    """
    Get formatted query prompt.

    Args:
        query: User query
        context: Retrieved context
        template: Optional custom template

    Returns:
        Formatted prompt
    """
    prompt_template = QueryPromptTemplate(template) if template else default_query_template
    return prompt_template.format_query(query=query, context=context)


def get_system_prompt(template: Optional[str] = None) -> str:
    """
    Get system prompt.

    Args:
        template: Optional custom template

    Returns:
        System prompt string
    """
    prompt_template = SystemPromptTemplate(template) if template else default_system_template
    return prompt_template.get_system_prompt()

