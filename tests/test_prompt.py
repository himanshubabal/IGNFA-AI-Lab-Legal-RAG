"""Tests for prompt module."""

from raganything.prompt import (
    ContextFormatter,
    QueryPromptTemplate,
    SystemPromptTemplate,
    get_query_prompt,
    get_system_prompt,
)


def test_query_prompt_template():
    """Test QueryPromptTemplate."""
    template = QueryPromptTemplate()
    prompt = template.format_query(query="Test query", context="Test context")

    assert "Test query" in prompt
    assert "Test context" in prompt


def test_system_prompt_template():
    """Test SystemPromptTemplate."""
    template = SystemPromptTemplate()
    prompt = template.get_system_prompt()

    assert len(prompt) > 0


def test_format_context():
    """Test context formatting."""
    chunks = [
        {"content": "Chunk 1", "metadata": {"source": "doc1"}},
        {"content": "Chunk 2", "metadata": {"source": "doc2"}},
    ]

    context = ContextFormatter.format_context(chunks, max_length=1000)

    assert "Chunk 1" in context
    assert "Chunk 2" in context


def test_format_context_with_scores():
    """Test context formatting with scores."""
    chunks = [
        {
            "content": "Chunk 1",
            "score": 0.9,
            "metadata": {"source": "doc1"},
        },
        {
            "content": "Chunk 2",
            "score": 0.8,
            "metadata": {"source": "doc2"},
        },
    ]

    context = ContextFormatter.format_context_with_scores(
        chunks, max_length=1000, show_scores=True
    )

    assert "Chunk 1" in context
    assert "0.9" in context or "0.900" in context


def test_get_query_prompt():
    """Test get_query_prompt function."""
    prompt = get_query_prompt("Test query", "Test context")
    assert "Test query" in prompt
    assert "Test context" in prompt


def test_get_system_prompt():
    """Test get_system_prompt function."""
    prompt = get_system_prompt()
    assert len(prompt) > 0

