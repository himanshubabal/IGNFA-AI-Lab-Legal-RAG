# Context-Aware Processing Guide

This guide explains how RAG-Anything handles context-aware processing for better retrieval and answer generation.

## Overview

Context-aware processing ensures that retrieved information maintains its context and relationships, leading to more accurate and relevant answers.

## Context Formatting

### Basic Context Formatting

```python
from raganything.prompt import ContextFormatter

chunks = [
    {
        "content": "This is a chunk of text...",
        "metadata": {"source": "document1.pdf", "page": 1},
    },
    {
        "content": "Another chunk of text...",
        "metadata": {"source": "document1.pdf", "page": 2},
    },
]

# Format context
context = ContextFormatter.format_context(
    chunks=chunks,
    max_length=2000,
    separator="\n\n---\n\n",
)
```

### Context with Scores

```python
# Format context with similarity scores
context = ContextFormatter.format_context_with_scores(
    chunks=chunks,
    max_length=2000,
    show_scores=True,
)
```

## Query Processing

### Basic Query

```python
from raganything import RAGAnything

rag = RAGAnything()

# Query with context awareness
result = rag.query(
    query="What is the main topic?",
    n_results=5,  # Number of context chunks
    max_context_length=2000,  # Maximum context length
)
```

### Query with Sources

```python
# Get detailed source information
result = rag.query_handler.query_with_sources(
    query="Explain the methodology",
    n_results=5,
    include_scores=True,
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Scores: {result['scores']}")
```

## Chunking Strategies

Different chunking strategies affect context preservation:

### Character-based Chunking

```python
rag = RAGAnything(
    chunk_strategy="character",
    chunk_size=1000,
    chunk_overlap=200,
)
```

### Sentence-based Chunking

```python
rag = RAGAnything(
    chunk_strategy="sentence",
    chunk_size=1000,
    chunk_overlap=200,
)
```

### Paragraph-based Chunking

```python
rag = RAGAnything(
    chunk_strategy="paragraph",
    chunk_size=1000,
)
```

## Metadata Preservation

Metadata is preserved throughout the processing pipeline:

```python
result = rag.process_document_complete(
    file_path="document.pdf",
    doc_id="doc1",
)

# Metadata is stored with each chunk
# and can be retrieved during querying
```

## Best Practices

1. **Chunk Overlap**: Use overlap to maintain context between chunks
2. **Chunk Size**: Balance between too small (fragmented) and too large (less precise)
3. **Context Length**: Limit context length to stay within LLM token limits
4. **Source Tracking**: Always include source metadata for traceability

## Example

```python
from raganything import RAGAnything

rag = RAGAnything(
    chunk_strategy="sentence",
    chunk_size=1000,
    chunk_overlap=200,
)

# Process document
rag.process_document_complete("document.pdf", doc_id="doc1")

# Query with context
result = rag.query(
    "What are the key findings?",
    n_results=5,
    max_context_length=2000,
)

print(result['answer'])
print(f"Sources: {result['sources']}")
```

