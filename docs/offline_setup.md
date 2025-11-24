# Offline Setup Guide

This guide explains how to set up and use AI Lab IGNFA - Legal RAG System in offline environments.

## Overview

AI Lab IGNFA - Legal RAG System can be configured to work offline by using local models and avoiding cloud-based services where possible.

## Local Embedding Models

### Using Sentence Transformers

To use local embedding models instead of OpenAI:

```python
# Install sentence-transformers
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer

# Load local model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings locally
embeddings = model.encode(["Your text here"])
```

### Custom Embedding Integration

You can create a custom embedding generator:

```python
from raganything.processor import EmbeddingGenerator

class LocalEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts, model=None):
        return self.model.encode(texts).tolist()
```

## Local LLM Integration

### Using Ollama

```python
from raganything.query import RAGQuery

# Configure for Ollama
rag_query = RAGQuery(
    vector_store=vector_store,
    api_key="ollama",  # Not used but required
    base_url="http://localhost:11434/v1",  # Ollama API
    model="llama2",
)
```

### Using LM Studio

```python
rag_query = RAGQuery(
    vector_store=vector_store,
    base_url="http://localhost:1234/v1",  # LM Studio API
    model="local-model",
)
```

## Parser Configuration

### MinerU (Local)

MinerU runs locally and doesn't require internet:

```python
from raganything import RAGAnything

rag = RAGAnything(
    parser="mineru",
    parse_method="auto",
)
```

### Docling (Local)

Docling also runs locally:

```python
rag = RAGAnything(
    parser="docling",
)
```

## Vector Store (Local)

ChromaDB stores data locally:

```python
from raganything.processor import ChromaVectorStore

# Local persistent storage
vector_store = ChromaVectorStore(
    collection_name="raganything",
    persist_directory="./local_db",
)
```

## Complete Offline Setup

```python
from raganything import RAGAnything
from raganything.processor import ContentProcessor, ChromaVectorStore
from sentence_transformers import SentenceTransformer

# 1. Local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Local vector store
vector_store = ChromaVectorStore(
    persist_directory="./local_db",
)

# 3. Custom processor with local embeddings
class LocalContentProcessor(ContentProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
    
    # Override embedding generation
    # (Implementation details)

# 4. RAG-Anything with local components
rag = RAGAnything(
    parser="mineru",  # Local parser
    # Use custom processor
)
```

## Environment Variables

For offline setup, you can skip OpenAI configuration:

```env
# .env file for offline setup
# OPENAI_API_KEY=  # Leave empty or omit
OUTPUT_DIR=./output
PARSER=mineru
PARSE_METHOD=auto
```

## Limitations

1. **Embeddings**: Local models may have lower quality than OpenAI embeddings
2. **LLM**: Local LLMs may have different capabilities than cloud models
3. **Performance**: Local models may be slower depending on hardware

## Recommendations

1. Use `sentence-transformers` for local embeddings
2. Use Ollama or LM Studio for local LLM inference
3. Store vector database locally with ChromaDB
4. Use MinerU or Docling for local document parsing

## Example

See the examples directory for offline usage patterns.

