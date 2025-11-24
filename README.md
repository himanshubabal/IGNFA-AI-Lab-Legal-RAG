# RAG-Anything: All-in-One RAG Framework

RAG-Anything is a comprehensive framework for processing multimodal documents (PDFs, Office documents, images, tables, equations) and enabling semantic search and Q&A over the extracted content.

## Features

- **Multimodal Document Processing**: Support for PDFs, Office documents, images, and more
- **Multiple Parsers**: MinerU and Docling parser support
- **Vector Storage**: ChromaDB integration for semantic search
- **LLM Integration**: OpenAI API support for answer generation
- **Batch Processing**: Efficient processing of multiple documents
- **Flexible Chunking**: Multiple chunking strategies (character, sentence, paragraph)

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# With optional dependencies
pip install -e ".[all,ui]"
```

## Quick Start

```python
from raganything import RAGAnything

# Initialize RAG-Anything
rag = RAGAnything()

# Process a document
result = rag.process_document_complete("document.pdf")

# Query the knowledge base
answer = rag.query("What is this document about?")
print(answer["answer"])
```

## Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
OUTPUT_DIR=./output
PARSER=mineru
PARSE_METHOD=auto
```

## Documentation

See the `docs/` directory for detailed documentation.

## License

MIT License

