# RAG-Anything: All-in-One RAG Framework

RAG-Anything is a comprehensive framework for processing multimodal documents (PDFs, Office documents, images, tables, equations) and enabling semantic search and Q&A over the extracted content.

## Features

- **Multimodal Document Processing**: Support for PDFs, Office documents, images, and more
- **Multiple Parsers**: MinerU and Docling parser support with flexible configuration
- **Vector Storage**: ChromaDB integration for semantic search
- **LLM Integration**: OpenAI API support for answer generation
- **Batch Processing**: Efficient processing of multiple documents with parallel support
- **Flexible Chunking**: Multiple chunking strategies (character, sentence, paragraph)
- **Enhanced Markdown**: Structured markdown generation with multimodal elements
- **Web UI**: Streamlit-based web interface for document processing and chat
- **CLI**: Command-line interface for easy integration

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG-Alt

# Set up Python 3.11 virtual environment (recommended)
./setup_venv.sh

# Or manually:
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### With Optional Dependencies

```bash
# Install with all optional features
pip install -e ".[all,ui,dev]"

# Or install specific extras:
# pip install -e ".[image]"  # Extended image format support
# pip install -e ".[text]"   # Text processing features
# pip install -e ".[ui]"     # Streamlit web UI
# pip install -e ".[dev]"    # Development tools
```

### System Dependencies

For Office document support, install LibreOffice:
- **macOS**: `brew install --cask libreoffice`
- **Linux**: `sudo apt-get install libreoffice`
- **Windows**: Download from [LibreOffice website](https://www.libreoffice.org/download/)

## Quick Start

### Basic Usage

```python
from raganything import RAGAnything

# Initialize RAG-Anything
rag = RAGAnything(
    parser="mineru",  # or "docling"
    parse_method="auto",
    chunk_size=1000,
    chunk_overlap=200,
)

# Process a document (parse + vectorize)
result = rag.process_document_complete("document.pdf")

# Query the knowledge base
answer = rag.query("What is this document about?")
print(answer["answer"])
print(f"Sources: {answer['sources']}")
```

### Using the Web UI

```bash
# Start Streamlit web interface
streamlit run raganything/webui/streamlit_app.py

# Or use the CLI
python -m raganything.cli ui
```

### Using the CLI

```bash
# Process a document
python -m raganything.cli process --file document.pdf

# Query the knowledge base
python -m raganything.cli query --query "What is this about?"

# Launch web UI
python -m raganything.cli ui
```

## Configuration

Create a `.env` file in the project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url  # Optional

# Output Configuration
OUTPUT_DIR=./output             # Default output directory

# Parser Configuration
PARSER=mineru                   # Parser selection: mineru or docling
PARSE_METHOD=auto               # Parse method: auto, ocr, or txt
```

## Supported Document Formats

### Document Formats
- **PDFs**: Research papers, reports, presentations
- **Office Documents**: DOC, DOCX, PPT, PPTX, XLS, XLSX
- **Images**: JPG, PNG, BMP, TIFF, GIF, WebP
- **Text Files**: TXT, MD

### Multimodal Elements
- **Images**: Photographs, diagrams, charts, screenshots
- **Tables**: Data tables, comparison charts, statistical summaries
- **Equations**: Mathematical formulas in LaTeX format

## Examples

See the `examples/` directory for comprehensive usage examples:

- `raganything_example.py`: End-to-end document processing
- `modalprocessors_example.py`: Multimodal content processing
- `batch_processing_example.py`: Batch document processing
- `enhanced_markdown_example.py`: Markdown generation
- `insert_content_list_example.py`: Content insertion
- `office_document_test.py`: Office document parsing test
- `image_format_test.py`: Image format parsing test
- `text_format_test.py`: Text format parsing test

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Batch Processing Guide](docs/batch_processing.md)
- [Context-Aware Processing](docs/context_aware_processing.md)
- [Enhanced Markdown](docs/enhanced_markdown.md)
- [Offline Setup](docs/offline_setup.md)

## API Reference

### Main Classes

- `RAGAnything`: Main orchestrator class
- `ParserFactory`: Factory for creating parsers
- `ContentProcessor`: Content processing and chunking
- `RAGQuery`: Query and retrieval functionality
- `BatchProcessor`: Batch document processing

### Key Methods

```python
# Process document completely (parse + vectorize)
rag.process_document_complete(file_path, doc_id=None, **kwargs)

# Process document (parse only)
rag.process_document(file_path, output_dir=None, **kwargs)

# Query knowledge base
rag.query(query, n_results=5, max_context_length=2000)

# Insert pre-parsed content
rag.insert_content_list(content_list, doc_id=None)

# Search without generating answer
rag.search(query, n_results=5, min_score=0.0)
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=raganything --cov-report=html
```

### Code Quality

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Format code
black raganything/ tests/ examples/
isort raganything/ tests/ examples/

# Lint code
flake8 raganything/ tests/ examples/
mypy raganything/
```

## Project Structure

```
RAG-Alt/
├── raganything/          # Main package
│   ├── base.py           # Base classes
│   ├── config.py         # Configuration
│   ├── parser.py         # Document parsers
│   ├── processor.py      # Content processing
│   ├── query.py          # Query functionality
│   ├── raganything.py    # Main orchestrator
│   └── webui/            # Web UI
├── examples/             # Usage examples
├── docs/                 # Documentation
├── tests/                # Test suite
└── scripts/              # Utility scripts
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by and based on the [RAG-Anything](https://github.com/HKUDS/RAG-Anything) framework by HKUDS.

