# AI Lab IGNFA - Legal RAG System

**A comprehensive Retrieval-Augmented Generation (RAG) framework specifically designed for processing legal documents and enabling intelligent question-answering over complex legal content.**

---

## 🎯 Overview

The AI Lab IGNFA - Legal RAG System is an end-to-end solution for processing multimodal legal documents (PDFs, Office documents, images, tables, equations) and enabling semantic search and question-answering over extracted content. Built for legal researchers, practitioners, and institutions working with Indian environmental law and legal documentation.

### Key Capabilities

- **📄 Multimodal Document Processing**: Extract text, images, tables, and equations from complex legal documents
- **🔍 Semantic Search**: Find relevant information across your entire document corpus using vector embeddings
- **💬 Intelligent Q&A**: Get accurate, cited answers to legal questions with source attribution
- **🤖 Multiple LLM Support**: Flexible integration with OpenAI models (GPT-4o, GPT-4, GPT-3.5-turbo)
- **📊 Smart Document Management**: Automatic tracking, incremental processing, and document lifecycle management
- **🌐 Web Interface**: User-friendly Streamlit-based UI for document upload, processing, and chat
- **⚡ Batch Processing**: Efficient processing of large document collections
- **🎛️ Highly Configurable**: Extensive configuration options for parsing, chunking, and retrieval

---

## ✨ Features in Detail

### 1. **Multimodal Document Processing**

The system supports a wide range of document formats and multimodal content:

#### Supported Document Formats
- **PDFs** (.pdf): Research papers, legal documents, reports, presentations
- **Office Documents**: 
  - Word: .doc, .docx
  - PowerPoint: .ppt, .pptx
  - Excel: .xls, .xlsx
- **Images**: .jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp
- **Text Files**: .txt, .md

#### Multimodal Elements Extracted
- **Images**: Photographs, diagrams, charts, screenshots with metadata
- **Tables**: Structured data tables with content preservation
- **Equations**: Mathematical formulas in LaTeX format
- **Enhanced Markdown**: Structured markdown with multimodal element placeholders

### 2. **Advanced Parsing Capabilities**

#### Two Powerful Parsers

**MinerU (Recommended)**
- State-of-the-art document understanding
- Automatic layout detection
- OCR capabilities for scanned documents
- Verification files (span.pdf, layout.pdf) for quality assurance
- Supports auto, OCR, and text extraction modes

**Docling**
- Alternative parser option
- Robust document structure extraction
- Good performance on structured documents

#### Parse Methods
- **auto**: Automatic detection of best parsing strategy
- **ocr**: Force OCR extraction (useful for scanned documents)
- **txt**: Text-only extraction

### 3. **Intelligent Content Chunking**

Flexible chunking strategies optimized for legal documents:

- **Character-based**: Fixed-size chunks with overlap (default: 1000 chars, 200 overlap)
- **Sentence-based**: Respect sentence boundaries for better context
- **Paragraph-based**: Preserve paragraph structure for legal precision

Configurable parameters:
- Chunk size (500-2000 characters)
- Chunk overlap (0-500 characters)
- Chunk strategy selection

### 4. **Vector Storage & Semantic Search**

- **ChromaDB Integration**: Persistent vector database for embeddings
- **Multiple Embedding Models**: 
  - `text-embedding-3-large` (3072 dimensions, highest quality)
  - `text-embedding-3-small` (1536 dimensions, balanced, default)
  - `text-embedding-ada-002` (1536 dimensions, legacy)
- **Similarity Search**: Efficient retrieval of relevant document chunks
- **Metadata Filtering**: Search with document-level and chunk-level metadata

### 5. **LLM-Powered Q&A**

- **Multiple Model Support**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo
- **Configurable Parameters**:
  - Temperature (0.0-2.0) for response creativity
  - Top-p for nucleus sampling
  - Max tokens for response length control
- **Custom Prompts**: Domain-specific system prompts (e.g., legal expert persona)
- **Source Attribution**: Automatic citation of document sources
- **Context Management**: Configurable context window and chunk selection

### 6. **Smart Document Management**

#### Document Tracker
- Automatic tracking of processed documents
- File modification time detection
- Incremental processing (only new/updated documents)
- Removal tracking for deleted documents

#### Smart Processor
- Detects new, updated, and removed documents
- Skips unchanged documents for efficiency
- Batch processing with error handling
- Status reporting and metrics

### 7. **Web User Interface**

Comprehensive Streamlit-based web interface with:

**💬 Chat Tab**
- Interactive chat with your documents
- Real-time query processing
- Source citations
- Chat history management

**📄 Documents Tab**
- Drag-and-drop document upload
- Auto-processing on upload
- Document list with details (chunks, processing time)
- Individual document processing
- Document removal

**📊 Status Tab**
- Comprehensive status dashboard
- Processed documents table
- Unprocessed documents list
- Removed documents tracking

**⚙️ Configuration Sidebar**
- LLM model selection and parameters
- Embedding model configuration
- Query configuration (number of results, context length)
- Parser settings (parser type, parse method, chunk settings)
- Custom prompt editor
- Document processing controls

### 8. **Command-Line Interface**

Full-featured CLI for automation and integration:

```bash
# Process single document
python -m raganything.cli process --file document.pdf

# Process all documents (smart incremental)
python -m raganything.cli process-all

# Check document status
python -m raganything.cli status

# Query knowledge base
python -m raganything.cli query --query "Your question here"

# Launch web UI
python -m raganything.cli ui

# Reset all data
python -m raganything.cli reset
```

### 9. **Batch Processing**

- Process entire directories recursively
- Pattern-based file filtering (e.g., `*.pdf`)
- Parallel processing support
- Progress tracking and error reporting
- Resume capability

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Configuration](#⚙️-configuration)
4. [Usage Guide](#📖-usage-guide)
5. [API Reference](#📚-api-reference)
6. [Web UI Guide](#🌐-web-ui-guide)
7. [CLI Reference](#💻-cli-reference)
8. [Advanced Features](#🚀-advanced-features)
9. [Troubleshooting](#🔧-troubleshooting)
10. [Deployment](#🚀-deployment)
11. [Development](#👨‍💻-development)
12. [Architecture](#🏗️-architecture)

---

## 📦 Installation

### Prerequisites

- **Python**: 3.11 or 3.12 (recommended)
  - Python 3.11 is optimal for ChromaDB compatibility
  - Python 3.14+ may have dependency issues
- **Git**: For cloning the repository
- **LibreOffice** (for Office document support):
  - **macOS**: `brew install --cask libreoffice`
  - **Linux**: `sudo apt-get install libreoffice`
  - **Windows**: Download from [LibreOffice website](https://www.libreoffice.org/download/)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd RAG-Alt
```

### Step 2: Create Virtual Environment

**Recommended: Use the setup script**

```bash
# Make script executable (Linux/macOS)
chmod +x setup_venv.sh

# Run setup script
./setup_venv.sh
```

**Or manually:**

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Step 4: Install Parser Dependencies

**MinerU (Recommended):**

```bash
# Install UV (MinerU's package manager)
pip install uv

# Install MinerU
uv pip install -U "mineru[core]"
```

**Docling (Alternative):**

```bash
pip install docling
```

### Step 5: Configure Environment

Create a `.env` file in the project root:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom OpenAI endpoint
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Embedding Model (default: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small

# Optional: LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0
LLM_MAX_TOKENS=2000

# Optional: Query Configuration
QUERY_N_RESULTS=5
QUERY_MAX_CONTEXT_LENGTH=2000
QUERY_MIN_SCORE=0.0

# Optional: Output Directory
OUTPUT_DIR=./output

# Optional: Parser Configuration
PARSER=mineru
PARSE_METHOD=auto
MINERU_OUTPUT_FLAG_SPAN=true

# Optional: Custom Prompt File
PROMPT_FILE=prompt.md
```

### Step 6: Verify Installation

```bash
# Test import
python -c "from raganything import RAGAnything; print('Installation successful!')"

# Check CLI
python -m raganything.cli --help
```

---

## 🚀 Quick Start

### Basic Usage (Python)

```python
from raganything import RAGAnything

# Initialize the system
rag = RAGAnything(
    parser="mineru",        # or "docling"
    parse_method="auto",    # or "ocr", "txt"
    chunk_size=1000,        # characters per chunk
    chunk_overlap=200,      # overlap between chunks
)

# Process a legal document (parse + chunk + embed)
result = rag.process_document_complete(
    file_path="documents/legal-act.pdf",
    doc_id="legal-act-1980"
)

print(f"Created {result['num_chunks']} chunks")

# Query your legal documents
answer = rag.query(
    query="What are the key provisions of this act?",
    n_results=5,              # Number of chunks to retrieve
    max_context_length=2000   # Maximum context length
)

print(answer["answer"])
print(f"Sources: {answer['sources']}")
```

### Using the Web UI

```bash
# Launch web UI (recommended - auto-processes documents)
python run_ui.py

# Or use CLI
python -m raganything.cli ui

# Or directly with Streamlit
streamlit run streamlit_app.py
```

The web UI will:
1. Auto-detect and process documents in `documents/` directory
2. Launch at `http://localhost:8501`
3. Allow you to upload, process, and chat with documents

### Using the CLI

```bash
# Process a single document
python -m raganything.cli process --file documents/act.pdf

# Process all documents in documents/ directory
python -m raganything.cli process-all

# Check status
python -m raganything.cli status

# Query
python -m raganything.cli query --query "What does this document say about penalties?"
```

---

## ⚙️ Configuration

### Environment Variables

All configuration is managed through environment variables in `.env` file:

#### OpenAI Configuration

```env
# Required for LLM and embeddings
OPENAI_API_KEY=sk-...

# Optional: Custom endpoint (for OpenAI-compatible APIs)
OPENAI_BASE_URL=https://api.openai.com/v1
```

#### Embedding Model

```env
# Options: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002
EMBEDDING_MODEL=text-embedding-3-small
```

**Model Comparison:**
- `text-embedding-3-large`: 3072 dimensions, highest quality, slower
- `text-embedding-3-small`: 1536 dimensions, balanced (recommended)
- `text-embedding-ada-002`: 1536 dimensions, legacy, cost-effective

#### LLM Configuration

```env
# Model selection
LLM_MODEL=gpt-3.5-turbo  # Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

# Response parameters
LLM_TEMPERATURE=0.7      # 0.0-2.0, higher = more creative
LLM_TOP_P=1.0           # 0.0-1.0, nucleus sampling
LLM_MAX_TOKENS=2000     # Max response length (optional, None = no limit)
```

#### Query Configuration

```env
QUERY_N_RESULTS=5              # Number of chunks to retrieve (1-20)
QUERY_MAX_CONTEXT_LENGTH=2000  # Max context in characters (500-8000)
QUERY_MIN_SCORE=0.0            # Minimum similarity score (0.0-1.0)
```

#### Parser Configuration

```env
# Parser selection
PARSER=mineru  # Options: mineru (recommended), docling

# Parse method
PARSE_METHOD=auto  # Options: auto (recommended), ocr, txt

# MinerU specific
MINERU_OUTPUT_FLAG_SPAN=true  # Generate verification files (span.pdf, etc.)
```

#### Output Configuration

```env
OUTPUT_DIR=./output  # Directory for processed outputs and vector database
```

#### Prompt Configuration

```env
PROMPT_FILE=prompt.md  # Path to custom system prompt file
```

### Directory Structure

```
RAG-Alt/
├── documents/              # Place your legal documents here
│   ├── act-1980.pdf
│   ├── case-study.docx
│   └── ...
├── output/                 # Generated outputs (auto-created)
│   ├── chroma_db/          # Vector database (Git LFS)
│   ├── .document_tracker.json  # Document tracking
│   └── [document-name]/    # Parsed outputs per document
│       ├── document_extracted.md
│       └── ...
├── prompt.md              # Custom system prompt (optional)
├── .env                   # Environment variables (create this)
└── ...
```

---

## 📖 Usage Guide

### Document Processing Workflow

1. **Place Documents**: Add PDFs, DOCX, PPTX files to `documents/` directory
2. **Process**: Run `python -m raganything.cli process-all` or use Web UI
3. **Query**: Ask questions using CLI, Web UI, or Python API

### Processing Options

#### Extract Only (No Embedding)

```bash
# Extract text without chunking/embedding
python -m raganything.cli process-all --extract-only

# Force re-extraction
python -m raganything.cli process-all --extract-only --force-extract
```

#### Force Reprocessing

```bash
# Reprocess all documents even if unchanged
python -m raganything.cli process-all --force
```

### Python API Examples

#### Basic Document Processing

```python
from raganything import RAGAnything

rag = RAGAnything()

# Process with all default settings
result = rag.process_document_complete("documents/legal-doc.pdf")

# Process with custom document ID
result = rag.process_document_complete(
    file_path="documents/act.pdf",
    doc_id="environmental-act-1980"
)

# Process with custom output directory
result = rag.process_document_complete(
    file_path="documents/doc.pdf",
    output_dir="./custom_output"
)
```

#### Querying Documents

```python
# Basic query
result = rag.query("What are the penalties for violations?")

# Query with custom parameters
result = rag.query(
    query="Explain the key provisions",
    n_results=10,              # Retrieve more chunks
    max_context_length=4000,   # Larger context window
    temperature=0.3,           # More deterministic
)

print(result["answer"])       # Generated answer
print(result["sources"])      # List of source documents
print(result["context"])      # Retrieved chunks
```

#### Search Without LLM

```python
# Get search results without generating answer
results = rag.search(
    query="environmental impact assessment",
    n_results=5,
    min_score=0.7  # Minimum similarity threshold
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Metadata: {result['metadata']}")
```

#### Batch Processing

```python
from raganything.batch import BatchProcessor

batch = BatchProcessor(raganything=RAGAnything())

# Process all PDFs in directory
results = batch.process_directory(
    directory="documents",
    pattern="*.pdf",
    recursive=True
)

print(f"Processed: {results['successful']}")
print(f"Failed: {results['failed']}")
```

### Custom Prompts

Create `prompt.md` in project root for domain-specific prompts:

```markdown
# Legal Expert Assistant

You are a senior legal expert specializing in Indian environmental law.

## Instructions

1. Answer questions using ONLY information from the provided documents
2. Cite sources using format: [Document Name, Page/Section]
3. If information is not in documents, state "Insufficient information"
4. Provide precise legal citations
5. Distinguish between facts and legal interpretations

## Answer Format

- Concise answer first
- Numbered supporting points with citations
- List all source documents
```

The system automatically loads `prompt.md` if it exists.

---

## 📚 API Reference

### Main Classes

#### `RAGAnything`

Main orchestrator class for the RAG system.

**Initialization:**
```python
RAGAnything(
    parser: str = "mineru",
    parse_method: str = "auto",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_strategy: str = "character",
    vector_store_persist_dir: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_top_p: Optional[float] = None,
    llm_max_tokens: Optional[int] = None,
)
```

**Key Methods:**

- `process_document_complete(file_path, doc_id=None, output_dir=None, **kwargs)` - Parse + chunk + embed
- `process_document(file_path, output_dir=None, doc_id=None, **kwargs)` - Parse only
- `query(query, n_results=5, max_context_length=2000, **kwargs)` - Query with LLM answer
- `search(query, n_results=5, min_score=0.0)` - Search without LLM
- `insert_content_list(content_list, doc_id=None, metadata=None)` - Insert pre-parsed content

#### `SmartProcessor`

Intelligent document processor with tracking.

```python
SmartProcessor(
    documents_dir: str = "documents",
    raganything: Optional[RAGAnything] = None
)

# Methods
processor.process_all(force_reprocess=False, extract_only=False, force_extract=False)
processor.get_document_status()
```

#### `ContentProcessor`

Core content processing and chunking.

```python
ContentProcessor(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_strategy: str = "character",
    vector_store: Optional[BaseVectorStore] = None
)
```

#### `RAGQuery`

Query handler for semantic search and answer generation.

```python
RAGQuery(
    vector_store: BaseVectorStore,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None
)
```

---

## 🌐 Web UI Guide

### Launching the Web UI

```bash
# Recommended: Auto-processes documents on startup
python run_ui.py

# Alternative: CLI command
python -m raganything.cli ui

# Direct Streamlit
streamlit run streamlit_app.py
```

### Web UI Features

#### Chat Interface (💬 Tab)

- **Ask Questions**: Type questions in the chat input
- **View Answers**: Get LLM-generated answers with source citations
- **Source Expansion**: Click "Sources" to see document references
- **Chat History**: Conversation persists during session
- **Clear Chat**: Reset conversation history

#### Document Management (📄 Tab)

**Upload Documents:**
1. Click "Upload a document"
2. Select PDF, DOCX, PPTX, XLSX, TXT, MD, or images
3. Click "Save & Process Document"
4. Document is saved to `documents/` and processed automatically (if auto-process enabled)

**View Documents:**
- Processed documents show chunk count and processing time
- Expand to see details: path, document ID, chunks
- Remove button to delete document and tracking

**Process Documents:**
- "Refresh & Process" button processes all new/updated documents
- Individual "Process" buttons for unprocessed documents
- Force reprocess checkbox available

#### Status Dashboard (📊 Tab)

- **Metrics**: Total files, processed, unprocessed, removed
- **Processed Table**: Detailed table with all processed documents
- **Unprocessed List**: Documents waiting to be processed
- **Removed List**: Documents tracked but files deleted

#### Configuration Sidebar (⚙️)

**LLM Configuration:**
- Model selection (GPT-4o, GPT-4, GPT-3.5-turbo, etc.)
- Temperature slider (0.0-2.0)
- Top P slider (0.0-1.0)
- Max tokens input

**Query Configuration:**
- Number of results slider (1-20)
- Max context length slider (500-8000)

**Embedding Model:**
- Model selection dropdown
- Info button for model details

**Parser Configuration:**
- Parser selection (MinerU/Docling)
- Parse method (auto/ocr/txt)
- Chunk size slider (500-2000)
- Chunk overlap slider (0-500)
- MinerU verification files toggle

**Custom Prompt:**
- View current prompt
- Edit prompt in text area
- Save to `prompt.md`
- Reload prompt

**Document Processing:**
- Auto-process toggle
- Extract-only checkbox
- Force re-extraction checkbox
- Force reprocess checkbox
- Process All button

**Reset & Clear:**
- Reset all data (confirmation required)
- Clears vector store and document tracker

---

## 💻 CLI Reference

### Commands

#### `process` - Process Single Document

```bash
python -m raganything.cli process --file <path>
```

**Options:**
- `--file, -f`: File path to process (required)
- `--parser`: Parser to use (mineru/docling)
- `--output-dir, -o`: Output directory
- `--output-flag-span`: Enable MinerU verification files
- `--no-output-flag-span`: Disable verification files
- `--extract-only`: Extract text only (skip chunking/embedding)
- `--force-extract`: Force re-extraction

**Examples:**
```bash
python -m raganything.cli process --file documents/act.pdf
python -m raganything.cli process --file documents/doc.pdf --extract-only
python -m raganything.cli process --file documents/doc.pdf --parser docling
```

#### `process-all` - Process All Documents

```bash
python -m raganything.cli process-all
```

**Options:**
- `--documents-dir, -d`: Documents directory (default: documents)
- `--force`: Force reprocess all documents
- `--extract-only`: Extract only (skip chunking/embedding)
- `--force-extract`: Force re-extraction

**Examples:**
```bash
python -m raganything.cli process-all
python -m raganything.cli process-all --force
python -m raganything.cli process-all --extract-only
```

#### `status` - Check Document Status

```bash
python -m raganything.cli status
```

**Options:**
- `--documents-dir, -d`: Documents directory (default: documents)

**Output:**
- Total files count
- Processed documents with chunk counts
- Unprocessed documents
- Removed documents (tracked but missing)

#### `query` - Query Knowledge Base

```bash
python -m raganything.cli query --query "<your question>"
```

**Options:**
- `--query, -q`: Query string (required)
- `--parser`: Parser selection (for initialization)

**Example:**
```bash
python -m raganything.cli query --query "What are the penalties for environmental violations?"
```

#### `ui` - Launch Web UI

```bash
python -m raganything.cli ui
```

Opens Streamlit web interface in your browser.

#### `reset` - Reset All Data

```bash
python -m raganything.cli reset
```

**Warning**: This will delete:
- All vector embeddings
- Document tracking data
- Optionally: All output directory contents

Requires confirmation.

#### `list-embeddings` - List Available Embedding Models

```bash
python -m raganything.cli list-embeddings
```

Shows available OpenAI embedding models and their specifications.

---

## 🚀 Advanced Features

### Custom Embedding Models

The system uses OpenAI embeddings by default, but you can integrate custom models:

```python
from raganything.processor import EmbeddingGenerator

class CustomEmbeddingGenerator(EmbeddingGenerator):
    def generate_embeddings(self, texts, model=None):
        # Your custom embedding logic
        return embeddings
```

### Offline Setup

For offline environments, see [docs/offline_setup.md](docs/offline_setup.md) for:
- Local embedding models (Sentence Transformers)
- Local LLM integration (Ollama, LM Studio)
- Offline parser configuration

### Batch Processing Patterns

```python
from raganything.batch import BatchProcessor
from raganything import RAGAnything

batch = BatchProcessor(raganything=RAGAnything())

# Process specific file types
results = batch.process_directory(
    directory="documents",
    pattern="*.pdf",
    recursive=True
)

# Process with custom output
results = batch.process_directory(
    directory="documents",
    output_dir="./custom_output"
)
```

### Enhanced Markdown Processing

For structured markdown with multimodal elements:

```python
from raganything.enhanced_markdown import EnhancedMarkdownProcessor

processor = EnhancedMarkdownProcessor()
enhanced_md = processor.process(content_list)
```

See [docs/enhanced_markdown.md](docs/enhanced_markdown.md) for details.

### Context-Aware Processing

For document-aware chunking and context preservation:

See [docs/context_aware_processing.md](docs/context_aware_processing.md)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` for dependencies

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -e .
```

#### 2. ChromaDB Errors

**Problem**: ChromaDB metadata corruption

**Solution**:
```bash
# Reset the database
python -m raganything.cli reset
```

#### 3. OpenAI API Errors

**Problem**: `Invalid API key` or rate limit errors

**Solution**:
- Verify `OPENAI_API_KEY` in `.env`
- Check API key validity
- Implement rate limiting for large batches

#### 4. Parser Not Found

**Problem**: MinerU or Docling not found

**Solution**:
```bash
# Install MinerU
pip install uv
uv pip install -U "mineru[core]"

# Or install Docling
pip install docling
```

#### 5. Office Document Errors

**Problem**: Office documents fail to parse

**Solution**:
- Install LibreOffice
- macOS: `brew install --cask libreoffice`
- Linux: `sudo apt-get install libreoffice`

#### 6. Out of Memory

**Problem**: Large documents cause memory issues

**Solution**:
- Reduce chunk size
- Process documents individually
- Increase system memory

#### 7. Vector Database Too Large

**Problem**: ChromaDB exceeds GitHub file size limits

**Solution**:
- System uses Git LFS for chroma_db
- Database is already configured for LFS
- If issues persist, check `.gitattributes`

### Getting Help

1. Check documentation in `docs/` directory
2. Review error messages carefully
3. Check `.env` configuration
4. Verify file permissions
5. Review logs in output directory

---

## 🚀 Deployment

### Streamlit Community Cloud

The system is configured for easy deployment on Streamlit Community Cloud.

#### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Add secrets (environment variables):
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `EMBEDDING_MODEL`: (Optional) embedding model
     - Other optional configs

3. **Configure Environment**
   - Secrets are automatically loaded as environment variables
   - `.env` file in repo is also read (but secrets are preferred)

#### Important Notes for Cloud Deployment

- **ChromaDB**: Uses Git LFS (already configured)
- **Vector Database**: Persists across deployments via Git
- **Documents**: Upload via web UI (stored in ephemeral storage)
- **API Keys**: Use Streamlit secrets, not `.env` in repo

### Local Production Deployment

For local production:

1. Use systemd/PM2 for process management
2. Set up reverse proxy (nginx)
3. Configure SSL certificates
4. Use persistent storage for `output/` directory
5. Set up backups for `chroma_db/`

---

## 👨‍💻 Development

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=raganything --cov-report=html

# Run specific test file
pytest tests/test_processor.py
```

### Code Quality

```bash
# Format code
black raganything/ tests/ examples/

# Sort imports
isort raganything/ tests/ examples/

# Lint
flake8 raganything/ tests/ examples/
pylint raganything/

# Type checking
mypy raganything/
```

### Project Structure

```
RAG-Alt/
├── raganything/              # Main package
│   ├── __init__.py          # Package exports
│   ├── base.py              # Base classes and interfaces
│   ├── config.py            # Configuration management
│   ├── parser.py            # Document parsers (MinerU, Docling)
│   ├── processor.py         # Content processing and chunking
│   ├── query.py             # Query and retrieval
│   ├── raganything.py       # Main orchestrator class
│   ├── smart_processor.py   # Smart document processor
│   ├── document_tracker.py  # Document tracking system
│   ├── prompt.py            # Prompt management
│   ├── enhanced_markdown.py # Enhanced markdown processing
│   ├── modalprocessors.py   # Multimodal content processors
│   ├── batch.py             # Batch processing
│   ├── batch_parser.py      # Batch parser utilities
│   ├── utils.py             # Utility functions
│   └── webui/               # Web UI
│       └── streamlit_app.py # Streamlit application
├── examples/                 # Usage examples
├── docs/                     # Documentation
├── tests/                    # Test suite
├── documents/                # Input documents directory
├── output/                   # Output directory
├── prompt.md                 # Custom prompt file
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project metadata
├── README.md                 # This file
├── INSTALLATION.md           # Detailed installation guide
├── USAGE.md                  # Usage examples
├── FEATURES.md               # Features documentation
├── streamlit_app.py          # Streamlit Community Cloud entry point
└── run_ui.py                 # UI launch script
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Web UI     │  │     CLI      │  │  Python API  │ │
│  │ (Streamlit)  │  │   Commands   │  │   Library    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  RAGAnything (Orchestrator)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Parser     │  │  Processor   │  │ Query Handler│ │
│  │  (MinerU/    │  │  (Chunking/  │  │ (Search/     │ │
│  │   Docling)   │  │  Embedding)  │  │  LLM Q&A)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   OpenAI     │  │   ChromaDB   │  │   Document   │
│   APIs       │  │  (Vector DB) │  │   Tracker    │
│ (LLM/Embed)  │  │              │  │   (JSON)     │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Data Flow

1. **Document Input**: User uploads/places document in `documents/`
2. **Parsing**: Parser (MinerU/Docling) extracts content
3. **Chunking**: ContentProcessor splits into chunks
4. **Embedding**: OpenAI API generates vector embeddings
5. **Storage**: Embeddings stored in ChromaDB with metadata
6. **Query**: User asks question
7. **Retrieval**: Semantic search finds relevant chunks
8. **Generation**: LLM generates answer from retrieved context
9. **Response**: Answer with source citations returned

### Key Design Principles

- **Modularity**: Separate concerns (parsing, processing, querying)
- **Extensibility**: Easy to add new parsers, processors, LLMs
- **Persistence**: Vector database and tracking persist across sessions
- **Smart Processing**: Incremental updates, skip unchanged documents
- **Flexibility**: Multiple interfaces (Web, CLI, API)
- **Configuration**: Environment-based, no code changes needed

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project is inspired by and based on the [RAG-Anything](https://github.com/HKUDS/RAG-Anything) framework by HKUDS.

**Parser Credits:**
- [MinerU](https://github.com/opendatalab/MinerU) - Advanced document understanding
- [Docling](https://github.com/DS4SD/docling) - Document parsing library

---

## 📞 Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review troubleshooting section
3. Check GitHub issues
4. Create new issue with details

---

**Built with ❤️ for Legal Research and Document Intelligence**
