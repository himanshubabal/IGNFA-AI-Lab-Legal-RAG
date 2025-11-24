# Usage Guide

## Document Directory Structure

### Input Documents
Place your documents in the `documents/` directory:

```
documents/
├── your-document.pdf
├── report.docx
├── presentation.pptx
├── data.xlsx
└── notes.txt
```

**Supported formats:**
- PDF files (`.pdf`)
- Office documents (`.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`)
- Images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.webp`)
- Text files (`.txt`, `.md`)

### Output Directory
Processed documents and parsed outputs are saved to the `output/` directory (configurable via `.env`):

```
output/
├── parsed/
│   └── document_name.json
└── chroma_db/
    └── (vector database files)
```

## Basic Usage

### 1. Process a Single Document

```bash
# Activate virtual environment
source venv/bin/activate

# Process a document from the documents directory
python -m raganything.cli process --file documents/your-document.pdf

# Or use absolute path
python -m raganything.cli process --file /path/to/your/document.pdf
```

### 2. Process a Document with Python

```python
from raganything import RAGAnything

# Initialize
rag = RAGAnything()

# Process a document
result = rag.process_document_complete(
    file_path="documents/your-document.pdf",
    doc_id="my-document"
)

print(f"Processed {result['num_chunks']} chunks")
```

### 3. Query Your Documents

```bash
# Using CLI
python -m raganything.cli query --query "What is this document about?"

# Using Python
from raganything import RAGAnything

rag = RAGAnything()
answer = rag.query("What is this document about?")
print(answer["answer"])
```

### 4. Batch Process Multiple Documents

```python
from raganything.batch import BatchProcessor
from raganything import RAGAnything

# Initialize batch processor
batch_processor = BatchProcessor(raganything=RAGAnything())

# Process all PDFs in documents directory
results = batch_processor.process_directory(
    directory="documents",
    pattern="*.pdf",
    recursive=True
)

print(f"Processed {results['successful']} documents")
```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=your_base_url  # Optional

# Output Configuration
OUTPUT_DIR=./output             # Default: ./output

# Parser Configuration
PARSER=mineru                   # Options: mineru or docling
PARSE_METHOD=auto               # Options: auto, ocr, or txt
```

### Default Directories

- **Input Documents**: `documents/` (create this directory)
- **Output**: `./output` (created automatically)
- **Vector Database**: `./output/chroma_db/` (created automatically)

## Examples

### Example 1: Process and Query a PDF

```python
from raganything import RAGAnything

# Initialize
rag = RAGAnything()

# Process document
rag.process_document_complete("documents/research-paper.pdf")

# Query
result = rag.query("What are the main findings?")
print(result["answer"])
```

### Example 2: Process Multiple Documents

```python
from raganything.batch import BatchProcessor
from raganything import RAGAnything

batch = BatchProcessor(raganything=RAGAnything())

# Process all documents in directory
results = batch.process_directory(
    directory="documents",
    pattern="*.pdf",
    recursive=True
)
```

### Example 3: Use Custom Output Directory

```python
from raganything import RAGAnything

rag = RAGAnything()

# Process with custom output directory
result = rag.process_document_complete(
    file_path="documents/document.pdf",
    output_dir="./custom_output"
)
```

## Web UI (Optional)

If Streamlit is installed:

```bash
# Install Streamlit
pip install streamlit

# Launch web UI
python -m raganything.cli ui

# Or directly
streamlit run raganything/webui/streamlit_app.py
```

The web UI allows you to:
- Upload documents through the browser
- Process documents interactively
- Chat with your documents
- View processing results

## Tips

1. **Organize Documents**: Keep your documents organized in the `documents/` directory
2. **Check Output**: Processed outputs are saved in `output/` directory
3. **Vector Database**: The vector database persists in `output/chroma_db/` - you can reuse it across sessions
4. **Batch Processing**: Use batch processing for multiple documents to save time
5. **Document IDs**: Use meaningful `doc_id` values to track documents in the vector database

