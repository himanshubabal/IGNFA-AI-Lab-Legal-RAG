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

## Custom Prompts

You can customize the system prompt used by the LLM by creating a `prompt.md` file in the project root.

### Creating a Custom Prompt

1. Create a file named `prompt.md` in the project root directory
2. Write your custom system prompt in the file
3. The prompt will be automatically loaded and used for all queries

Example `prompt.md`:

```markdown
# Custom System Prompt

You are a helpful assistant that answers questions based on provided context from documents.

## Instructions

- Provide accurate answers based on the context
- Cite sources when possible
- Say "I don't know" if the answer is not in the context
- Be concise and clear
- Use the context to provide detailed and accurate responses
```

### Managing Prompts in Web UI

The web UI provides a section in the sidebar to:
- View the current prompt (default or custom)
- Edit the prompt directly in the UI
- Save changes to `prompt.md`
- Reload the prompt after editing

### Configuration

You can specify a custom prompt file path in `.env`:

```env
PROMPT_FILE=prompt.md  # Default: prompt.md
```

The system will look for the prompt file in:
1. The path specified in `PROMPT_FILE`
2. The project root directory
3. The current working directory

## Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=your_base_url  # Optional

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo        # Options: gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo, etc.
LLM_TEMPERATURE=0.7            # 0.0-2.0, controls randomness
LLM_TOP_P=1.0                 # 0.0-1.0, nucleus sampling
LLM_MAX_TOKENS=2000            # Maximum tokens in response (optional)

# Query Configuration
QUERY_N_RESULTS=5              # Number of context chunks to retrieve
QUERY_MAX_CONTEXT_LENGTH=2000  # Maximum context length in characters

# Prompt Configuration
PROMPT_FILE=prompt.md          # Path to custom system prompt file (optional)

# Output Configuration
OUTPUT_DIR=./output             # Default: ./output

# Parser Configuration
PARSER=mineru                   # Options: mineru or docling
PARSE_METHOD=auto               # Options: auto, ocr, or txt
MINERU_OUTPUT_FLAG_SPAN=true   # Output verification files (span.pdf, layout.pdf, etc.) - default: true
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

