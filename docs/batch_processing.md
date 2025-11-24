# Batch Processing Guide

This guide explains how to use RAG-Anything for batch processing multiple documents.

## Overview

Batch processing allows you to efficiently process multiple documents at once, with progress tracking, error handling, and parallel processing support.

## Basic Usage

### Processing Multiple Files

```python
from raganything.batch import BatchProcessor
from raganything import RAGAnything

# Initialize batch processor
batch_processor = BatchProcessor(
    raganything=RAGAnything(parser="mineru")
)

# List of files to process
file_paths = [
    "document1.pdf",
    "document2.pdf",
    "document3.docx",
]

# Process batch
results = batch_processor.process_batch(
    file_paths=file_paths,
    display_progress=True,
    continue_on_error=True,
)

# Check results
print(f"Successful: {results['successful']}")
print(f"Failed: {results['failed']}")
print(f"Total chunks: {results['total_chunks']}")
```

### Processing a Directory

```python
# Process all PDFs in a directory
results = batch_processor.process_directory(
    directory="./documents",
    pattern="*.pdf",
    recursive=True,
)
```

## Advanced Features

### Error Handling

The batch processor can continue processing even if some files fail:

```python
results = batch_processor.process_batch(
    file_paths=file_paths,
    continue_on_error=True,  # Continue on errors
)

# Check errors
for error in results['errors']:
    print(f"Error in {error['file_path']}: {error['error']}")
```

### Custom Configuration

You can customize the RAG-Anything instance used for batch processing:

```python
rag = RAGAnything(
    parser="docling",
    parse_method="auto",
    chunk_size=1500,
    chunk_overlap=300,
)

batch_processor = BatchProcessor(raganything=rag)
```

## Batch Parser

For parsing-only operations (without vectorization), use `BatchParser`:

```python
from raganything.batch_parser import BatchParser

batch_parser = BatchParser(
    parser="mineru",
    parse_method="auto",
    max_workers=4,  # Parallel processing
)

results = batch_parser.parse_batch(
    file_paths=file_paths,
    output_dir="./parsed_output",
)
```

## Performance Tips

1. **Parallel Processing**: Use `BatchParser` with `max_workers` for faster parsing
2. **Error Recovery**: Set `continue_on_error=True` to process all files
3. **Progress Tracking**: Enable `display_progress=True` for monitoring

## Example

See `examples/batch_processing_example.py` for a complete example.

