# ContentProcessor vs SmartProcessor

## Overview

RAG-Anything has two processor classes that serve different purposes:

- **ContentProcessor**: Low-level content processing (chunking, embedding, storage)
- **SmartProcessor**: High-level document management with tracking and lifecycle management

## ContentProcessor

**Location**: `raganything/processor.py`

### Purpose
Core content processing operations - handles the actual work of chunking text, generating embeddings, and storing in vector database.

### Key Features
- ✅ Text chunking (character, sentence, paragraph strategies)
- ✅ Embedding generation (OpenAI API integration)
- ✅ Vector storage (ChromaDB integration)
- ✅ Content statistics calculation
- ❌ No document tracking
- ❌ No lifecycle management
- ❌ No file system operations

### Key Methods

```python
# Chunk text
chunks = processor.chunk_text(text)

# Process and store content
chunk_ids = processor.process_and_store(
    content="Document text...",
    doc_id="document1",
    metadata={"source": "doc.pdf"}
)

# Get statistics
stats = processor.get_statistics(content)
```

### Usage
- Used internally by `RAGAnything`
- Processes individual content strings
- No awareness of document files or state
- Stateless - each call is independent

### Example

```python
from raganything.processor import ContentProcessor

processor = ContentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.chunk_text("Long document text...")
chunk_ids = processor.process_and_store(
    content="Document content",
    doc_id="doc1",
    metadata={"source": "document.pdf"}
)
```

---

## SmartProcessor

**Location**: `raganything/smart_processor.py`

### Purpose
Intelligent document management with tracking, change detection, and lifecycle management.

### Key Features
- ✅ Document tracking (uses `DocumentTracker`)
- ✅ Batch processing (`process_all()`)
- ✅ Change detection (new, updated, removed documents)
- ✅ File exclusion (README, hidden files, system files)
- ✅ Progress tracking
- ✅ Lifecycle management
- ✅ Status reporting

### Key Methods

```python
# Process all documents in directory
results = processor.process_all(force_reprocess=False)
# Returns: {'new': [...], 'updated': [...], 'removed': [...], 'unchanged': N, 'errors': [...]}

# Get document status
status = processor.get_document_status()
# Returns: {'total_files': N, 'processed': [...], 'unprocessed': [...], 'removed': [...]}
```

### Usage
- Used by `process-all` CLI command
- Works with document directories
- Tracks processing state in `.document_tracker.json`
- Handles document lifecycle (add, update, remove)

### Example

```python
from raganything.smart_processor import SmartProcessor

processor = SmartProcessor(documents_dir="documents")
results = processor.process_all(force_reprocess=False)

print(f"New: {len(results['new'])}")
print(f"Updated: {len(results['updated'])}")
print(f"Removed: {len(results['removed'])}")
```

---

## Key Differences

| Feature | ContentProcessor | SmartProcessor |
|---------|-----------------|----------------|
| **Scope** | Single content string | Directory of documents |
| **Tracking** | ❌ No tracking | ✅ Full document tracking |
| **State** | Stateless | Stateful (uses DocumentTracker) |
| **File Management** | ❌ Doesn't handle files | ✅ Scans directories, excludes files |
| **Change Detection** | ❌ No | ✅ Detects new/updated/removed |
| **Lifecycle** | Process once | Handles updates/removals |
| **Use Cases** | Internal processing | Batch processing, management |
| **Dependencies** | Vector store, embeddings | RAGAnything, DocumentTracker |

---

## Relationship

`SmartProcessor` uses `RAGAnything` internally, which in turn uses `ContentProcessor`:

```
SmartProcessor
  └── RAGAnything
      └── ContentProcessor (for chunking/storage)
```

### Processing Flow

1. **SmartProcessor** scans documents directory
2. **SmartProcessor** detects new/updated/removed documents
3. **SmartProcessor** calls `RAGAnything.process_document_complete()` for each document
4. **RAGAnything** uses **ContentProcessor** for chunking and storage
5. **SmartProcessor** updates **DocumentTracker** with results

---

## When to Use Which?

### Use ContentProcessor when:
- Processing individual content strings programmatically
- Building custom processing pipelines
- Need direct control over chunking/embedding
- Don't need document tracking

### Use SmartProcessor when:
- Processing multiple documents from a directory
- Need to track what's been processed
- Want automatic change detection
- Need batch processing with status reporting
- Using `process-all` CLI command

---

## Summary

- **ContentProcessor** = Low-level content processing engine
- **SmartProcessor** = High-level document management system

Both are essential parts of RAG-Anything, serving different layers of the processing pipeline.

