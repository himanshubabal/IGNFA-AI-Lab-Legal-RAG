# New Features Summary

## ✅ Implemented Features

### 1. Smart Document Processing
- **Document Tracker**: Tracks which documents have been processed
- **Smart Processor**: Only processes new/updated documents
- **Automatic Cleanup**: Removes tracking for deleted documents

### 2. CLI Commands

#### Process All Documents
```bash
python -m raganything.cli process-all
```

Processes all documents in the `documents/` directory:
- ✅ New documents: Automatically detected and processed
- ✅ Updated documents: Reprocessed if file was modified
- ✅ Removed documents: Cleaned up from tracking
- ✅ Unchanged documents: Skipped (not reprocessed)

Options:
- `--force`: Force reprocess all documents
- `--documents-dir DIR`: Specify documents directory (default: `documents`)

#### Document Status
```bash
python -m raganything.cli status
```

Shows:
- Total files in documents directory
- Processed documents (with chunk counts)
- Unprocessed documents
- Removed documents (tracked but missing)

### 3. Enhanced Web UI

#### Launch Web UI
```bash
# Option 1: Using launch script (recommended)
python run_ui.py

# Option 2: Using CLI
python -m raganything.cli ui
```

#### Web UI Features

**📄 Documents Tab:**
- ✅ Upload documents directly to `documents/` directory
- ✅ Auto-process on upload (if enabled)
- ✅ View all processed documents with details
- ✅ View unprocessed documents
- ✅ Remove documents (file + tracking)
- ✅ Process individual documents
- ✅ Refresh and process all documents

**💬 Chat Tab:**
- ✅ Chat with your processed documents
- ✅ View sources for answers
- ✅ Chat history

**📊 Status Tab:**
- ✅ Detailed document status
- ✅ Processed documents table
- ✅ Unprocessed documents list
- ✅ Removed documents list

**⚙️ Sidebar:**
- ✅ Configuration settings
- ✅ Document status metrics
- ✅ Auto-processing toggle
- ✅ Process all button

### 4. Document Tracking

The system maintains a tracker file at `output/.document_tracker.json` that stores:
- File paths (normalized)
- Document IDs
- Number of chunks
- File metadata (size, modified time)
- Processing timestamp

## Usage Examples

### Process All Documents
```bash
# Process all new/updated documents
python -m raganything.cli process-all

# Force reprocess everything
python -m raganything.cli process-all --force
```

### Check Status
```bash
python -m raganything.cli status
```

### Launch Web UI
```bash
# Automatically processes documents on startup
python run_ui.py
```

### Web UI Workflow
1. Launch: `python run_ui.py`
2. Upload documents via web interface
3. Documents are automatically processed (if auto-process enabled)
4. Chat with your documents
5. Manage documents (view, remove, reprocess)

## How It Works

### Document Lifecycle

1. **New Document Added:**
   - Detected by comparing files in `documents/` with tracker
   - Automatically processed
   - Added to tracker

2. **Document Updated:**
   - Detected by comparing file modification time
   - Old tracking entry removed
   - Document reprocessed
   - New entry added to tracker

3. **Document Removed:**
   - Detected by comparing current files with tracker
   - Removed from tracker
   - Embeddings remain in vector store (for now)

### Tracking File

Location: `output/.document_tracker.json`

Format:
```json
{
  "processed_docs": {
    "/absolute/path/to/document.pdf": {
      "file_path": "/absolute/path/to/document.pdf",
      "doc_id": "document",
      "num_chunks": 42,
      "file_size": 1234567,
      "modified_time": 1234567890.0,
      "processed_time": "2025-01-01T12:00:00",
      "metadata": {}
    }
  },
  "last_updated": "2025-01-01T12:00:00"
}
```

## Notes

- Documents are tracked by absolute path (resolved)
- File modification time is used to detect updates
- Vector store cleanup for removed documents is a future enhancement
- The tracker file is automatically created and maintained

