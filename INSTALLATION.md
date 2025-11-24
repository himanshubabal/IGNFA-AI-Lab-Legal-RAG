# Installation Guide

## Python Version Requirements

**Recommended: Python 3.11 or 3.12**

AI Lab IGNFA - Legal RAG System is designed to work with Python 3.11 or 3.12. Some dependencies (notably ChromaDB's `onnxruntime`) may not be available for newer Python versions (e.g., Python 3.14).

### Installing Python 3.11

#### macOS (using Homebrew)
```bash
brew install python@3.11
```

#### Linux (using apt)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

#### Windows
Download Python 3.11 from [python.org](https://www.python.org/downloads/)

### Installing Python 3.12 (Alternative)

#### macOS (using Homebrew)
```bash
brew install python@3.12
```

#### Linux (using apt)
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

#### Windows
Download Python 3.12 from [python.org](https://www.python.org/downloads/)

## Virtual Environment Setup

### Automatic Setup

Run the setup script:
```bash
./setup_venv.sh
```

The script will:
1. Detect Python 3.12 (or use available Python 3.x)
2. Create a virtual environment
3. Install dependencies
4. Install RAG-Anything in development mode

### Manual Setup

```bash
# Create virtual environment with Python 3.11 (or 3.12)
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Dependency Notes

### ChromaDB and onnxruntime

ChromaDB requires `onnxruntime`, which may not be available for all Python versions. If you encounter installation issues:

1. **Use Python 3.12** (recommended)
2. **Use Python 3.11** as an alternative
3. **Install ChromaDB without dependencies** and install them separately:
   ```bash
   pip install --no-deps chromadb==0.5.0
   pip install numpy fastapi uvicorn requests  # Install other dependencies manually
   ```

### Optional Dependencies

For extended format support:
```bash
pip install Pillow  # Extended image formats
pip install ReportLab  # Text processing
```

For web UI:
```bash
pip install streamlit
```

## Troubleshooting

### Issue: `onnxruntime` not found

**Solution**: Use Python 3.11 or 3.12. Python 3.14 is not yet supported by onnxruntime.

### Issue: ChromaDB dependency conflicts

**Solution**: Install ChromaDB 0.5.0+ which supports pydantic 2.0:
```bash
pip install "chromadb>=0.5.0"
```

### Issue: LibreOffice not found (for Office document parsing)

**Solution**: Install LibreOffice:
- macOS: `brew install --cask libreoffice`
- Linux: `sudo apt-get install libreoffice`
- Windows: Download from [LibreOffice website](https://www.libreoffice.org/download/)

## Verification

After installation, verify the setup:

```bash
# Activate virtual environment
source venv/bin/activate

# Check Python version
python --version  # Should show Python 3.11.x or 3.12.x

# Test import
python -c "from raganything import RAGAnything; print('Installation successful!')"
```

