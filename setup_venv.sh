#!/bin/bash
# Setup script for RAG-Anything virtual environment

set -e

echo "Setting up Python virtual environment for RAG-Anything..."

# Check for Python 3.12
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "Found Python 3.12"
    PYTHON_VERSION=$($PYTHON_CMD --version)
    echo "Using: $PYTHON_VERSION"
elif python3 --version | grep -q "3.12"; then
    PYTHON_CMD="python3"
    echo "Found Python 3.12 via python3"
    PYTHON_VERSION=$(python3 --version)
    echo "Using: $PYTHON_VERSION"
else
    echo "Warning: Python 3.12 not found. Using available Python 3 version."
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version)
    echo "Using: $PYTHON_VERSION"
    echo "Note: Some dependencies may require Python 3.12. Consider installing Python 3.12."
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing RAG-Anything in development mode..."
pip install -e .

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

