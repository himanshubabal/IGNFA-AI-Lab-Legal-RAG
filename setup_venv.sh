#!/bin/bash
# Setup script for RAG-Anything virtual environment

set -e

echo "Setting up Python virtual environment for AI Lab IGNFA - Legal RAG System..."

# Check for Python 3.11 (preferred) or 3.12
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Found Python 3.11"
    PYTHON_VERSION=$($PYTHON_CMD --version)
    echo "Using: $PYTHON_VERSION"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "Found Python 3.12"
    PYTHON_VERSION=$($PYTHON_CMD --version)
    echo "Using: $PYTHON_VERSION"
elif python3 --version | grep -q "3.11"; then
    PYTHON_CMD="python3"
    echo "Found Python 3.11 via python3"
    PYTHON_VERSION=$(python3 --version)
    echo "Using: $PYTHON_VERSION"
elif python3 --version | grep -q "3.12"; then
    PYTHON_CMD="python3"
    echo "Found Python 3.12 via python3"
    PYTHON_VERSION=$(python3 --version)
    echo "Using: $PYTHON_VERSION"
else
    echo "Warning: Python 3.11 or 3.12 not found. Using available Python 3 version."
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version)
    echo "Using: $PYTHON_VERSION"
    echo "Note: Python 3.11 or 3.12 is recommended for best compatibility."
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
echo "Installing AI Lab IGNFA - Legal RAG System in development mode..."
pip install -e .

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

