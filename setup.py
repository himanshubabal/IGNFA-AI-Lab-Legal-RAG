"""Setup script for AI Lab IGNFA - Legal RAG System package."""

from setuptools import setup, find_packages

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "All-in-One RAG Framework for Multimodal Document Processing"

setup(
    name="raganything",
    version="0.1.0",
    author="AI Lab IGNFA",
    author_email="",
    description="AI Lab IGNFA - Legal RAG System: Legal Document Processing and Q&A System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HKUDS/RAG-Anything",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "chromadb>=0.5.0",
        "pydantic>=2.0.0",
        "aiofiles>=23.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "image": ["Pillow>=10.0.0"],
        "text": ["ReportLab>=4.0.0"],
        "all": ["Pillow>=10.0.0", "ReportLab>=4.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=3.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "ui": ["streamlit>=1.28.0"],
    },
    entry_points={
        "console_scripts": [
            "raganything=raganything.cli:main",
        ],
    },
)

