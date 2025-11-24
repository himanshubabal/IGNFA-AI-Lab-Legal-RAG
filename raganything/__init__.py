"""
RAG-Anything: All-in-One RAG Framework for Multimodal Document Processing.

This package provides a comprehensive framework for processing multimodal documents
(PDFs, Office documents, images, tables, equations) and enabling semantic search
and Q&A over the extracted content.
"""

__version__ = "0.1.0"
__author__ = "RAG-Anything Contributors"

from raganything.raganything import RAGAnything
from raganything.config import get_config, Config
from raganything.parser import ParserFactory, MinerUParser, DoclingParser
from raganything.processor import ContentProcessor, TextChunker, EmbeddingGenerator
from raganything.query import RAGQuery
from raganything.modalprocessors import (
    ImageProcessor,
    TableProcessor,
    EquationProcessor,
    GenericProcessor,
    get_processor,
)

__all__ = [
    "RAGAnything",
    "get_config",
    "Config",
    "ParserFactory",
    "MinerUParser",
    "DoclingParser",
    "ContentProcessor",
    "TextChunker",
    "EmbeddingGenerator",
    "RAGQuery",
    "ImageProcessor",
    "TableProcessor",
    "EquationProcessor",
    "GenericProcessor",
    "get_processor",
]

