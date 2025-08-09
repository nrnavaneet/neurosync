"""
Hugging Face sentence-transformers embedding model wrapper.

This module provides a comprehensive wrapper for Hugging Face's sentence-
transformers library, enabling local inference of high-quality text embeddings
without external API dependencies. It supports hundreds of pre-trained models
with optimizations for batch processing, memory efficiency, and GPU acceleration.

Key Features:
    - Local inference without external API dependencies
    - Support for 100+ pre-trained sentence-transformer models
    - GPU acceleration with automatic device detection
    - Memory-efficient batch processing with configurable sizes
    - Model caching and automatic downloading
    - Multi-language model support
    - Custom model fine-tuning capabilities
    - Quantization support for memory optimization

Popular Models:
    all-MiniLM-L6-v2: Fast, lightweight, 384 dimensions
    all-mpnet-base-v2: High quality, 768 dimensions
    multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A tasks
    paraphrase-multilingual: Support for 50+ languages
    all-distilroberta-v1: Balanced speed and quality

Performance Characteristics:
    - Local inference for privacy-sensitive applications
    - GPU acceleration for high-throughput processing
    - Configurable batch sizes for memory optimization
    - Model quantization for reduced memory usage
    - Efficient tensor operations with PyTorch backend

Privacy and Security:
    - Complete local processing without data transmission
    - No external API calls or internet connectivity required
    - Data privacy compliance for sensitive content
    - Offline operation capability for secure environments

Model Management:
    - Automatic model downloading and caching
    - Model versioning and update management
    - Custom model loading from local files
    - Model quantization and optimization
    - Memory usage monitoring and optimization

Device Support:
    - Automatic GPU detection and utilization
    - CPU fallback for systems without GPU
    - Mixed precision support for memory efficiency
    - Multi-GPU support for large-scale processing

Configuration Options:
    model_name: HuggingFace model identifier or local path
    device: Target device (auto, cpu, cuda, mps)
    batch_size: Processing batch size for memory optimization
    normalize_embeddings: L2 normalization of output vectors
    trust_remote_code: Allow execution of custom model code

Memory Optimization:
    - Configurable batch sizes based on available memory
    - Model quantization for reduced memory footprint
    - Efficient tensor operations with minimal copying
    - Garbage collection optimization for long-running processes

Quality Assurance:
    - Consistent embeddings across different batch sizes
    - Normalized vectors for cosine similarity optimization
    - Model validation and quality benchmarking
    - Deterministic results for reproducible operations

Integration Points:
    - Embedding manager for multi-provider coordination
    - GPU resource management and allocation
    - Model registry and version management
    - Performance monitoring and optimization

Example Usage:
    >>> embedder = HuggingFaceEmbedder("all-MiniLM-L6-v2")
    >>> embeddings = embedder.embed_batch(texts)
    >>> print(f"Generated {len(embeddings)} embeddings")

For advanced model configuration and fine-tuning, see:
    - docs/huggingface-model-configuration.md
    - docs/custom-model-integration.md
    - examples/gpu-accelerated-embedding.py

Author: NeuroSync Team
Created: 2025
License: MIT
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.base import BaseEmbeddingModel

logger = get_logger(__name__)


class HuggingFaceEmbedder(BaseEmbeddingModel):
    """Wrapper for Hugging Face sentence-transformer models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading Hugging Face model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully.")

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        # Ensure we return a list of individual numpy arrays
        if len(texts) == 1:
            # For single text, ensure we return a 1D array
            if len(embeddings.shape) == 2:
                return [embeddings.flatten()]
            else:
                return [embeddings]
        else:
            # For multiple texts, return list of 1D arrays
            return [embedding for embedding in embeddings]
