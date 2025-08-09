"""
NeuroSync Storage Module - Vector Database and Storage Management.

This module provides comprehensive storage solutions for vector embeddings,
metadata, and application data. It implements a unified interface for
multiple storage backends with optimizations for similarity search,
scalability, and production deployment.

Key Features:
    - Multi-backend vector storage (FAISS, Qdrant, Chroma)
    - Optimized similarity search with configurable algorithms
    - Metadata filtering and faceted search capabilities
    - Backup and versioning system for disaster recovery
    - Performance monitoring and optimization tools
    - Distributed storage support for horizontal scaling
    - Index compression and memory optimization
    - Real-time updates and incremental indexing

Storage Backends:
    - FAISS: High-performance CPU/GPU similarity search
    - Qdrant: Production-scale vector database with filtering
    - Chroma: Developer-friendly embedded vector store
    - In-Memory: Fast development and testing backend

Vector Operations:
    - Index creation and management
    - Batch and streaming vector insertion
    - Similarity search with distance metrics
    - Metadata filtering and complex queries
    - Index optimization and compression
    - Backup and restoration procedures

Search Capabilities:
    - Cosine similarity, L2 distance, dot product metrics
    - Approximate nearest neighbor (ANN) algorithms
    - Exact search for small datasets
    - Hybrid search combining vector and metadata filters
    - Result ranking and score normalization
    - Query expansion and refinement

The storage system provides:
    - Automatic backend selection based on dataset size
    - Performance optimization for different use cases
    - Seamless migration between storage backends
    - Comprehensive monitoring and alerting
    - Data integrity and consistency guarantees

Example:
    >>> from neurosync.storage.vector_store.manager import VectorStoreManager
    >>>
    >>> # Initialize vector store with configuration
    >>> config = {
    ...     "backend": "faiss",
    ...     "dimension": 384,
    ...     "index_type": "hnsw"
    ... }
    >>> store = VectorStoreManager(config)
    >>>
    >>> # Add vectors with metadata
    >>> vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    >>> metadata = [{"doc_id": "1"}, {"doc_id": "2"}]
    >>> store.add_vectors(vectors, metadata)
    >>>
    >>> # Similarity search
    >>> query_vector = [0.15, 0.25, ...]
    >>> results = store.search(query_vector, k=5)
    >>> print(f"Found {len(results)} similar vectors")

For configuration examples and performance tuning, see:
    - docs/vector-storage.md
    - docs/search-optimization.md
    - examples/storage-backends.py
"""
