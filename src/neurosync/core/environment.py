"""
Environment setup for NeuroSync to handle ML library threading conflicts.

This module configures the runtime environment to prevent threading conflicts
between different machine learning libraries, particularly PyTorch, FAISS,
and various BLAS implementations. These conflicts can cause segmentation
faults and performance degradation in production environments.

Key Issues Addressed:
    - OpenMP library conflicts between PyTorch and FAISS
    - BLAS threading conflicts with sentence-transformers
    - Memory corruption from concurrent tensor operations
    - Performance degradation from thread over-subscription
    - Segmentation faults during embedding operations

Environment Variables Configured:
    - KMP_DUPLICATE_LIB_OK: Allows multiple OpenMP libraries
    - OMP_NUM_THREADS: Limits OpenMP thread pool size
    - MKL_NUM_THREADS: Controls Intel MKL threading
    - OPENBLAS_NUM_THREADS: Manages OpenBLAS parallelism
    - VECLIB_MAXIMUM_THREADS: Limits macOS Accelerate framework
    - NUMEXPR_NUM_THREADS: Controls NumExpr threading

CRITICAL: This module must be imported before any ML libraries to prevent
runtime conflicts. The setup_threading_environment() function is called
automatically during module import to ensure proper initialization.

Threading Strategy:
    - Single-threaded BLAS operations to prevent conflicts
    - Application-level parallelism for batch processing
    - Explicit thread pool management for async operations
    - Memory-safe tensor operations with proper cleanup

Platform Considerations:
    - macOS: Accelerate framework conflicts with OpenMP
    - Linux: Multiple BLAS library conflicts common
    - Windows: Intel MKL conflicts with other implementations
    - Docker: Container threading limits require adjustment

Example:
    >>> # Import this module FIRST, before any ML libraries
    >>> from neurosync.core.environment import setup_threading_environment
    >>>
    >>> # Now safe to import ML libraries
    >>> import torch
    >>> import faiss
    >>> from sentence_transformers import SentenceTransformer

For deployment and threading configuration, see:
    - docs/deployment-environment.md
    - docs/threading-configuration.md
    - examples/production-setup.py
"""
import os


def setup_threading_environment():
    """
    Configure environment variables to prevent ML library threading conflicts.

    Sets up the runtime environment to prevent segmentation faults and
    performance issues caused by threading conflicts between PyTorch, FAISS,
    and various BLAS implementations. This function must be called before
    importing any machine learning libraries.

    Threading Conflicts Resolved:
        - OpenMP library duplication between PyTorch and FAISS
        - BLAS threading conflicts with sentence-transformers
        - Memory corruption from concurrent tensor operations
        - Thread over-subscription causing performance degradation
        - Platform-specific threading library conflicts

    Environment Variables Set:
        KMP_DUPLICATE_LIB_OK="TRUE": Allows multiple OpenMP runtime libraries
            to coexist without runtime errors
        OMP_NUM_THREADS="1": Limits OpenMP to single thread to prevent conflicts
        MKL_NUM_THREADS="1": Restricts Intel MKL to single-threaded operation
        OPENBLAS_NUM_THREADS="1": Forces OpenBLAS single-threaded mode
        VECLIB_MAXIMUM_THREADS="1": Limits macOS Accelerate framework threads
        NUMEXPR_NUM_THREADS="1": Restricts NumExpr expression evaluation

    Performance Implications:
        - BLAS operations use single thread (application manages parallelism)
        - Vector operations may be slower but more stable
        - Memory usage is more predictable and controlled
        - Better suited for concurrent request handling
        - Eliminates non-deterministic threading behavior

    Call Order Requirements:
        This function MUST be called before importing:
        - torch, pytorch, transformers
        - faiss-cpu, faiss-gpu
        - sentence-transformers
        - numpy with MKL backend
        - scikit-learn with BLAS

    Example:
        >>> # CORRECT: Call before ML imports
        >>> from neurosync.core.environment import setup_threading_environment
        >>> setup_threading_environment()  # Called automatically on import
        >>> import torch  # Now safe
        >>>
        >>> # INCORRECT: Will cause conflicts
        >>> import torch  # Threading already initialized
        >>> setup_threading_environment()  # Too late!

    Note:
        This function is automatically called when the module is imported,
        ensuring proper initialization order. Manual calls are not required
        but are safe to perform multiple times.
    """
    # Prevent OpenMP library conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Limit threading to prevent conflicts
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Call this immediately when the module is imported
setup_threading_environment()
