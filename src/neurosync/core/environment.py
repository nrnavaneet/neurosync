"""
Environment setup for NeuroSync to handle threading conflicts between PyTorch and FAISS.
"""
import os


def setup_threading_environment():
    """
    Set up environment variables to prevent threading conflicts between PyTorch
    and FAISS.

    This must be called before importing any ML libraries to prevent
    segmentation faults caused by OpenMP library conflicts between
    sentence-transformers and FAISS.
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
