"""
NeuroSync API Server
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neurosync.core.config.settings import settings
from neurosync.core.logging.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info(f"Starting {settings.APP_NAME} API server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    yield
    logger.info("Shutting down NeuroSync API server")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Native ETL Pipeline for RAG and LLM Applications",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "NeuroSync API",
        "version": settings.APP_VERSION,
        "status": "active",
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "neurosync-api",
        "environment": settings.ENVIRONMENT,
    }
