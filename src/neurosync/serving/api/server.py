"""
FastAPI server for NeuroSync API.
"""
import hashlib
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from neurosync.core.config.settings import Settings
from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.manager import EmbeddingManager
from neurosync.serving.api.middleware import RateLimiter
from neurosync.serving.llm.manager import LLMManager
from neurosync.serving.rag.cache import CacheManager
from neurosync.serving.rag.retriever import Retriever
from neurosync.storage.vector_store.manager import VectorStoreManager

logger = get_logger(__name__)


# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1)
    model: Optional[str] = Field(None, description="LLM model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    stream: bool = Field(False, description="Enable streaming response")
    use_cache: bool = Field(True, description="Use cached responses")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    context_limit: int = Field(
        8000, description="Maximum context length", ge=1000, le=32000
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, description="Number of results", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved sources")
    cached: bool = Field(..., description="Whether response was cached")
    response_time: float = Field(..., description="Response time in seconds")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query_time: float = Field(..., description="Query time in seconds")


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    version: str = "1.0.0"


# Global state for dependency injection
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting NeuroSync API server...")

    # Initialize services
    settings = Settings()

    # Initialize components
    app_state["settings"] = settings
    app_state["start_time"] = time.time()
    app_state["llm_manager"] = LLMManager(settings)
    app_state["cache_manager"] = CacheManager(
        redis_url=getattr(settings, "redis_url", "redis://localhost:6379")
    )
    app_state["rate_limiter"] = RateLimiter(
        redis_url=getattr(settings, "redis_url", "redis://localhost:6379")
    )

    # Convert settings to dict for VectorStoreManager
    vector_config = {
        "vector_store_type": "faiss",
        "index_path": "./vector_store",
        "dimension": 384,
    }
    app_state["vector_store_manager"] = VectorStoreManager(vector_config)

    # Create embedding manager
    embedding_config = {
        "type": "huggingface",
        "model_name": "all-MiniLM-L6-v2",
        "enable_monitoring": True,
    }
    app_state["embedding_manager"] = EmbeddingManager(embedding_config)

    app_state["retriever"] = Retriever(
        embedding_manager=app_state["embedding_manager"],
        vector_store_manager=app_state["vector_store_manager"],
    )

    # Health check
    logger.info("Performing health checks...")
    health_status = await check_services_health()
    if not health_status["vector_store"]:
        logger.warning("Vector store not available - some features may not work")

    logger.info("NeuroSync API server started successfully")
    yield

    # Cleanup
    logger.info("Shutting down NeuroSync API server...")
    try:
        cache_manager = app_state.get("cache_manager")
        if cache_manager and hasattr(cache_manager, "close"):
            await cache_manager.close()
    except Exception as e:
        logger.warning(f"Error closing cache manager: {e}")

    try:
        rate_limiter = app_state.get("rate_limiter")
        if rate_limiter and hasattr(rate_limiter, "close"):
            await rate_limiter.close()
    except Exception as e:
        logger.warning(f"Error closing rate limiter: {e}")
    logger.info("NeuroSync API server stopped")


# Create FastAPI app
app = FastAPI(
    title="NeuroSync API",
    description="AI-powered document processing and retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
async def get_llm_manager() -> LLMManager:
    """Get LLM manager dependency."""
    return app_state["llm_manager"]


async def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    return app_state["cache_manager"]


async def get_retriever() -> Retriever:
    """Get retriever dependency."""
    return app_state["retriever"]


async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter dependency."""
    return app_state["rate_limiter"]


# Helper functions
def generate_query_hash(query: str, **kwargs) -> str:
    """Generate hash for query caching."""
    content = f"{query}:{sorted(kwargs.items())}"
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()


async def check_services_health() -> Dict[str, bool]:
    """Check health of all services."""
    health_status = {}

    try:
        health_status["cache"] = await app_state["cache_manager"].health_check()
    except Exception:
        health_status["cache"] = False

    try:
        health_status["rate_limiter"] = await app_state["rate_limiter"].health_check()
    except Exception:
        health_status["rate_limiter"] = False

    try:
        # Test vector store
        _ = await app_state["vector_store_manager"].list_collections()
        health_status["vector_store"] = True
    except Exception:
        health_status["vector_store"] = False

    try:
        # Test LLM
        test_response = await app_state["llm_manager"].generate("Test", max_tokens=5)
        health_status["llm"] = bool(test_response)
    except Exception:
        health_status["llm"] = False

    return health_status


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "NeuroSync API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = await check_services_health()

    overall_status = "healthy" if all(services.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        services=services,
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    http_request: Request = None,
):
    """Search documents in the vector store."""
    start_time = time.time()

    # Check rate limit
    if http_request:
        await rate_limiter.check_rate_limit(http_request, limit=30, window=60)

    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
        )

        query_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total=len(results),
            query_time=query_time,
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_manager: LLMManager = Depends(get_llm_manager),
    retriever: Retriever = Depends(get_retriever),
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    http_request: Request = None,
):
    """Chat with documents using RAG."""
    start_time = time.time()

    # Check rate limit
    if http_request:
        await rate_limiter.check_rate_limit(http_request, limit=20, window=60)

    # Generate cache key
    cache_params = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_k": request.top_k,
    }
    query_hash = generate_query_hash(request.query, **cache_params)

    # Check cache if enabled
    cached_response = None
    if request.use_cache:
        cached_response = await cache_manager.get_response(query_hash)
        if cached_response:
            return ChatResponse(
                response=cached_response,
                model=request.model or "cached",
                usage={"cached": True},
                sources=[],
                cached=True,
                response_time=time.time() - start_time,
            )

    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
        )

        # Build context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            content = doc.metadata.get("content", "No content available")
            context_parts.append(f"Content: {content}")
            if doc.metadata.get("source"):
                context_parts.append(f"Source: {doc.metadata['source']}")
            context_parts.append("---")

        context = "\n".join(context_parts)

        # Build prompt
        prompt = (
            "Based on the following context, answer the user's question. "
            "If the answer cannot be found in the context, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.query}\n\n"
            "Answer:"
        )

        # Generate response
        response = await llm_manager.generate(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Cache response if enabled
        if request.use_cache and response:
            await cache_manager.set_response(query_hash, response)

        response_time = time.time() - start_time

        return ChatResponse(
            response=response or "I'm sorry, I couldn't generate a response.",
            model=llm_manager.current_model or "unknown",
            usage={"tokens": len(response.split()) if response else 0},
            sources=retrieved_docs,
            cached=False,
            response_time=response_time,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm_manager: LLMManager = Depends(get_llm_manager),
    retriever: Retriever = Depends(get_retriever),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    http_request: Request = None,
):
    """Stream chat response."""
    # Check rate limit
    if http_request:
        await rate_limiter.check_rate_limit(http_request, limit=10, window=60)

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Retrieve documents
            retrieved_docs = retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
            )

            # Build context
            context_parts = []
            for doc in retrieved_docs:
                content = doc.metadata.get("content", "No content available")
                context_parts.append(f"Content: {content}")
                if doc.metadata.get("source"):
                    context_parts.append(f"Source: {doc.metadata['source']}")
                context_parts.append("---")

            context = "\n".join(context_parts)

            # Build prompt
            prompt = (
                "Based on the following context, answer the user's question. "
                "If the answer cannot be found in the context, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {request.query}\n\n"
                "Answer:"
            )

            # Stream response
            async for chunk in llm_manager.generate_stream(
                prompt=prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/stream-server-sent-events",
    )


@app.get("/models")
async def list_models(llm_manager: LLMManager = Depends(get_llm_manager)):
    """List available LLM models."""
    try:
        models = llm_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@app.get("/stats")
async def get_stats(cache_manager: CacheManager = Depends(get_cache_manager)):
    """Get API statistics."""
    try:
        cache_stats = await cache_manager.get_cache_stats()
        return {
            "cache": cache_stats,
            "uptime": time.time() - app_state.get("start_time", time.time()),
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    # Rate limiting is handled in individual endpoints
    return await call_next(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "neurosync.serving.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
