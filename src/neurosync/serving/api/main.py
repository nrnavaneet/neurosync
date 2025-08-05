"""
Main FastAPI application for serving the NeuroSync RAG API.
"""
import json
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from neurosync.core.logging.logger import get_logger
from neurosync.processing.embedding.manager import EmbeddingManager
from neurosync.serving.llm.manager import LLMManager
from neurosync.serving.rag.prompts import PromptManager
from neurosync.serving.rag.retriever import Retriever
from neurosync.storage.vector_store.manager import VectorStoreManager

# --- App Initialization ---
app = FastAPI(
    title="NeuroSync RAG API",
    description="A real-time API for Retrieval-Augmented Generation.",
    version="1.0.0",
)
logger = get_logger(__name__)

# --- In-memory Cache for Global Objects ---
# In a real production app, this would be managed more robustly.
app.state.cache = {}


# --- Dependency Injection Functions ---
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_embedding_manager() -> EmbeddingManager:
    if "embedding_manager" not in app.state.cache:
        config = load_config("configs/embedding_config.json")
        app.state.cache["embedding_manager"] = EmbeddingManager(config)
    return app.state.cache["embedding_manager"]


def get_vector_store_manager(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
) -> VectorStoreManager:
    if "vector_store_manager" not in app.state.cache:
        config = load_config("configs/vs_config.json")
        config["dimension"] = embedding_manager.get_dimension()
        app.state.cache["vector_store_manager"] = VectorStoreManager(config)
    return app.state.cache["vector_store_manager"]


def get_llm_manager() -> LLMManager:
    if "llm_manager" not in app.state.cache:
        config = load_config("configs/llm_config.json")
        app.state.cache["llm_manager"] = LLMManager(config)
    return app.state.cache["llm_manager"]


def get_retriever(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_store_manager: VectorStoreManager = Depends(get_vector_store_manager),
) -> Retriever:
    return Retriever(embedding_manager, vector_store_manager)


def get_prompt_manager() -> PromptManager:
    return PromptManager(template_dir="src/neurosync/serving/rag/templates")


# Note: Redis caching temporarily disabled for simpler setup
# async def get_redis_client() -> redis.Redis:
#     if "redis" not in app.state.cache:
#         app.state.cache["redis"] = redis.from_url("redis://localhost")
#     return app.state.cache["redis"]


# --- Pydantic Models for API ---
class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class HealthCheckResponse(BaseModel):
    status: str
    version: str


# --- API Endpoints ---
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Performs a health check of the API."""
    return {"status": "ok", "version": app.version}


@app.post("/rag/stream")
async def rag_stream(
    request: RAGQueryRequest,
    retriever: Retriever = Depends(get_retriever),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    llm_manager: LLMManager = Depends(get_llm_manager),
    # Redis caching temporarily disabled
    # redis_client: redis.Redis = Depends(get_redis_client),
):
    """
    Performs Retrieval-Augmented Generation and streams the response.
    """

    # Note: Caching temporarily disabled for simpler setup
    # 1. Check Cache
    # cached_response = await redis_client.get(request.query)
    # if cached_response:
    #     async def cached_generator():
    #         yield cached_response.decode()
    #     return StreamingResponse(cached_generator(), media_type="text/plain")

    # 2. Retrieve Context
    context_chunks = retriever.retrieve(request.query, request.top_k)

    if not context_chunks:

        async def no_context_generator():
            yield "I could not find any relevant information to answer your question."

        return StreamingResponse(no_context_generator(), media_type="text/plain")

    # 3. Construct Prompt
    prompt = prompt_manager.render(
        "rag_prompt.jinja", {"query": request.query, "context_chunks": context_chunks}
    )

    # 4. Generate and Stream Response
    response_generator = llm_manager.generate_stream(prompt)

    # Simple streaming without caching
    async def stream_wrapper() -> AsyncGenerator[str, None]:
        async for token in response_generator:
            yield token

    return StreamingResponse(stream_wrapper(), media_type="text/plain")


# --- Application Runner ---
def start_server():
    """Function to start the Uvicorn server."""
    # Create configurations:
    # mkdir configs
    # neurosync vector-store create-config embedding \
    #   --output configs/embedding_config.json
    # neurosync vector-store create-config vector-store \
    #   --output configs/vs_config.json
    # neurosync serving create-config --output configs/llm_config.json

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    start_server()
