"""
Redis-based caching layer for embeddings and responses.
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis  # type: ignore

from neurosync.core.logging.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Redis-based cache manager for embeddings and responses."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        embedding_ttl: int = 86400,  # 24 hours
    ):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.embedding_ttl = embedding_ttl

    async def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        cache_key = f"embedding:{model_name}:{hash(text)}"
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                # Use numpy's fromstring for safe deserialization
                embedding = np.frombuffer(cached_data, dtype=np.float32)
                logger.debug(f"Cache hit for embedding: {cache_key}")
                return embedding
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        return None

    async def set_embedding(
        self, text: str, model_name: str, embedding: np.ndarray
    ) -> None:
        """Cache embedding for text."""
        cache_key = f"embedding:{model_name}:{hash(text)}"
        try:
            # Use numpy's tobytes for safe serialization
            cached_data = embedding.astype(np.float32).tobytes()
            await self.redis_client.set(cache_key, cached_data, ex=self.embedding_ttl)
            logger.debug(f"Cached embedding: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    async def get_response(self, query_hash: str) -> Optional[str]:
        """Get cached response for query."""
        cache_key = f"response:{query_hash}"
        try:
            cached_response = await self.redis_client.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for response: {cache_key}")
                return cached_response.decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to get cached response: {e}")
        return None

    async def set_response(
        self, query_hash: str, response: str, ttl: Optional[int] = None
    ) -> None:
        """Cache response for query."""
        cache_key = f"response:{query_hash}"
        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.set(cache_key, response, ex=ttl)
            logger.debug(f"Cached response: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def get_search_results(
        self, query_hash: str, top_k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        cache_key = f"search:{query_hash}:{top_k}"
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                results = json.loads(cached_data.decode("utf-8"))
                logger.debug(f"Cache hit for search results: {cache_key}")
                return results
        except Exception as e:
            logger.warning(f"Failed to get cached search results: {e}")
        return None

    async def set_search_results(
        self,
        query_hash: str,
        top_k: int,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache search results."""
        cache_key = f"search:{query_hash}:{top_k}"
        try:
            ttl = ttl or self.default_ttl
            cached_data = json.dumps(results)
            await self.redis_client.set(cache_key, cached_data, ex=ttl)
            logger.debug(f"Cached search results: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching: {pattern}")
                return deleted
        except Exception as e:
            logger.warning(f"Failed to invalidate cache pattern {pattern}: {e}")
        return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis_client.info("keyspace")
            memory_info = await self.redis_client.info("memory")
            return {
                "keyspace": info,
                "memory_usage": memory_info.get("used_memory_human", "N/A"),
                "connected_clients": memory_info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to close Redis connection: {e}")
