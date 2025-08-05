"""
Rate limiting middleware for API endpoints.
"""
import time
from typing import Dict, Optional

import redis.asyncio as redis  # type: ignore
from fastapi import HTTPException, Request, status

from neurosync.core.logging.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Redis-based rate limiter."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_limit: int = 60,
        window_size: int = 60,
    ):
        self.redis_client = redis.from_url(redis_url)
        self.default_limit = default_limit
        self.window_size = window_size

    async def is_allowed(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed based on rate limits.
        Returns (is_allowed, headers_dict)
        """
        limit = limit or self.default_limit
        window = window or self.window_size
        now = int(time.time())

        try:
            # Use sliding window log approach
            pipe = self.redis_client.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(key, 0, now - window)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Set expiration
            pipe.expire(key, window)

            results = await pipe.execute()
            current_requests = results[1]

            remaining = max(0, limit - current_requests - 1)
            reset_time = now + window

            headers = {
                "X-RateLimit-Limit": limit,
                "X-RateLimit-Remaining": remaining,
                "X-RateLimit-Reset": reset_time,
                "X-RateLimit-Window": window,
            }

            if current_requests >= limit:
                return False, headers

            return True, headers

        except Exception as e:
            logger.error(f"Rate limit check failed for key {key}: {e}")
            # Fail open - allow request if Redis is down
            return True, {"X-RateLimit-Limit": limit}

    async def get_client_key(self, request: Request) -> str:
        """Generate rate limit key for client."""
        # Try to get API key from headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    async def check_rate_limit(
        self,
        request: Request,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Dict[str, int]:
        """Check rate limit and raise HTTPException if exceeded."""
        client_key = await self.get_client_key(request)
        is_allowed, headers = await self.is_allowed(client_key, limit, window)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {client_key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": headers.get("X-RateLimit-Reset", 0)
                    - int(time.time()),
                },
                headers=headers,
            )

        return headers

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Rate limiter health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to close rate limiter Redis connection: {e}")


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Check rate limit
        try:
            headers = await self.rate_limiter.check_rate_limit(request)
            response = await call_next(request)

            # Add rate limit headers to response
            for key, value in headers.items():
                response.headers[key] = str(value)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue without rate limiting if there's an error
            return await call_next(request)
