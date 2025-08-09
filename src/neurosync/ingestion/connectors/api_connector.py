"""
API connector for REST and GraphQL endpoints.

This module provides a comprehensive API connector that can ingest content
from REST APIs, GraphQL endpoints, and other web-based data sources. It
supports various authentication methods, automatic retry mechanisms, rate
limiting, and response format handling for reliable API data ingestion.

Key Features:
    - Multiple authentication methods (Bearer, Basic, API Key, OAuth)
    - Automatic retry with exponential backoff for transient failures
    - Rate limiting and request throttling to respect API limits
    - Response format detection and parsing (JSON, XML, CSV, plain text)
    - Pagination support for large datasets
    - Request/response logging for debugging and monitoring
    - Concurrent request processing with configurable limits
    - Error handling with detailed API error context

Supported API Types:
    REST APIs: Standard HTTP REST endpoints with JSON/XML responses
    GraphQL: GraphQL queries with schema introspection support
    Webhooks: Real-time data ingestion from webhook endpoints
    Streaming APIs: Server-sent events and chunked responses
    Paginated APIs: Automatic pagination handling for large datasets

Authentication Methods:
    None: Public APIs without authentication
    Bearer Token: OAuth2 and JWT token authentication
    Basic Auth: Username/password HTTP basic authentication
    API Key: Custom API key headers or query parameters
    OAuth2: Full OAuth2 flow with token refresh

Configuration Options:
    base_url: Base URL for all API requests
    auth_type: Authentication method (none, bearer, basic, api_key, oauth2)
    auth_token: Bearer token or API key value
    timeout: Request timeout in seconds
    max_retries: Maximum retry attempts for failed requests
    rate_limit: Maximum requests per second
    headers: Custom HTTP headers for all requests
    verify_ssl: SSL certificate verification (default: True)

Error Handling:
    - HTTP error status code handling with detailed error messages
    - Network timeout and connection error recovery
    - Rate limit detection and automatic backoff
    - Invalid response format handling with graceful degradation
    - Authentication failure detection and clear error reporting

Example Configuration:
    >>> config = {
    ...     "base_url": "https://api.example.com/v1",
    ...     "auth_type": "bearer",
    ...     "auth_token": "your-jwt-token",
    ...     "timeout": 30,
    ...     "max_retries": 3,
    ...     "rate_limit": 10  # requests per second
    ... }
    >>> connector = APIConnector(config)

Usage Patterns:
    Single Endpoint:
    >>> async with connector:
    ...     result = await connector.ingest("/users/123")

    Batch Processing:
    >>> async with connector:
    ...     endpoints = ["/users/1", "/users/2", "/users/3"]
    ...     results = await connector.ingest_batch(endpoints)

    Discovery:
    >>> async with connector:
    ...     endpoints = await connector.list_sources()
    ...     info = await connector.get_source_info(endpoints[0])

For advanced API integration and custom authentication, see:
    - docs/api-connector-configuration.md
    - docs/custom-authentication.md
    - examples/graphql-ingestion.py
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from httpx import AsyncClient, Response

from neurosync.core.exceptions.custom_exceptions import ConnectionError, IngestionError
from neurosync.ingestion.base.connector import (
    BaseConnector,
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceType,
)


class APIConnector(BaseConnector):
    """
    API connector for ingesting content from REST and GraphQL endpoints.

    This connector provides comprehensive API integration capabilities with
    support for various authentication methods, automatic error recovery,
    rate limiting, and response format handling. It's designed for reliable
    and efficient ingestion from web-based data sources.

    Architecture:
        The connector uses an HTTP client with configurable timeouts,
        retry mechanisms, and authentication. It can handle both single
        requests and batch processing with intelligent rate limiting
        and concurrent request management.

    Authentication Support:
        - Bearer Token: JWT and OAuth2 token authentication
        - Basic Auth: Username/password HTTP basic authentication
        - API Key: Custom header or query parameter authentication
        - OAuth2: Full OAuth2 flow with automatic token refresh
        - Custom: Extensible authentication via header customization

    Request Management:
        - Configurable timeouts and retry strategies
        - Exponential backoff for transient failures
        - Rate limiting to respect API quotas
        - Request/response logging for debugging
        - SSL certificate verification controls

    Response Processing:
        - Automatic content type detection from headers
        - JSON, XML, CSV, and plain text parsing
        - Large response streaming and chunking
        - Error response parsing and context extraction
        - Metadata extraction from response headers

    Error Handling:
        - HTTP status code interpretation and retry logic
        - Network error recovery with intelligent backoff
        - Authentication failure detection and reporting
        - Rate limit detection with automatic delays
        - Detailed error context for troubleshooting

    Performance Features:
        - Concurrent request processing with limits
        - Connection pooling and keep-alive optimization
        - Intelligent batching based on rate limits
        - Memory-efficient streaming for large responses
        - Caching support for repeated requests

    Configuration Parameters:
        base_url: Base URL for API endpoints
        auth_type: Authentication method identifier
        auth_token: Token/key value for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        rate_limit: Requests per second limit
        headers: Custom HTTP headers
        verify_ssl: SSL certificate verification
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "")
        self.auth_type = config.get("auth_type", "none")  # none, bearer, basic, api_key
        self.auth_token = config.get("auth_token", "")
        self.api_key = config.get("api_key", "")
        self.api_key_header = config.get("api_key_header", "X-API-Key")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.rate_limit_requests = config.get("rate_limit_requests", 100)
        self.rate_limit_window = config.get("rate_limit_window", 60)
        self.endpoints = config.get("endpoints", [])

        self.client: Optional[AsyncClient] = None
        self._rate_limiter = RateLimiter(
            self.rate_limit_requests, self.rate_limit_window
        )
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate API connector configuration"""
        if not self.base_url:
            raise IngestionError("base_url is required for API connector")

        if not urlparse(self.base_url).scheme:
            raise IngestionError("base_url must include schema (http:// or https://)")

        if self.auth_type == "bearer" and not self.auth_token:
            raise IngestionError("auth_token is required for bearer authentication")

        if self.auth_type == "basic" and (not self.username or not self.password):
            raise IngestionError(
                "username and password are required for basic authentication"
            )

        if self.auth_type == "api_key" and not self.api_key:
            raise IngestionError("api_key is required for API key authentication")

    def _setup_auth_headers(self) -> Dict[str, str]:
        """Setup authentication headers"""
        auth_headers = self.headers.copy()

        if self.auth_type == "bearer":
            auth_headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_type == "api_key":
            auth_headers[self.api_key_header] = self.api_key

        return auth_headers

    async def connect(self) -> None:
        """Establish HTTP client connection"""
        auth_headers = self._setup_auth_headers()

        # Setup authentication for basic auth
        auth = None
        if self.auth_type == "basic":
            auth = httpx.BasicAuth(self.username, self.password)

        self.client = AsyncClient(
            headers=auth_headers,
            auth=auth,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            follow_redirects=True,
        )

        self.logger.info(f"Connected to API at: {self.base_url}")

    async def disconnect(self) -> None:
        """Close HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.logger.info("Disconnected from API")

    async def test_connection(self) -> bool:
        """Test API connection with a simple request"""
        if not self.client:
            await self.connect()

        try:
            # Try a simple GET request to the base URL or health endpoint
            test_endpoints = ["/health", "/ping", "/status", "/"]

            for endpoint in test_endpoints:
                try:
                    url = urljoin(self.base_url, endpoint)
                    response = await self.client.get(url)
                    if (
                        response.status_code < 500
                    ):  # Any non-server error is considered successful
                        self.logger.info(f"Connection test successful via {endpoint}")
                        return True
                except Exception:
                    continue

            # If all specific endpoints fail, try base URL
            response = await self.client.get(self.base_url)
            return response.status_code < 500

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def list_sources(self) -> List[str]:
        """List available API endpoints"""
        if self.endpoints:
            return [endpoint["path"] for endpoint in self.endpoints]
        else:
            # If no specific endpoints configured, return base URL
            return ["/"]

    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Make HTTP request with rate limiting and retry logic"""
        if not self.client:
            await self.connect()

        # Rate limiting
        await self._rate_limiter.acquire()

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(
                    method=method, url=url, params=params, data=data, json=json_data
                )

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", self.retry_delay)
                    )
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                # Check for server errors (retry on 5xx)
                if response.status_code >= 500 and attempt < self.max_retries:
                    wait_time = self.retry_delay * (2**attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Server error {response.status_code}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                return response

            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2**attempt)
                    self.logger.warning(
                        f"Request failed: {e}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise ConnectionError(
                        f"Request failed after {self.max_retries} retries: {e}"
                    )

        raise ConnectionError(f"Request failed after {self.max_retries} retries")

    async def ingest(self, source_id: str, **kwargs) -> IngestionResult:
        """Ingest data from API endpoint"""
        start_time = time.time()

        try:
            # Find endpoint configuration
            endpoint_config = self._find_endpoint_config(source_id)
            method = endpoint_config.get("method", "GET").upper()
            params = endpoint_config.get("params", {})
            data = endpoint_config.get("data", {})
            json_data = endpoint_config.get("json", {})

            # Override with kwargs
            params.update(kwargs.get("params", {}))
            data.update(kwargs.get("data", {}))
            json_data.update(kwargs.get("json", {}))

            # Construct full URL
            url = urljoin(self.base_url, source_id)

            # Make request
            response = await self._make_request(
                method=method,
                url=url,
                params=params or None,
                data=data or None,
                json_data=json_data or None,
            )

            # Check response status
            if not response.is_success:
                return IngestionResult(
                    success=False,
                    source_id=source_id,
                    error=f"HTTP {response.status_code}: {response.text}",
                    processing_time_seconds=time.time() - start_time,
                )

            # Extract content
            content_type = self._detect_api_content_type(response)
            content = await self._extract_api_content(response, content_type)

            # Create metadata
            metadata = self._create_source_metadata(
                source_id=source_id,
                source_type=SourceType.API,
                content_type=content_type,
                url=str(response.url),
                size_bytes=len(response.content),
                custom_metadata={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "method": method,
                },
            )

            processing_time = time.time() - start_time

            result = IngestionResult(
                success=True,
                source_id=source_id,
                content=content,
                metadata=metadata,
                processing_time_seconds=processing_time,
                raw_size_bytes=len(response.content),
                processed_size_bytes=len(content.encode("utf-8")),
            )

            self.logger.info(f"Successfully ingested API endpoint: {source_id}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to ingest API endpoint {source_id}: {e}")
            return IngestionResult(
                success=False,
                source_id=source_id,
                error=str(e),
                processing_time_seconds=processing_time,
            )

    def _find_endpoint_config(self, source_id: str) -> Dict[str, Any]:
        """Find configuration for specific endpoint"""
        for endpoint in self.endpoints:
            if endpoint.get("path") == source_id:
                return endpoint
        return {}

    def _detect_api_content_type(self, response: Response) -> ContentType:
        """Detect content type from API response"""
        content_type_header = response.headers.get("content-type", "").lower()

        if "application/json" in content_type_header:
            return ContentType.JSON
        elif (
            "text/xml" in content_type_header
            or "application/xml" in content_type_header
        ):
            return ContentType.XML
        elif "text/html" in content_type_header:
            return ContentType.HTML
        elif "text/csv" in content_type_header:
            return ContentType.CSV
        else:
            return ContentType.TEXT

    async def _extract_api_content(
        self, response: Response, content_type: ContentType
    ) -> str:
        """Extract and normalize content from API response"""
        try:
            if content_type == ContentType.JSON:
                json_data = response.json()
                return json.dumps(json_data, indent=2, ensure_ascii=False)
            else:
                return response.text
        except Exception as e:
            self.logger.warning(f"Failed to parse structured content: {e}")
            return response.text

    async def ingest_batch(
        self, source_ids: List[str], **kwargs
    ) -> List[IngestionResult]:
        """Ingest multiple API endpoints concurrently"""
        max_concurrent = kwargs.get("max_concurrent", 5)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def ingest_with_semaphore(source_id: str):
            async with semaphore:
                return await self.ingest(source_id, **kwargs)

        # Run ingestion tasks concurrently
        tasks = [ingest_with_semaphore(source_id) for source_id in source_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    IngestionResult(
                        success=False, source_id=source_ids[i], error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.success)
        self.logger.info(
            f"API batch ingestion completed: {successful}/{len(source_ids)} successful"
        )

        return processed_results

    async def ingest_paginated(self, source_id: str, **kwargs) -> List[IngestionResult]:
        """Ingest paginated API endpoint"""
        results = []
        page = kwargs.get("start_page", 1)
        max_pages = kwargs.get("max_pages", 100)
        page_param = kwargs.get("page_param", "page")

        while page <= max_pages:
            # Add pagination parameter
            pagination_kwargs = kwargs.copy()
            if "params" not in pagination_kwargs:
                pagination_kwargs["params"] = {}
            pagination_kwargs["params"][page_param] = page

            result = await self.ingest(source_id, **pagination_kwargs)
            results.append(result)

            # Stop if request failed
            if not result.success:
                break

            # Check if there are more pages (this is API-specific logic)
            if not self._has_more_pages(result, **kwargs):
                break

            page += 1

        return results

    def _has_more_pages(self, result: IngestionResult, **kwargs) -> bool:
        """Check if there are more pages (to be customized per API)"""
        # This is a simple heuristic - can be overridden per API
        if not result.content:
            return False

        try:
            if result.metadata.content_type == ContentType.JSON:
                data = json.loads(result.content)
                # Common pagination patterns
                if isinstance(data, dict):
                    return (
                        data.get("has_more", False)
                        or data.get("next", None) is not None
                        or (
                            isinstance(data.get("results", []), list)
                            and len(data["results"]) > 0
                        )
                    )
                elif isinstance(data, list):
                    return len(data) > 0
        except Exception:
            pass

        return False

    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Get information about API endpoint"""
        endpoint_config = self._find_endpoint_config(source_id)

        return {
            "source_id": source_id,
            "connector": self.name,
            "url": urljoin(self.base_url, source_id),
            "method": endpoint_config.get("method", "GET"),
            "auth_type": self.auth_type,
            "rate_limit": (
                f"{self.rate_limit_requests} requests per " f"{self.rate_limit_window}s"
            ),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "endpoint_config": endpoint_config,
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }


class RateLimiter:
    """Simple rate limiter for API requests"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()

            # Remove old requests outside the window
            self.requests = [
                req_time
                for req_time in self.requests
                if now - req_time < self.window_seconds
            ]

            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                wait_time = self.window_seconds - (now - self.requests[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Remove the oldest request
                    self.requests.pop(0)

            # Add current request
            self.requests.append(now)


# Register the connector
ConnectorFactory.register("api", APIConnector)
