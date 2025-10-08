"""
FastAPI Middleware for Prometheus Metrics Collection

This middleware automatically collects HTTP request metrics for all endpoints.
"""

import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_CONNECTIONS,
    update_memory_usage,
    update_gpu_memory_usage
)

logger = logging.getLogger(__name__)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that collects Prometheus metrics for all HTTP requests.

    This middleware:
    - Tracks request count by method, endpoint, and status
    - Measures request latency
    - Updates system resource metrics
    - Logs performance information
    """

    def __init__(self, app: Callable, exclude_paths: list = None):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from metrics (e.g., ['/health', '/metrics'])
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/health', '/favicon.ico']
        self._last_memory_update = 0
        self._memory_update_interval = 30  # Update memory metrics every 30 seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request and collect metrics.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response: FastAPI response object
        """
        # Skip metrics for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        start_time = time.time()
        method = request.method
        path = request.url.path

        # Increment active connections
        ACTIVE_CONNECTIONS.inc()

        try:
            # Process the request
            response = await call_next(request)

            # Calculate latency
            latency = time.time() - start_time

            # Record metrics
            status_code = str(response.status_code)
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path
            ).observe(latency)

            # Log performance for slow requests
            if latency > 1.0:  # Log requests taking more than 1 second
                logger.warning(
                    f"Slow request: {method} {path} - {latency:.3f}s - Status: {status_code}"
                )
            elif latency > 0.1:  # Log requests taking more than 100ms
                logger.info(
                    f"Request: {method} {path} - {latency:.3f}s - Status: {status_code}"
                )

            return response

        except Exception as e:
            # Record error metrics
            latency = time.time() - start_time
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status="500"
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path
            ).observe(latency)

            logger.error(
                f"Request error: {method} {path} - {latency:.3f}s - Error: {str(e)}"
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )

        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

            # Update system metrics periodically
            current_time = time.time()
            if current_time - self._last_memory_update > self._memory_update_interval:
                try:
                    update_memory_usage()
                    update_gpu_memory_usage()
                    self._last_memory_update = current_time
                except Exception as e:
                    logger.warning(f"Failed to update system metrics: {e}")

class MetricsEndpoint:
    """
    Helper class to add Prometheus metrics endpoint to FastAPI app.
    """

    @staticmethod
    def add_to_app(app, endpoint: str = "/metrics"):
        """
        Add Prometheus metrics endpoint to FastAPI application.

        Args:
            app: FastAPI application instance
            endpoint: URL path for metrics endpoint
        """
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        @app.get(endpoint)
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )