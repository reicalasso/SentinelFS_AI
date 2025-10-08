# Dockerfile for SentinelZer0
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.10-slim

# Set metadata
LABEL maintainer="SentinelZer0 Team"
LABEL version="3.8.0"
LABEL description="Enterprise AI Threat Detection System"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN useradd -m -u 1000 sentinelzer0 && \
    mkdir -p /app /data /logs /models && \
    chown -R sentinelzer0:sentinelzer0 /app /data /logs /models

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=sentinelzer0:sentinelzer0 . /app/

# Switch to app user
USER sentinelzer0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command (can be overridden)
CMD ["uvicorn", "sentinelzer0.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
