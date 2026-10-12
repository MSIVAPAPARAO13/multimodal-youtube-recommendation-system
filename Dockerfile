# Dockerfile for Multimodal Video Recommender
#
# Multi-stage build:
# 1. Base: Common dependencies
# 2. Dev: Development tools
# 3. Prod: Optimized for production

# ============ Base Stage ============
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    # For audio processing
    libsndfile1 \
    ffmpeg \
    # For PostgreSQL
    libpq-dev \
    # Utilities & HTTP
    curl \
    # Build tools
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ============ Development Stage ============
FROM base as dev

# Install dev dependencies
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    ipython

# Copy source code
COPY . .

# Default command for development
CMD ["uvicorn", "src.api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


# ============ Production Stage ============
FROM base as prod

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command (more workers, no reload)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
