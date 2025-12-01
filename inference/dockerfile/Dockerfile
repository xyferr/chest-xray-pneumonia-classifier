# =============================================================================
# Chest X-Ray Pneumonia Classifier - Docker Image
# =============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t chest-xray-classifier -f inference/dockerfile/Dockerfile .
# Run:   docker run -p 8000:8000 chest-xray-classifier
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and prepare environment
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY inference/api/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install PyTorch CPU version for smaller image (use CUDA version if GPU available)
# For CPU: torch torchvision --index-url https://download.pytorch.org/whl/cpu
# For CUDA: torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu


# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.10-slim as production

# Labels
LABEL maintainer="Chest X-Ray Classifier Team" \
      version="1.0.0" \
      description="Chest X-Ray Pneumonia Classification API"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Model configuration
    MODEL_PATH=/app/model/best_model.pt \
    MODEL_NAME=resnet50 \
    IMAGE_SIZE=224 \
    DEVICE=cpu \
    # Server configuration
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY src/ /app/src/
COPY inference/api/ /app/inference/api/

# Create model directory (model will be mounted or copied)
RUN mkdir -p /app/model && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "inference.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# -----------------------------------------------------------------------------
# Stage 3: Development - Includes additional tools
# -----------------------------------------------------------------------------
FROM production as development

# Switch to root for installing dev tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    isort \
    flake8

# Switch back to appuser
USER appuser

# Override CMD for development (with reload)
CMD ["python", "-m", "uvicorn", "inference.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
