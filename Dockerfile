# Multi-stage Dockerfile for RL Portfolio Optimization
# Build context: project root (docker build -f infrastructure/Dockerfile .)
# Supports both CPU and GPU training

ARG CUDA_VERSION=11.8.0
ARG PYTHON_VERSION=3.10

# Base stage with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 as base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    git \
    wget \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY code/requirements.txt ./requirements.txt
COPY code/requirements-prod.txt ./requirements-prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code (preserves import structure: code/ at /app/code/)
COPY code/ ./code/

# Create necessary runtime directories
RUN mkdir -p data models results/figures results/logs results/reports

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — run from /app so `code.production.api` resolves correctly
CMD ["python", "-m", "uvicorn", "code.production.api:app", "--host", "0.0.0.0", "--port", "8000"]
