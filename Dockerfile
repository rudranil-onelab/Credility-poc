# Credibility AI - Document Processing Platform
# Production-ready Dockerfile for EC2 deployment

FROM python:3.11-slim

# Metadata
LABEL maintainer="Credibility AI Team"
LABEL description="Credibility AI - Document Validation API Service"
LABEL version="2.0.0"

# Environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including OCR and PDF processing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories for outputs and reports
RUN mkdir -p outputs \
    Nodes/outputs \
    cross_validation/reports \
    result \
    logs

# Set proper permissions
RUN chmod -R 755 /app

# Expose port 8000 for FastAPI service
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default CMD - Run FastAPI server
CMD ["python", "-m", "uvicorn", "S3_Sqs.fe_push_simple_api:app", "--host", "0.0.0.0", "--port", "8000"]
