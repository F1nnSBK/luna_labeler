# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to a local folder
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final runner stage
FROM python:3.11-slim

# Install runtime dependencies for OpenCV/Rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user named "user" with UID 1000
RUN useradd -m -u 1000 user

# Configure HF / Numba writable cache dirs under /tmp
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR $HOME/app

# Copy installed site-packages from builder stage
COPY --from=builder --chown=user:user /root/.local /home/user/.local

# Copy the rest of the application files
COPY --chown=user:user . .

USER user

# Create directories for caching
RUN mkdir -p /tmp/hf_cache /tmp/numba_cache

EXPOSE 7860

# Run uvicorn server with production-grade worker count configuration
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]