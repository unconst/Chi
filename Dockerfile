# Lemma Subnet Validator Docker Image
# Build: docker build -t lemma-validator .
# Run: docker-compose up -d

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml ./

# Install uv for faster dependency resolution
RUN pip install --no-cache-dir --upgrade pip uv

# Copy application code
COPY . .

# Install dependencies
RUN uv pip install --system --no-cache -e .

# Create non-root user for security
RUN useradd -m -u 1000 validator && \
    chown -R validator:validator /app

# Environment configuration
ENV NETWORK=finney
ENV NETUID=1
ENV WALLET_NAME=validator
ENV HOTKEY_NAME=default
ENV LOG_LEVEL=INFO
ENV EMA_ALPHA=0.35
ENV MINER_REQUEST_TIMEOUT=45
ENV LEAN_TIMEOUT_SECONDS=30
ENV MAX_RESPONSE_CHARS=16000
ENV MAX_PROOF_CHARS_FOR_SCORING=12000

# Run the validator
CMD ["python", "validator.py"]
