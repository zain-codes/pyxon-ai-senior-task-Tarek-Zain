# ==============================================================================
# Pyxon AI — Multi-Stage Docker Build
# ==============================================================================
# Stage 1: Install dependencies into a virtual environment.
# Stage 2: Copy only the venv + source code (smaller final image).
# ==============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building native wheels (faiss-cpu, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir \
        torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.lock

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Create non-root user with a writable home dir for HuggingFace model cache
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Copy virtual environment from builder stage
COPY --from=builder /app/venv /app/venv

# Copy application source code
COPY src/ ./src/
COPY main.py .

# Switch to non-root user
USER appuser

# Default command — override in docker-compose or k8s as needed
CMD ["python", "main.py"]
