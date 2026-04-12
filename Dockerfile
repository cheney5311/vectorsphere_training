FROM python:3.10-slim

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Use a virtualenv inside image to isolate
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install requirements if present
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi || true
# Ensure prometheus client available for metrics (optional)
RUN pip install --no-cache-dir prometheus_client || true

EXPOSE 8001

# Default config: use environment variables to configure
ENV CHECKPOINT_QUEUE_DIR=/tmp/checkpoint_upload_queue
ENV CHECKPOINT_UPLOADER_POLL_INTERVAL=5
ENV CHECKPOINT_UPLOADER_METRICS_PORT=8001

CMD ["/usr/bin/env", "python", "-m", "backend.services.checkpoint_uploader_service"]
