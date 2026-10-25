# Production Multi-Stage Dockerfile for Robotics Object Detection & Distance Estimation
# Optimized for Edge AMRs, Low Latency, and Industrial Security Standards

FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for industrial security hardening (least privilege)
RUN useradd -m -u 1000 robotics && \
    mkdir -p /tmp/matplotlib /app/output /app/data /app/weights && \
    chown -R robotics:robotics /app /tmp/matplotlib

# Copy requirements and install dependencies
COPY --chown=robotics:robotics requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, weights, and scripts
COPY --chown=robotics:robotics configs/ /app/configs/
COPY --chown=robotics:robotics src/ /app/src/
COPY --chown=robotics:robotics scripts/ /app/scripts/
COPY --chown=robotics:robotics weights/ /app/weights/
COPY --chown=robotics:robotics deploy_node.py /app/deploy_node.py
COPY --chown=robotics:robotics yolov8n.pt /app/yolov8n.pt
COPY --chown=robotics:robotics yolo11n.pt /app/yolo11n.pt

# Switch to non-root user
USER robotics

# Default command: run inference pipeline
CMD ["python", "deploy_node.py", "--source", "data/bdd100k_sample/images", "--output", "output/docker_run"]
