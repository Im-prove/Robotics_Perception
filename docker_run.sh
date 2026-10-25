#!/usr/bin/env bash
# =====================================================================
#  Autonomous Robotics Perception Pipeline - Docker Build & Run Script
# =====================================================================
set -e

echo "[*] Checking Docker daemon status..."
if ! docker info > /dev/null 2>&1; then
    echo "[!] ERROR: Docker daemon is not running."
    echo "Please start the Docker service (e.g. 'sudo systemctl start docker' or launch Docker Desktop)."
    exit 1
fi

echo "[*] Building Docker image: eric-robotics/perception-pipeline:1.0.0..."
docker build -t eric-robotics/perception-pipeline:1.0.0 .

echo ""
echo "[*] Running Perception Pipeline Container..."
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/output:/app/output:rw" \
  -v "$(pwd)/weights:/app/weights:ro" \
  eric-robotics/perception-pipeline:1.0.0 \
  python deploy_node.py --source data/bdd100k_sample/images --output output/docker_run

echo ""
echo "[OK] Execution finished! Check results in: output/docker_run"
