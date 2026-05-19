#!/usr/bin/env bash
set -e

echo "[GraspSense] Installing project environments..."

if ! command -v conda &> /dev/null; then
    echo "[GraspSense] ERROR: conda was not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "[GraspSense] Creating main environment..."
conda env create -f envs/environment.main.yml || echo "[GraspSense] Main env may already exist."

echo "[GraspSense] Creating SAM3D environment..."
conda env create -f envs/environment.sam3d.yml || echo "[GraspSense] SAM3D env may already exist."

echo "[GraspSense] Setup finished."
echo "Run:"
echo "  bash run.sh --command \"Take a cup carefully\" --image path/to/image.png"
