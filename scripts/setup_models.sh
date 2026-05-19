#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p models/yolo
mkdir -p models/sam
mkdir -p models/qwen
mkdir -p models/sam3d

echo "[GraspSense] Model folders created:"
echo "  models/yolo"
echo "  models/sam"
echo "  models/qwen"
echo "  models/sam3d"

echo "[GraspSense] TODO:"
echo "  - download YOLO-World weights"
echo "  - download SAM checkpoint"
echo "  - download or configure Qwen model"
echo "  - configure SAM3D checkpoints"
