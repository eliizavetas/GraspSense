#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[GraspSense] Running pipeline..."

conda run -n graspsense-main python scripts/run_pipeline.py "$@"
