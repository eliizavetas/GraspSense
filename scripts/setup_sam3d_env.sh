#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

ENV_FILE="envs/environment.sam3d.yml"

if ! command -v conda &> /dev/null; then
    echo "[GraspSense] ERROR: conda was not found."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "[GraspSense] ERROR: $ENV_FILE not found."
    exit 1
fi

echo "[GraspSense] Creating SAM3D conda environment from $ENV_FILE"
conda env create -f "$ENV_FILE" || {
    echo "[GraspSense] Environment may already exist. Try:"
    echo "  conda env update -f $ENV_FILE --prune"
}

echo ""
echo "[GraspSense] SAM3D environment setup step finished."
echo ""
echo "Next steps:"
echo "1. Install/download the SAM3D repository separately."
echo "2. Download SAM3D checkpoints according to the SAM3D installation guide."
echo "3. Copy configs/local.example.yaml to configs/local.yaml and set:"
echo "   - paths.sam3d_repo"
echo "   - paths.sam3d_config"
echo "   - paths.sam_checkpoint"
