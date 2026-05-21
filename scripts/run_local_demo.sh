#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python scripts/run_pipeline.py \
  --command "Take a paper cup carefully" \
  --image sandbox/legacy_sources/RASA+YOLO/cups_scene.png \
  --yolo-model sandbox/legacy_sources/RASA+YOLO/vision/yolov8s-worldv2.pt \
  --sam-checkpoint sandbox/legacy_sources/masha_ws/sam-3d-objects/sam_vit_h_4b8939.pth \
  --sam3d-repo sandbox/legacy_sources/masha_ws/sam-3d-objects \
  --sam3d-config sandbox/legacy_sources/masha_ws/sam-3d-objects/checkpoints/pipeline.yaml \
  --compact
