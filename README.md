# GraspSense

GraspSense is a language-guided perception and force-map construction pipeline for dexterous robotic manipulation.

The current version focuses on the first part of the full GraspSense system:

1. Language command + RGB image input
2. Qwen-based task understanding
3. YOLO-World → SAM → SAM3D → Isaac Sim asset preparation
4. Force map construction

Grasp generation, force-map-aware grasp ranking, motion planning, and impedance-based grip execution are planned as future extensions.

## Current Pipeline

```text
Language command + RGB image
        ↓
Qwen / fallback task understanding
        ↓
YOLO-World object detection
        ↓
SAM segmentation
        ↓
SAM3D reconstruction
        ↓
GLB / mesh → USD conversion
        ↓
Force map construction

