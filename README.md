# GraspSense

GraspSense is a language-guided perception, 3D reconstruction, and IsaacLab scene preparation pipeline for robotic manipulation.

The long-term goal is to convert a natural-language manipulation command and an RGB camera observation into a physically grounded object representation that can later be used for grasp planning, force-map-aware grasp ranking, and grip control.

## Current status

The current repository contains a working end-to-end skeleton for the perception and scene-preparation part of the system.

Currently working:

- command parsing with Qwen fallback logic
- YOLO-World object detection
- SAM object segmentation
- SAM3D reconstruction through a dedicated conda environment
- GLB / mesh to USD conversion through IsaacLab
- IsaacLab scene generation with the reconstructed object imported at `/World/ReconstructedObject`
- compact JSON pipeline output
- internal demo script for the shared lab machine

Still in progress:

- full Qwen VLM setup; fallback parsing is currently used when Qwen is unavailable
- automatic model/checkpoint download
- real physically grounded force-map construction
- USD primvar export for force-map values
- grasp generation, grasp ranking, motion planning, and impedance-based grip execution

## Current pipeline

    Language command + RGB image
            |
            v
    Qwen / fallback task understanding
            |
            v
    YOLO-World object detection
            |
            v
    SAM segmentation
            |
            v
    SAM3D reconstruction
            |
            v
    object.glb / object_gaussian.ply
            |
            v
    IsaacLab GLB to USD conversion
            |
            v
    IsaacLab scene generation
            |
            v
    data/output/stage/grasp_scene.usd
            |
            v
    placeholder force map summary

## Repository structure

    GraspSense/
    ├── modules/
    │   ├── vlm/              # command understanding; Qwen/fallback
    │   ├── detection/        # YOLO-World object detection
    │   ├── segmentation/     # SAM segmentation and SAM3D adapter
    │   ├── sim/              # IsaacLab USD conversion and scene import adapters
    │   └── force_map/        # placeholder force-map construction
    │
    ├── scripts/
    │   ├── run_pipeline.py
    │   ├── run_local_demo.sh
    │   ├── run_sam3d_reconstruction.py
    │   ├── convert_mesh_to_usd_isaaclab.py
    │   ├── import_usd_to_scene_isaaclab.py
    │   ├── setup_sam3d_env.sh
    │   └── setup_models.sh
    │
    ├── envs/
    │   ├── environment.main.yml
    │   └── environment.sam3d.yml
    │
    ├── configs/
    │   └── local.example.yaml
    │
    ├── docs/
    │   └── SETUP.md
    │
    ├── data/
    ├── pipeline.py
    ├── install.sh
    └── run.sh

## Setup scenarios

GraspSense supports two setup scenarios.

### Scenario A: internal team setup

Use this if you are working on the shared ISR/IsaacLab machine where IsaacLab, SAM3D, YOLO weights, and SAM checkpoints are already available.

Run:

    bash scripts/run_local_demo.sh

This uses local paths under `sandbox/legacy_sources`.

### Scenario B: external user setup

Use this if you cloned GraspSense on a new machine.

External users are expected to install separately:

- Isaac Sim
- IsaacLab
- SAM3D repository and checkpoints
- YOLO-World weights
- SAM checkpoint
- optionally Qwen for real VLM-based task understanding

The repository does not store large model weights, SAM3D checkpoints, Isaac Sim, or IsaacLab.

See the full setup guide:

    docs/SETUP.md

## Example command

    python scripts/run_pipeline.py \
      --command "Take a cup carefully" \
      --image data/input/example_rgb.png \
      --yolo-model /path/to/yolov8s-worldv2.pt \
      --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
      --sam3d-repo /path/to/sam-3d-objects \
      --sam3d-config /path/to/sam-3d-objects/checkpoints/pipeline.yaml \
      --compact

## Outputs

The pipeline writes runtime results to:

    data/output/masks/
    data/output/reconstruction/
    data/output/usd/
    data/output/stage/

The final generated IsaacLab stage is:

    data/output/stage/grasp_scene.usd

Isaac Sim does not open automatically. The pipeline runs IsaacLab in headless mode for USD conversion and scene generation. To inspect the result visually, open Isaac Sim manually and load `data/output/stage/grasp_scene.usd`.

## Important notes

- SAM3D is executed in a dedicated conda environment to avoid dependency conflicts.
- IsaacLab operations are executed through `isaaclab.sh` subprocess wrappers.
- `configs/local.yaml` is intentionally ignored by Git because it contains machine-specific paths.
- The current force map is a lightweight placeholder and should not yet be treated as the final physical force model.
