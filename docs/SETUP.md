# GraspSense Setup

GraspSense supports two setup scenarios.

## Scenario A: Internal team setup

Use this if you are working on the shared ISR/IsaacLab machine where SAM3D, IsaacLab, YOLO weights and SAM checkpoints are already installed.

Run:

    bash scripts/run_local_demo.sh

This script uses local paths under `sandbox/legacy_sources`.

Expected outputs:

    data/output/masks/
    data/output/reconstruction/
    data/output/usd/
    data/output/stage/grasp_scene.usd

## Scenario B: External user setup

Use this if you cloned GraspSense on a new machine.

### Prerequisites

You must install separately:

1. Isaac Sim
2. IsaacLab
3. SAM3D repository and checkpoints
4. YOLO-World weights
5. SAM checkpoint
6. Optional: Qwen model for real VLM-based command understanding

GraspSense does not store large model weights, SAM3D checkpoints, or Isaac Sim / IsaacLab inside the repository.

### Create the main GraspSense environment

    conda env create -f envs/environment.main.yml
    conda activate graspsense-main

### Create the SAM3D environment

    bash scripts/setup_sam3d_env.sh

This creates a conda environment from:

    envs/environment.sam3d.yml

It does not download SAM3D checkpoints automatically.

### Configure local paths

Copy:

    cp configs/local.example.yaml configs/local.yaml

Then edit:

    configs/local.yaml

and set paths to:

    IsaacLab root
    SAM3D repo
    SAM3D config
    SAM checkpoint
    YOLO-World weights

### Run the pipeline

Example:

    python scripts/run_pipeline.py \
      --command "Take a cup carefully" \
      --image data/input/example_rgb.png \
      --yolo-model /path/to/yolov8s-worldv2.pt \
      --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
      --sam3d-repo /path/to/sam-3d-objects \
      --sam3d-config /path/to/sam-3d-objects/checkpoints/pipeline.yaml \
      --compact

## Output

The pipeline writes results to:

    data/output/masks/
    data/output/reconstruction/
    data/output/usd/
    data/output/stage/

The final IsaacLab scene is saved as:

    data/output/stage/grasp_scene.usd

Isaac Sim does not open automatically because the pipeline uses headless mode for USD conversion and scene generation.

To view the scene manually, open Isaac Sim and load:

    data/output/stage/grasp_scene.usd
