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

Use this if you cloned GraspSense on a new machine and do not have access to the internal ISR Lab workspace.

GraspSense provides the pipeline code and wrappers for external tools, but it does **not** store large model weights, third-party checkpoints, Isaac Sim, IsaacLab, or the full SAM3D installation inside this repository.

External users are expected to install and configure the following components separately:

- **Isaac Sim**  
  Required for opening and visually inspecting the final USD scene.

- **IsaacLab**  
  Required for the headless backend steps used by GraspSense:
  - GLB / mesh to USD conversion
  - physics material assignment
  - collision setup
  - creation of the final IsaacLab scene file

- **SAM3D repository and checkpoints**  
  Required for 3D reconstruction from the RGB image and SAM mask.  
  GraspSense does not vendor SAM3D checkpoints. You need to install SAM3D separately and provide paths to:
  - the SAM3D repository
  - the SAM3D pipeline config
  - the SAM3D checkpoints

- **SAM checkpoint**  
  Required for 2D object segmentation, for example `sam_vit_h_4b8939.pth`.

- **YOLO-World weights**  
  Required for open-vocabulary object detection, for example `yolov8s-worldv2.pt`.

- **Qwen model, optional for now**  
  Required only if you want real VLM-based task understanding.  
  If Qwen is not available, GraspSense currently falls back to a simple rule-based parser.

Recommended external setup flow:

1. Clone GraspSense:

        git clone https://github.com/eliizavetas/GraspSense.git
        cd GraspSense

2. Create the main GraspSense environment:

        conda env create -f envs/environment.main.yml
        conda activate graspsense-main

3. Create the SAM3D environment from the provided recipe:

        conda env create -f envs/environment.sam3d.yml

   or use:

        bash scripts/setup_sam3d_env.sh

   This creates the Python environment needed to run the SAM3D wrapper, but it does **not** download the SAM3D repository or checkpoints automatically.

4. Install SAM3D separately according to the SAM3D installation instructions.

   After installation, you should know the paths to:

        /path/to/sam-3d-objects
        /path/to/sam-3d-objects/checkpoints/pipeline.yaml

5. Install Isaac Sim and IsaacLab separately.

   Before running GraspSense, verify that IsaacLab works in headless mode:

        cd /path/to/IsaacLab
        conda activate <your_isaaclab_env>
        ./isaaclab.sh -p -c "print('IsaacLab works')" --headless

6. Copy the example local config:

        cp configs/local.example.yaml configs/local.yaml

   Then edit `configs/local.yaml` and set paths for your machine:

        paths:
          isaaclab_root: /path/to/IsaacLab
          sam3d_repo: /path/to/sam-3d-objects
          sam3d_config: /path/to/sam-3d-objects/checkpoints/pipeline.yaml
          sam_checkpoint: /path/to/sam_vit_h_4b8939.pth
          yolo_model: /path/to/yolov8s-worldv2.pt

        envs:
          main: graspsense-main
          sam3d: graspsense-sam3d

`configs/local.yaml` is machine-specific and should not be committed.

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

- Qwen task understanding is executed through an isolated subprocess. This prevents the Qwen model from keeping GPU memory occupied before the SAM3D reconstruction stage.
- SAM3D is executed in a dedicated conda environment to avoid dependency conflicts.
- IsaacLab operations are executed through `isaaclab.sh` subprocess wrappers.
- `configs/local.yaml` is intentionally ignored by Git because it contains machine-specific paths.
- The current force map is a lightweight placeholder and should not yet be treated as the final physical force model.
