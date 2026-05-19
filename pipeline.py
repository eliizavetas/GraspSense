from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


class GraspSensePipeline:
    """
    Main GraspSense pipeline.

    Current scope:
    1. Language command + RGB input
    2. Qwen-based or fallback task understanding
    3. YOLO-World detection
    4. SAM segmentation
    5. SAM3D reconstruction
    6. GLB/mesh to USD conversion
    7. Force map construction

    Future extensions:
    - grasp generation
    - force-map-aware grasp ranking
    - motion planning
    - impedance grip execution
    """

    def run(
        self,
        command: str,
        image_path: str,
        output_dir: str = "data/output",
        yolo_model_path: str | None = None,
        sam_checkpoint_path: str | None = None,
        sam3d_repo_dir: str | None = None,
        sam3d_config_path: str | None = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "started",
            "errors": [],
            "input": {
                "command": command,
                "image_path": image_path,
                "output_dir": output_dir,
                "yolo_model_path": yolo_model_path,
                "sam_checkpoint_path": sam_checkpoint_path,
                "sam3d_repo_dir": sam3d_repo_dir,
                "sam3d_config_path": sam3d_config_path,
            },
            "task_understanding": None,
            "detection": None,
            "segmentation": None,
            "reconstruction": None,
            "simulation": None,
            "force_map": None,
        }

        image = Path(image_path)
        if not image.exists():
            result["status"] = "error"
            result["errors"].append(f"Input image does not exist: {image_path}")
            return result

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Stage 1: Qwen / fallback task understanding
        try:
            from modules.vlm import VLMModule

            vlm = VLMModule()
            task_output = vlm.parse(command=command, image_path=image_path)
            result["task_understanding"] = self._to_dict(task_output)
        except Exception as exc:
            result["errors"].append(f"Task understanding failed: {exc}")

        task = result.get("task_understanding") or {}
        detection_query = (
            task.get("detection_query")
            or task.get("target_object")
            or command
        )
        material = task.get("material", "unknown")

        # Stage 2: YOLO-World detection
        try:
            from modules.detection import YOLOWorldDetector

            detector = YOLOWorldDetector(model_path=yolo_model_path)
            detection_output = detector.detect(
                image_path=image_path,
                query=detection_query,
            )
            result["detection"] = self._to_dict(detection_output)
        except Exception as exc:
            result["errors"].append(f"Detection failed: {exc}")

        detection = result.get("detection") or {}
        bbox = detection.get("bbox")
        point_xy = detection.get("center") or detection.get("point_xy")

        # Stage 3: SAM segmentation
        try:
            from modules.segmentation import SAMSegmentor

            segmentor = SAMSegmentor(
                checkpoint_path=sam_checkpoint_path,
                output_dir=f"{output_dir}/masks",
            )
            segmentation_output = segmentor.segment(
                image_path=image_path,
                point_xy=point_xy,
                bbox=bbox,
            )
            result["segmentation"] = self._to_dict(segmentation_output)
        except Exception as exc:
            result["errors"].append(f"Segmentation failed: {exc}")

        segmentation = result.get("segmentation") or {}
        mask_path = segmentation.get("mask_path")

        # Stage 4: SAM3D reconstruction
        if not mask_path:
            result["reconstruction"] = {
                "success": False,
                "status": "skipped",
                "error": "SAM3D reconstruction skipped because no SAM mask_path is available.",
                "glb_path": None,
                "ply_path": None,
                "debug_mask_path": None,
            }
        else:
            try:
                from modules.segmentation import SAM3DReconstructor

                reconstructor = SAM3DReconstructor(
                    repo_dir=sam3d_repo_dir,
                    config_path=sam3d_config_path,
                )
                reconstruction_output = reconstructor.reconstruct(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_dir=f"{output_dir}/reconstruction",
                )
                result["reconstruction"] = self._to_dict(reconstruction_output)
            except Exception as exc:
                result["errors"].append(f"Reconstruction failed: {exc}")

        reconstruction = result.get("reconstruction") or {}
        mesh_path = (
            reconstruction.get("glb_path")
            or reconstruction.get("mesh_path")
            or reconstruction.get("ply_path")
        )

        # Stage 5: USD conversion and scene import
        try:
            from modules.sim import SceneManager, USDConverter

            converter = USDConverter(output_dir=f"{output_dir}/usd")
            conversion_output = converter.convert(
                mesh_path=mesh_path,
                material=material,
            )
            conversion_dict = self._to_dict(conversion_output)

            scene = SceneManager()
            scene_output = scene.import_usd(
                usd_path=conversion_dict.get("output_usd_path"),
            )

            result["simulation"] = {
                "usd_conversion": conversion_dict,
                "scene_import": self._to_dict(scene_output),
            }
        except Exception as exc:
            result["errors"].append(f"Simulation preparation failed: {exc}")

        # Stage 6: Force map construction
        try:
            from modules.force_map import ForceMapBuilder

            force_builder = ForceMapBuilder()
            force_output = force_builder.build(
                mesh_path=mesh_path,
                material=material,
            )
            result["force_map"] = self._to_dict(force_output)
        except Exception as exc:
            result["errors"].append(f"Force map construction failed: {exc}")

        result["status"] = "success" if not result["errors"] else "partial_success"
        return result

    @staticmethod
    def _to_dict(value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "to_dict"):
            return value.to_dict()

        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)

        if isinstance(value, dict):
            return value

        return value
