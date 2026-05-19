from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SAM3DUnavailable(RuntimeError):
    """Raised when SAM3D cannot be loaded or run."""


@dataclass(slots=True)
class ReconstructionResult:
    glb_path: str | None
    ply_path: str | None
    debug_mask_path: str | None
    raw_output_metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SAM3DReconstructor:
    """SAM3D Objects adapter with lazy imports.

    Expects a SAM3D repository/config/checkpoints to be supplied at runtime.
    """

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        compile_model: bool = False,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.config_path = Path(config_path).resolve() if config_path else None
        self.compile_model = compile_model
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        if self.repo_dir is None:
            raise SAM3DUnavailable("SAM3D repo_dir was not provided.")
        if self.config_path is None:
            raise SAM3DUnavailable("SAM3D config_path was not provided.")
        if not self.repo_dir.exists():
            raise SAM3DUnavailable(f"SAM3D repo_dir not found: {self.repo_dir}")
        if not self.config_path.exists():
            raise SAM3DUnavailable(f"SAM3D config not found: {self.config_path}")

        inference_path = self.repo_dir / "notebook" / "inference.py"
        if not inference_path.exists():
            raise SAM3DUnavailable(f"SAM3D inference.py not found: {inference_path}")

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            spec = importlib.util.spec_from_file_location("graspsense_sam3d_inference", inference_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create import spec for {inference_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            inference_cls = getattr(module, "Inference")
            self._model = inference_cls(str(self.config_path), compile=self.compile_model)
        except Exception as exc:
            raise SAM3DUnavailable(
                f"SAM3D dependencies or checkpoints are unavailable: {exc}"
            ) from exc

    def reconstruct(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        output_dir: str | Path,
        seed: int = 42,
    ) -> ReconstructionResult:
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_dir = Path(output_dir)
        if not image_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"Image not found: {image_path}")
        if not mask_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"Mask not found: {mask_path}")

        try:
            self.load_model()
        except Exception as exc:
            return ReconstructionResult(None, None, None, success=False, error=str(exc))

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            return ReconstructionResult(None, None, None, success=False, error=f"OpenCV/numpy unavailable: {exc}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ReconstructionResult(None, None, None, success=False, error=f"Unable to read image: {image_path}")
        if mask is None:
            return ReconstructionResult(None, None, None, success=False, error=f"Unable to read mask: {mask_path}")

        mask = (mask > 127).astype(np.uint8) * 255
        if int(np.count_nonzero(mask)) == 0:
            return ReconstructionResult(None, None, None, success=False, error="Mask is empty after binarization")

        output_dir.mkdir(parents=True, exist_ok=True)
        debug_mask_path = output_dir / "debug_mask.png"
        cv2.imwrite(str(debug_mask_path), mask)

        try:
            model = self._model
            if model is None:
                raise SAM3DUnavailable("SAM3D model was not loaded")
            output = model(image, mask, seed=seed)

            saved: dict[str, str | None] = {"glb_path": None, "ply_path": None}
            if isinstance(output, dict):
                glb_obj = output.get("glb")
                if glb_obj is not None:
                    glb_path = output_dir / "object.glb"
                    if hasattr(glb_obj, "export"):
                        glb_obj.export(str(glb_path))
                        saved["glb_path"] = str(glb_path)
                    elif isinstance(glb_obj, (bytes, bytearray)):
                        glb_path.write_bytes(glb_obj)
                        saved["glb_path"] = str(glb_path)

                gs_obj = output.get("gs") or _first_or_none(output.get("gaussian"))
                if gs_obj is not None and hasattr(gs_obj, "save_ply"):
                    ply_path = output_dir / "object_gs.ply"
                    gs_obj.save_ply(str(ply_path))
                    saved["ply_path"] = str(ply_path)

            metadata = _summarize_raw_output(output)
            return ReconstructionResult(
                glb_path=saved["glb_path"],
                ply_path=saved["ply_path"],
                debug_mask_path=str(debug_mask_path),
                raw_output_metadata=metadata,
                success=bool(saved["glb_path"] or saved["ply_path"]),
                error=None if (saved["glb_path"] or saved["ply_path"]) else "SAM3D ran but no GLB/PLY output was saved",
            )
        except Exception as exc:
            return ReconstructionResult(
                glb_path=None,
                ply_path=None,
                debug_mask_path=str(debug_mask_path),
                raw_output_metadata={},
                success=False,
                error=f"SAM3D reconstruction failed: {exc}",
            )


def _first_or_none(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _summarize_raw_output(output) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"type": type(output).__name__}
    summary: dict[str, Any] = {"keys": sorted(output.keys())}
    for key in ("rotation", "translation", "scale", "iou", "iou_before_optim"):
        value = output.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        summary[key] = value
    return summary
