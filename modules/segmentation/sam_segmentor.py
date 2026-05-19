from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SAMUnavailable(RuntimeError):
    """Raised when SAM cannot be loaded or run."""


@dataclass(slots=True)
class SegmentationResult:
    mask_path: str | None
    source_image_path: str
    point_xy: list[int] | None
    bbox: list[int] | None
    score: float | None
    mask_area: int | None
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SAMSegmentor:
    """Prompted SAM segmentor.

    Uses lazy imports and runtime checkpoint configuration. The repository does
    not vendor SAM or store SAM checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "vit_h",
        device: str | None = None,
        output_dir: str | Path = "outputs/segmentation",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.model_type = model_type
        self.device = device
        self.output_dir = Path(output_dir)
        self._predictor = None

    def load_model(self) -> None:
        if self._predictor is not None:
            return
        if self.checkpoint_path is None:
            raise SAMUnavailable(
                "SAM checkpoint path was not provided. Pass a checkpoint path at runtime; checkpoints are not stored in this repo."
            )
        if not self.checkpoint_path.exists():
            raise SAMUnavailable(f"SAM checkpoint not found: {self.checkpoint_path}")

        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except Exception as exc:
            raise SAMUnavailable(
                "SAM dependencies are unavailable. Install segment-anything in the runtime environment or expose it on PYTHONPATH."
            ) from exc

        if self.model_type not in sam_model_registry:
            raise SAMUnavailable(f"Unsupported SAM model type: {self.model_type}")

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        if self.device and hasattr(sam, "to"):
            sam.to(device=self.device)
        self._predictor = SamPredictor(sam)

    def segment(
        self,
        image_path: str | Path,
        point_xy: list[int] | tuple[int, int] | None,
        bbox: list[int] | tuple[int, int, int, int] | None = None,
        output_dir: str | Path | None = None,
    ) -> SegmentationResult:
        image_path = Path(image_path)
        if not image_path.exists():
            return self._failure(image_path, point_xy, bbox, f"Image not found: {image_path}")

        try:
            self.load_model()
        except Exception as exc:
            return self._failure(image_path, point_xy, bbox, str(exc))

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            return self._failure(image_path, point_xy, bbox, f"OpenCV/numpy unavailable: {exc}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return self._failure(image_path, point_xy, bbox, f"Unable to read image: {image_path}")

        try:
            predictor = self._predictor
            if predictor is None:
                raise SAMUnavailable("SAM predictor was not loaded")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            point_array = None
            label_array = None
            if point_xy is not None:
                px, py = _normalize_point(point_xy, image_bgr.shape)
                point_array = np.array([[px, py]])
                label_array = np.array([1])
            box_array = None
            normalized_bbox = None
            if bbox is not None:
                normalized_bbox = _normalize_bbox(bbox, image_bgr.shape)
                box_array = np.array(normalized_bbox)

            if point_array is None and box_array is None:
                return self._failure(image_path, point_xy, bbox, "SAM requires point_xy or bbox")

            masks, scores, _ = predictor.predict(
                point_coords=point_array,
                point_labels=label_array,
                box=box_array,
                multimask_output=True,
            )
            best_idx = int(scores.argmax())
            best_mask = masks[best_idx].astype(bool)
            mask_u8 = best_mask.astype("uint8") * 255
            mask_area = int(best_mask.sum())

            if mask_area == 0:
                return self._failure(image_path, point_xy, bbox, "SAM produced an empty mask")

            out_dir = Path(output_dir) if output_dir else self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            mask_path = out_dir / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask_u8)

            return SegmentationResult(
                mask_path=str(mask_path),
                source_image_path=str(image_path),
                point_xy=list(point_xy) if point_xy is not None else None,
                bbox=normalized_bbox,
                score=float(scores[best_idx]),
                mask_area=mask_area,
                success=True,
            )
        except Exception as exc:
            return self._failure(image_path, point_xy, bbox, f"SAM segmentation failed: {exc}")

    @staticmethod
    def _failure(
        image_path: Path,
        point_xy: list[int] | tuple[int, int] | None,
        bbox: list[int] | tuple[int, int, int, int] | None,
        error: str,
    ) -> SegmentationResult:
        return SegmentationResult(
            mask_path=None,
            source_image_path=str(image_path),
            point_xy=list(point_xy) if point_xy is not None else None,
            bbox=list(bbox) if bbox is not None else None,
            score=None,
            mask_area=None,
            success=False,
            error=error,
        )


def _normalize_point(point_xy, image_shape) -> tuple[int, int]:
    if len(point_xy) != 2:
        raise ValueError(f"point_xy must contain 2 values, got {point_xy!r}")
    x, y = int(point_xy[0]), int(point_xy[1])
    height, width = image_shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"point_xy {(x, y)} outside image bounds width={width}, height={height}")
    return x, y


def _normalize_bbox(bbox, image_shape) -> list[int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got {bbox!r}")
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clamping: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]
