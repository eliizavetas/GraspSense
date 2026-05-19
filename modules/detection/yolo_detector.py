from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DetectionUnavailable(RuntimeError):
    """Raised when YOLO-World cannot be loaded or run."""


@dataclass(slots=True)
class DetectionResult:
    bbox: list[int] | None
    point_xy: list[int] | None
    score: float
    label: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YOLOWorldDetector:
    """Small YOLO-World adapter.

    The detector intentionally does not ship or assume local model weights.
    Pass a checkpoint/model path at runtime when YOLO-World is available.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | int | None = None,
        image_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None:
            raise DetectionUnavailable(
                "YOLO-World model path was not provided. Pass a checkpoint/model path at runtime; weights are not stored in this repo."
            )
        if not self.model_path.exists():
            raise DetectionUnavailable(f"YOLO-World checkpoint not found: {self.model_path}")

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise DetectionUnavailable(
                "YOLO-World dependencies are unavailable. Install ultralytics/torch in the runtime environment."
            ) from exc

        self._model = YOLO(str(self.model_path))

    def detect(self, image_path: str | Path, query: str) -> DetectionResult:
        image_path = Path(image_path)
        if not image_path.exists():
            return DetectionResult(
                bbox=None,
                point_xy=None,
                score=0.0,
                label=query,
                success=False,
                error=f"Image not found: {image_path}",
            )

        try:
            self.load_model()
        except Exception as exc:
            return DetectionResult(
                bbox=None,
                point_xy=None,
                score=0.0,
                label=query,
                success=False,
                error=str(exc),
            )

        try:
            import cv2  # type: ignore
        except Exception as exc:
            return DetectionResult(
                bbox=None,
                point_xy=None,
                score=0.0,
                label=query,
                success=False,
                error=f"OpenCV is unavailable for image loading: {exc}",
            )

        image = cv2.imread(str(image_path))
        if image is None:
            return DetectionResult(
                bbox=None,
                point_xy=None,
                score=0.0,
                label=query,
                success=False,
                error=f"Unable to read image: {image_path}",
            )

        try:
            model = self._model
            if model is None:
                raise DetectionUnavailable("YOLO-World model was not loaded")

            class_names = [query]
            if hasattr(model, "set_classes"):
                model.set_classes(class_names)

            results = model.predict(
                image,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return DetectionResult(
                    bbox=None,
                    point_xy=None,
                    score=0.0,
                    label=query,
                    success=False,
                    error="No detection found",
                )

            conf = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            best_idx = int(conf.argmax())
            score = float(conf[best_idx])
            h, w = image.shape[:2]
            x1, y1, x2, y2 = _clamp_bbox(xyxy[best_idx], width=w, height=h)
            point_xy = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            return DetectionResult(
                bbox=[x1, y1, x2, y2],
                point_xy=point_xy,
                score=score,
                label=query,
                success=True,
            )
        except Exception as exc:
            return DetectionResult(
                bbox=None,
                point_xy=None,
                score=0.0,
                label=query,
                success=False,
                error=f"YOLO-World detection failed: {exc}",
            )


def _clamp_bbox(raw_box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in raw_box]
    xx1 = max(0, min(int(x1), width - 1))
    yy1 = max(0, min(int(y1), height - 1))
    xx2 = max(0, min(int(x2), width - 1))
    yy2 = max(0, min(int(y2), height - 1))
    if xx2 <= xx1 or yy2 <= yy1:
        raise ValueError(f"Invalid bbox after clamping: {(xx1, yy1, xx2, yy2)}")
    return xx1, yy1, xx2, yy2
