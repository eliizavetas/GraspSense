from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .prompts import build_qwen_prompt


class QwenBackendUnavailable(RuntimeError):
    """Raised when Qwen runtime dependencies or model objects are unavailable."""


@dataclass(slots=True)
class TaskUnderstandingResult:
    target_object: str
    action_type: str
    interaction_mode: str
    material: str
    properties: dict[str, Any] = field(default_factory=dict)
    detection_query: str = ""
    backend: str = "fallback"
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MATERIAL_OBJECTS = {
    "paper cup": {
        "target_object": "paper cup",
        "material": "paper",
        "properties": {"fragile": True, "deformable": True, "rim_sensitive": True},
    },
    "plastic cup": {
        "target_object": "plastic cup",
        "material": "plastic",
        "properties": {"fragile": False, "deformable": True, "transparent": False},
    },
    "glass cup": {
        "target_object": "glass cup",
        "material": "glass",
        "properties": {"fragile": True, "breakable": True, "transparent": True},
    },
    "glass": {
        "target_object": "glass cup",
        "material": "glass",
        "properties": {"fragile": True, "breakable": True, "transparent": True},
    },
}


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    candidate = match.group(0) if match else text.strip()
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Qwen response JSON must be an object")
    return data


def parse_command_fallback(command: str) -> TaskUnderstandingResult:
    text = command.strip().lower()

    action_type = "unknown"
    if re.search(r"\bpick\s+up\b", text) or re.search(r"\bpick\b", text):
        action_type = "pick_up"
    elif re.search(r"\btake\b|\bgrab\b", text):
        action_type = "take"
    elif re.search(r"\blift\b|\braise\b", text):
        action_type = "lift"
    elif re.search(r"\bfind\b|\blocate\b|\bsearch\b", text):
        action_type = "find"
    elif re.search(r"\bhold\b", text):
        action_type = "hold"

    interaction_mode = "default"
    if re.search(r"\bgently\b|\bgentle\b|\bcarefully\b|\bcareful\b|\bdelicately\b|\bfragile\b", text):
        interaction_mode = "gently"
    elif re.search(r"\bfirmly\b|\bfirm\b|\btightly\b|\bsecurely\b", text):
        interaction_mode = "firmly"

    object_info = None
    for phrase, info in _MATERIAL_OBJECTS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            object_info = info
            break

    if object_info is None:
        object_info = {
            "target_object": "unknown",
            "material": "unknown",
            "properties": {},
        }

    target_object = str(object_info["target_object"])
    material = str(object_info["material"])
    properties = dict(object_info["properties"])
    if interaction_mode == "gently":
        properties.setdefault("requires_care", True)
    if interaction_mode == "firmly":
        properties.setdefault("requires_secure_contact", True)

    detection_query = target_object if target_object != "unknown" else _fallback_detection_query(text)

    return TaskUnderstandingResult(
        target_object=target_object,
        action_type=action_type,
        interaction_mode=interaction_mode,
        material=material,
        properties=properties,
        detection_query=detection_query,
        backend="fallback",
    )


def _fallback_detection_query(text: str) -> str:
    # Conservative fallback: remove common command words and keep a short detector phrase.
    cleaned = re.sub(
        r"\b(take|pick|up|lift|grab|hold|find|locate|search|for|the|a|an|gently|carefully|firmly|please)\b",
        " ",
        text,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?")
    return cleaned or "object"


class QwenTaskUnderstandingService:
    """Qwen-oriented task understanding with deterministic fallback.

    Real Qwen inference is intentionally isolated behind lazy imports so the
    repository remains importable before Qwen dependencies are installed.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        backend: str = "auto",
    ) -> None:
        self.model_id = model_id or "Qwen/Qwen2.5-VL"
        self.device = device or "auto"
        self.backend = backend
        self._model = None
        self._processor = None

    def parse(self, command: str, image_path: str | None = None) -> TaskUnderstandingResult:
        """Compatibility alias used by the main GraspSense pipeline."""
        return self.understand(command=command, image_path=image_path)

    def understand(self, command: str, image_path: str | None = None) -> TaskUnderstandingResult:
        if self.backend == "fallback":
            return parse_command_fallback(command)

        try:
            return self._understand_with_qwen(command=command, image_path=image_path)
        except Exception as exc:
            fallback = parse_command_fallback(command)
            fallback.raw_response = f"Qwen unavailable or failed; used fallback. Reason: {exc}"
            return fallback

    def _load_qwen_if_needed(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import AutoProcessor  # type: ignore
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception as exc:
            raise QwenBackendUnavailable(
                "Qwen dependencies are not available. Install the appropriate transformers/Qwen stack to enable real VLM inference."
            ) from exc

        # TODO: Verify the exact Qwen class/model id and image message formatting
        # for the target deployment environment before enabling production inference.
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.device,
        )

    def _understand_with_qwen(self, command: str, image_path: str | None) -> TaskUnderstandingResult:
        self._load_qwen_if_needed()
        prompt = build_qwen_prompt(command)

        # TODO: Implement real Qwen image+text inference once runtime dependencies,
        # model id, processor chat template, and image loading conventions are fixed.
        # The current integration pass deliberately avoids depending on Qwen being installed.
        raise QwenBackendUnavailable(
            f"Qwen inference is not wired yet for model_id={self.model_id!r}, image_path={image_path!r}."
        )

    @staticmethod
    def result_from_qwen_text(text: str) -> TaskUnderstandingResult:
        payload = _extract_json_object(text)
        return TaskUnderstandingResult(
            target_object=str(payload.get("target_object", "unknown")),
            action_type=str(payload.get("action_type", "unknown")),
            interaction_mode=str(payload.get("interaction_mode", "default")),
            material=str(payload.get("material", "unknown")),
            properties=dict(payload.get("properties") or {}),
            detection_query=str(payload.get("detection_query") or payload.get("target_object") or "object"),
            backend="qwen",
            raw_response=text,
        )
