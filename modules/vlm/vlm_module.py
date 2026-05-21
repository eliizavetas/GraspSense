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
        self.model_id = model_id or "Qwen/Qwen2.5-VL-3B-Instruct"
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
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception as exc:
            raise QwenBackendUnavailable(
                "Qwen dependencies are not available. Install transformers, accelerate, and qwen-vl-utils."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

    def _understand_with_qwen(self, command: str, image_path: str | None) -> TaskUnderstandingResult:
        self._load_qwen_if_needed()

        if self._model is None or self._processor is None:
            raise QwenBackendUnavailable("Qwen model or processor was not loaded.")

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as exc:
            raise QwenBackendUnavailable("qwen-vl-utils is required for Qwen-VL image processing.") from exc

        prompt = build_qwen_prompt(command)

        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to the first available model device.
        try:
            device = next(self._model.parameters()).device
            inputs = inputs.to(device)
        except Exception:
            pass

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        try:
            return self.result_from_qwen_text(output_text)
        except Exception as exc:
            raise QwenBackendUnavailable(
                f"Failed to parse Qwen JSON response: {exc}. Raw response: {output_text!r}"
            ) from exc

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
