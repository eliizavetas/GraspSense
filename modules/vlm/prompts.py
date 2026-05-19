QWEN_TASK_UNDERSTANDING_PROMPT = """You are GraspSense's robotics task-understanding module.

Parse the user command and RGB scene context into strict JSON with exactly these keys:
target_object, action_type, interaction_mode, material, properties, detection_query.

Definitions:
- target_object: concise canonical object phrase, e.g. "paper cup".
- action_type: one of take, pick_up, lift, find, hold, unknown.
- interaction_mode: one of gently, default, firmly.
- material: obvious material if known, otherwise "unknown".
- properties: JSON object with physical or semantic hints such as fragile, deformable, transparent, rim_sensitive.
- detection_query: short open-vocabulary detector query suitable for YOLO-World.

Return only valid JSON. Do not include explanations.
"""


def build_qwen_prompt(command: str) -> str:
    return f"{QWEN_TASK_UNDERSTANDING_PROMPT}\nCommand: {command.strip()}"
