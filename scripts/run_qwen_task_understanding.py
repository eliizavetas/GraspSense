#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen task understanding in an isolated process.")
    parser.add_argument("--command", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = {
        "success": False,
        "result": None,
        "error": None,
        "traceback": None,
    }

    try:
        # Make project imports work when called from subprocess.
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from modules.vlm import QwenTaskUnderstandingService

        service = QwenTaskUnderstandingService(
            model_id=args.model_id,
            device=args.device,
            backend="local",
        )

        parsed = service.understand(
            command=args.command,
            image_path=args.image,
        )

        result["success"] = parsed.backend == "qwen"
        result["result"] = parsed.to_dict()

        if parsed.backend != "qwen":
            result["error"] = parsed.raw_response or "Qwen returned fallback result."

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
