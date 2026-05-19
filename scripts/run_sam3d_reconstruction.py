#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM3D reconstruction from image and mask.")
    parser.add_argument("--repo-dir", required=True, help="Path to sam-3d-objects repository.")
    parser.add_argument("--config", required=True, help="Path to SAM3D pipeline config YAML.")
    parser.add_argument("--image", required=True, help="Path to RGB input image.")
    parser.add_argument("--mask", required=True, help="Path to binary/object mask image.")
    parser.add_argument("--output-dir", required=True, help="Directory where 3D outputs will be saved.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SAM3D inference.")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation if supported.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    config_path = Path(args.config).resolve()
    image_path = Path(args.image).resolve()
    mask_path = Path(args.mask).resolve()
    output_dir = Path(args.output_dir).resolve()

    result = {
        "success": False,
        "repo_dir": str(repo_dir),
        "config_path": str(config_path),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "output_dir": str(output_dir),
        "saved_paths": [],
        "glb_path": None,
        "ply_path": None,
        "obj_path": None,
        "debug_mask_path": None,
        "error": None,
    }

    try:
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        runner_path = repo_dir / "sam3d_runner.py"
        if not runner_path.exists():
            raise FileNotFoundError(f"sam3d_runner.py not found: {runner_path}")

        import importlib.util

        spec = importlib.util.spec_from_file_location("sam3d_runner_local", runner_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load runner from {runner_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Sam3DRunner"):
            raise AttributeError("Sam3DRunner class was not found in sam3d_runner.py")

        runner = module.Sam3DRunner(
            repo_dir=repo_dir,
            config_path=config_path,
            compile_model=args.compile,
        )

        saved_paths = runner.run(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            seed=args.seed,
        )

        saved_paths = [Path(p).resolve() for p in saved_paths]
        result["saved_paths"] = [str(p) for p in saved_paths]
        result["success"] = True

        for path in saved_paths:
            suffix = path.suffix.lower()
            name = path.name.lower()

            if suffix == ".glb" and result["glb_path"] is None:
                result["glb_path"] = str(path)
            elif suffix == ".ply" and result["ply_path"] is None:
                result["ply_path"] = str(path)
            elif suffix == ".obj" and result["obj_path"] is None:
                result["obj_path"] = str(path)
            elif "debug_mask" in name:
                result["debug_mask_path"] = str(path)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
