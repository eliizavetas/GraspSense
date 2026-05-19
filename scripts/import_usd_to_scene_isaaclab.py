#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Import a USD asset into an IsaacLab USD stage.")
parser.add_argument("--usd", required=True, help="Path to prepared object USD.")
parser.add_argument("--prim-path", default="/World/ReconstructedObject", help="Prim path where the asset is referenced.")
parser.add_argument("--stage-output", default="data/output/stage/grasp_scene.usd", help="Output scene USD path.")
parser.add_argument("--set-default-prim", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, Sdf


def main() -> int:
    usd_path = Path(args_cli.usd).resolve()
    stage_output = Path(args_cli.stage_output).resolve()
    prim_path = args_cli.prim_path

    result = {
        "success": False,
        "usd_path": str(usd_path),
        "prim_path": prim_path,
        "stage_output": str(stage_output),
        "error": None,
    }

    try:
        if not usd_path.exists():
            raise FileNotFoundError(f"USD asset not found: {usd_path}")

        stage_output.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(stage_output))
        if stage is None:
            raise RuntimeError(f"Could not create stage: {stage_output}")

        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        asset_prim = stage.DefinePrim(prim_path, "Xform")
        asset_prim.GetReferences().AddReference(str(usd_path))

        # Store useful metadata on the imported prim.
        asset_prim.CreateAttribute("graspsense:sourceUsd", Sdf.ValueTypeNames.String).Set(str(usd_path))

        stage.GetRootLayer().Save()

        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2))
    simulation_app.close()
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
