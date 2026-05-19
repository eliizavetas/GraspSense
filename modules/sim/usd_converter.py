from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class USDConversionResult:
    input_mesh_path: str
    output_usd_path: Optional[str]
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class USDConverter:
    """
    Mesh/GLB to USD conversion wrapper for GraspSense.

    The actual IsaacLab conversion is executed via:
      <isaaclab_root>/isaaclab.sh -p GraspSense/scripts/convert_mesh_to_usd_isaaclab.py ...

    This keeps the main GraspSense pipeline safe to import outside Isaac Sim,
    while still using the real IsaacLab MeshConverter backend.
    """

    def __init__(
        self,
        output_dir: str = "data/output/usd",
        isaaclab_root: str | Path | None = None,
        collision_approximation: str = "sdf",
        mass: float = 0.5,
        static_friction: float = 0.6,
        dynamic_friction: float = 0.4,
        restitution: float = 0.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).resolve().parents[2]
        self.isaaclab_root = Path(isaaclab_root).resolve() if isaaclab_root else self.project_root.parent
        self.collision_approximation = collision_approximation
        self.mass = mass
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.restitution = restitution

    def convert(
        self,
        mesh_path: Optional[str],
        output_name: Optional[str] = None,
        material: str = "unknown",
    ) -> USDConversionResult:
        if not mesh_path:
            return USDConversionResult(
                input_mesh_path="",
                output_usd_path=None,
                status="skipped",
                error="No mesh_path was provided for USD conversion.",
                metadata={"material": material},
            )

        mesh_path_obj = Path(mesh_path).resolve()

        if not mesh_path_obj.exists():
            return USDConversionResult(
                input_mesh_path=str(mesh_path_obj),
                output_usd_path=None,
                status="error",
                error=f"Input mesh does not exist: {mesh_path_obj}",
                metadata={"material": material},
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            output_name = mesh_path_obj.stem + ".usd"

        output_usd_path = (self.output_dir / output_name).resolve()

        return self._convert_with_isaaclab_subprocess(
            mesh_path=mesh_path_obj,
            output_usd_path=output_usd_path,
            material=material,
        )

    def _convert_with_isaaclab_subprocess(
        self,
        mesh_path: Path,
        output_usd_path: Path,
        material: str,
    ) -> USDConversionResult:
        isaaclab_sh = self.isaaclab_root / "isaaclab.sh"
        converter_script = self.project_root / "scripts" / "convert_mesh_to_usd_isaaclab.py"

        if not isaaclab_sh.exists():
            return USDConversionResult(
                input_mesh_path=str(mesh_path),
                output_usd_path=None,
                status="error",
                error=f"isaaclab.sh not found: {isaaclab_sh}",
                metadata={"material": material, "isaaclab_root": str(self.isaaclab_root)},
            )

        if not converter_script.exists():
            return USDConversionResult(
                input_mesh_path=str(mesh_path),
                output_usd_path=None,
                status="error",
                error=f"IsaacLab converter script not found: {converter_script}",
                metadata={"material": material},
            )

        output_usd_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(isaaclab_sh),
            "-p",
            str(converter_script),
            "--input",
            str(mesh_path),
            "--output",
            str(output_usd_path),
            "--collision-approximation",
            self.collision_approximation,
            "--mass",
            str(self.mass),
            "--static-friction",
            str(self.static_friction),
            "--dynamic-friction",
            str(self.dynamic_friction),
            "--restitution",
            str(self.restitution),
            "--headless",
        ]

        try:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)

            completed = subprocess.run(
                cmd,
                cwd=str(self.isaaclab_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=env,
            )
        except Exception as exc:
            return USDConversionResult(
                input_mesh_path=str(mesh_path),
                output_usd_path=None,
                status="error",
                error=f"Failed to launch IsaacLab converter: {exc}",
                metadata={"material": material},
            )

        parsed = _extract_json_from_stdout(completed.stdout)

        if completed.returncode != 0:
            error = None
            if parsed:
                error = parsed.get("error")
            if not error:
                error = completed.stderr.strip() or completed.stdout.strip() or "IsaacLab conversion failed."

            return USDConversionResult(
                input_mesh_path=str(mesh_path),
                output_usd_path=parsed.get("output_usd_path") if parsed else None,
                status="error",
                error=error,
                metadata={
                    "material": material,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                    "isaaclab_root": str(self.isaaclab_root),
                    "converter": "isaaclab.sh + convert_mesh_to_usd_isaaclab.py",
                },
            )

        if not parsed:
            # Isaac/Kit can sometimes print noisy logs in a way that makes JSON parsing fail.
            # If the subprocess exited successfully and the requested USD exists, treat it as success.
            if completed.returncode == 0 and output_usd_path.exists():
                return USDConversionResult(
                    input_mesh_path=str(mesh_path),
                    output_usd_path=str(output_usd_path),
                    status="success",
                    error=None,
                    metadata={
                        "material": material,
                        "asset_root_path": None,
                        "collision_approximation": self.collision_approximation,
                        "mass": self.mass,
                        "returncode": completed.returncode,
                        "stdout_tail": completed.stdout[-2000:],
                        "stderr_tail": completed.stderr[-2000:],
                        "isaaclab_root": str(self.isaaclab_root),
                        "converter": "isaaclab.sh + convert_mesh_to_usd_isaaclab.py",
                        "note": "JSON parse failed, but USD file exists after successful subprocess run.",
                    },
                )

            return USDConversionResult(
                input_mesh_path=str(mesh_path),
                output_usd_path=None,
                status="error",
                error="IsaacLab converter finished, but no JSON result was parsed.",
                metadata={
                    "material": material,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                },
            )

        success = bool(parsed.get("success"))
        return USDConversionResult(
            input_mesh_path=str(mesh_path),
            output_usd_path=parsed.get("output_usd_path"),
            status="success" if success else "error",
            error=parsed.get("error"),
            metadata={
                "material": material,
                "asset_root_path": parsed.get("asset_root_path"),
                "collision_approximation": parsed.get("collision_approximation"),
                "mass": parsed.get("mass"),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
                "isaaclab_root": str(self.isaaclab_root),
                "converter": "isaaclab.sh + convert_mesh_to_usd_isaaclab.py",
            },
        )


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Extract a JSON object from noisy Isaac Sim stdout.

    Isaac/Kit can print logs before and after our JSON result. Therefore
    json.loads(stdout) is too fragile. We scan every "{" position and use
    JSONDecoder.raw_decode(), which accepts a valid JSON object even if
    non-JSON logs follow it.
    """
    if not stdout:
        return None

    decoder = json.JSONDecoder()

    candidates: list[dict[str, Any]] = []
    for start, char in enumerate(stdout):
        if char != "{":
            continue

        candidate = stdout[start:].lstrip()
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict):
            candidates.append(parsed)

    # Prefer the result object produced by convert_mesh_to_usd_isaaclab.py.
    for parsed in reversed(candidates):
        if "success" in parsed and "output_usd_path" in parsed:
            return parsed

    return candidates[-1] if candidates else None
