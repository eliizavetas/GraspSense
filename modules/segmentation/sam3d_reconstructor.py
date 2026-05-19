from __future__ import annotations

import json
import os
import subprocess
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
    """SAM3D adapter.

    Instead of importing SAM3D inside the main GraspSense environment, this adapter
    calls scripts/run_sam3d_reconstruction.py through a dedicated conda environment.

    This keeps heavy SAM3D dependencies such as kaolin, pytorch3d and custom CUDA
    packages isolated from the main pipeline environment.
    """

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        compile_model: bool = False,
        conda_env: str = "sam3d-objects",
        project_root: str | Path | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.config_path = Path(config_path).resolve() if config_path else None
        self.compile_model = compile_model
        self.conda_env = conda_env
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]

    def reconstruct(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        output_dir: str | Path,
        seed: int = 42,
    ) -> ReconstructionResult:
        image_path = Path(image_path).resolve()
        mask_path = Path(mask_path).resolve()
        output_dir = Path(output_dir).resolve()

        if self.repo_dir is None:
            return ReconstructionResult(None, None, None, success=False, error="SAM3D repo_dir was not provided.")
        if self.config_path is None:
            return ReconstructionResult(None, None, None, success=False, error="SAM3D config_path was not provided.")
        if not self.repo_dir.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"SAM3D repo_dir not found: {self.repo_dir}")
        if not self.config_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"SAM3D config not found: {self.config_path}")
        if not image_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"Image not found: {image_path}")
        if not mask_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"Mask not found: {mask_path}")

        wrapper_path = self.project_root / "scripts" / "run_sam3d_reconstruction.py"
        if not wrapper_path.exists():
            return ReconstructionResult(None, None, None, success=False, error=f"SAM3D wrapper not found: {wrapper_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(wrapper_path),
            "--repo-dir",
            str(self.repo_dir),
            "--config",
            str(self.config_path),
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
        ]

        if self.compile_model:
            cmd.append("--compile")

        try:
            env = os.environ.copy()
            # Do not leak the main GraspSense PYTHONPATH into the SAM3D conda environment.
            # Otherwise local sandbox/.deps packages can shadow the packages installed in sam3d-objects.
            env.pop("PYTHONPATH", None)

            completed = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=env,
            )
        except Exception as exc:
            return ReconstructionResult(
                None,
                None,
                None,
                success=False,
                error=f"Failed to launch SAM3D subprocess: {exc}",
            )

        parsed = _extract_json_from_stdout(completed.stdout)

        if completed.returncode != 0:
            error = None
            if parsed:
                error = parsed.get("error")
            if not error:
                error = completed.stderr.strip() or completed.stdout.strip() or "SAM3D subprocess failed."
            return ReconstructionResult(
                glb_path=parsed.get("glb_path") if parsed else None,
                ply_path=parsed.get("ply_path") if parsed else None,
                debug_mask_path=parsed.get("debug_mask_path") if parsed else None,
                raw_output_metadata={
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                },
                success=False,
                error=error,
            )

        if not parsed:
            return ReconstructionResult(
                glb_path=None,
                ply_path=None,
                debug_mask_path=None,
                raw_output_metadata={
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                },
                success=False,
                error="SAM3D subprocess finished, but no JSON result was parsed.",
            )

        return ReconstructionResult(
            glb_path=parsed.get("glb_path"),
            ply_path=parsed.get("ply_path"),
            debug_mask_path=parsed.get("debug_mask_path"),
            raw_output_metadata={
                "saved_paths": parsed.get("saved_paths", []),
                "repo_dir": parsed.get("repo_dir"),
                "config_path": parsed.get("config_path"),
                "output_dir": parsed.get("output_dir"),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            },
            success=bool(parsed.get("success")),
            error=parsed.get("error"),
        )


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Extract the final JSON object from noisy SAM3D stdout.

    SAM3D prints logs before/after the JSON block, so json.loads(stdout) is too fragile.
    This function searches for the last plausible JSON object.
    """

    if not stdout:
        return None

    starts = [idx for idx, char in enumerate(stdout) if char == "{"]
    for start in reversed(starts):
        candidate = stdout[start:].strip()
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None
