from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SceneImportResult:
    usd_path: Optional[str]
    prim_path: Optional[str]
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SceneManager:
    """
    IsaacLab scene manager wrapper.

    The real scene import is executed through IsaacLab/Isaac Sim via:
      <isaaclab_root>/isaaclab.sh -p GraspSense/scripts/import_usd_to_scene_isaaclab.py ...

    This keeps the main GraspSense pipeline safe to run outside Isaac Sim.
    """

    def __init__(
        self,
        default_prim_path: str = "/World/ReconstructedObject",
        stage_output: str = "data/output/stage/grasp_scene.usd",
        isaaclab_root: str | Path | None = None,
    ) -> None:
        self.default_prim_path = default_prim_path
        self.stage_output = Path(stage_output)
        self.project_root = Path(__file__).resolve().parents[2]
        self.isaaclab_root = Path(isaaclab_root).resolve() if isaaclab_root else self.project_root.parent

    def import_usd(
        self,
        usd_path: Optional[str],
        prim_path: Optional[str] = None,
    ) -> SceneImportResult:
        if prim_path is None:
            prim_path = self.default_prim_path

        if not usd_path:
            return SceneImportResult(
                usd_path=None,
                prim_path=prim_path,
                status="skipped",
                error="No USD path was provided for scene import.",
                metadata={},
            )

        usd_path_obj = Path(usd_path).resolve()
        if not usd_path_obj.exists():
            return SceneImportResult(
                usd_path=str(usd_path_obj),
                prim_path=prim_path,
                status="error",
                error=f"USD file does not exist: {usd_path_obj}",
                metadata={},
            )

        return self._import_with_isaaclab_subprocess(
            usd_path=usd_path_obj,
            prim_path=prim_path,
        )

    def _import_with_isaaclab_subprocess(self, usd_path: Path, prim_path: str) -> SceneImportResult:
        isaaclab_sh = self.isaaclab_root / "isaaclab.sh"
        import_script = self.project_root / "scripts" / "import_usd_to_scene_isaaclab.py"
        stage_output = (self.project_root / self.stage_output).resolve()

        if not isaaclab_sh.exists():
            return SceneImportResult(
                usd_path=str(usd_path),
                prim_path=prim_path,
                status="error",
                error=f"isaaclab.sh not found: {isaaclab_sh}",
                metadata={"isaaclab_root": str(self.isaaclab_root)},
            )

        if not import_script.exists():
            return SceneImportResult(
                usd_path=str(usd_path),
                prim_path=prim_path,
                status="error",
                error=f"IsaacLab scene import script not found: {import_script}",
                metadata={},
            )

        stage_output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(isaaclab_sh),
            "-p",
            str(import_script),
            "--usd",
            str(usd_path),
            "--prim-path",
            prim_path,
            "--stage-output",
            str(stage_output),
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
            return SceneImportResult(
                usd_path=str(usd_path),
                prim_path=prim_path,
                status="error",
                error=f"Failed to launch IsaacLab scene import: {exc}",
                metadata={},
            )

        parsed = _extract_json_from_stdout(completed.stdout)

        if completed.returncode != 0:
            error = parsed.get("error") if parsed else None
            if not error:
                error = completed.stderr.strip() or completed.stdout.strip() or "IsaacLab scene import failed."

            return SceneImportResult(
                usd_path=str(usd_path),
                prim_path=prim_path,
                status="error",
                error=error,
                metadata={
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                    "isaaclab_root": str(self.isaaclab_root),
                },
            )

        if not parsed:
            if completed.returncode == 0 and stage_output.exists():
                return SceneImportResult(
                    usd_path=str(usd_path),
                    prim_path=prim_path,
                    status="success",
                    error=None,
                    metadata={
                        "stage_output": str(stage_output),
                        "returncode": completed.returncode,
                        "stdout_tail": completed.stdout[-2000:],
                        "stderr_tail": completed.stderr[-2000:],
                        "isaaclab_root": str(self.isaaclab_root),
                        "note": "JSON parse failed, but stage file exists after successful subprocess run.",
                    },
                )

            return SceneImportResult(
                usd_path=str(usd_path),
                prim_path=prim_path,
                status="error",
                error="IsaacLab scene import finished, but no JSON result was parsed.",
                metadata={
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                },
            )

        success = bool(parsed.get("success"))
        return SceneImportResult(
            usd_path=parsed.get("usd_path", str(usd_path)),
            prim_path=parsed.get("prim_path", prim_path),
            status="success" if success else "error",
            error=parsed.get("error"),
            metadata={
                "stage_output": parsed.get("stage_output"),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
                "isaaclab_root": str(self.isaaclab_root),
                "importer": "isaaclab.sh + import_usd_to_scene_isaaclab.py",
            },
        )


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
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

    for parsed in reversed(candidates):
        if "success" in parsed and "stage_output" in parsed:
            return parsed

    return candidates[-1] if candidates else None
