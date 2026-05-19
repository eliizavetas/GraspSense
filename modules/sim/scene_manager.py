from dataclasses import dataclass, asdict
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
    Lightweight Isaac Sim scene manager placeholder.

    Safe to import outside Isaac Sim.

    TODO:
    - Create or connect to an Isaac Sim stage.
    - Add reconstructed object USD as a reference.
    - Assign transform and scale.
    - Validate collision and material properties.
    - Expose runtime hooks for force-map visualization.
    """

    def __init__(self, default_prim_path: str = "/World/ReconstructedObject") -> None:
        self.default_prim_path = default_prim_path

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

        try:
            return self._import_with_isaac(usd_path=usd_path, prim_path=prim_path)
        except ImportError as exc:
            return SceneImportResult(
                usd_path=usd_path,
                prim_path=prim_path,
                status="dependency_missing",
                error=(
                    "Isaac Sim APIs are not available in the current Python "
                    f"environment: {exc}"
                ),
                metadata={"fallback": "scene_import_not_executed"},
            )
        except Exception as exc:
            return SceneImportResult(
                usd_path=usd_path,
                prim_path=prim_path,
                status="error",
                error=str(exc),
                metadata={},
            )

    def _import_with_isaac(self, usd_path: str, prim_path: str) -> SceneImportResult:
        """
        Isaac-backed USD scene import.

        The exact API may differ depending on whether the script is launched
        inside Isaac Sim standalone app, IsaacLab, or a regular Python process.
        """
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        except Exception as exc:
            raise ImportError(exc) from exc

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        return SceneImportResult(
            usd_path=usd_path,
            prim_path=prim_path,
            status="success",
            error=None,
            metadata={"import_api": "omni.isaac.core.utils.stage.add_reference_to_stage"},
        )
