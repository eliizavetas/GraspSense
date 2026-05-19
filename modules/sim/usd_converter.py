from dataclasses import dataclass, asdict
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

    This module is intentionally safe to import outside Isaac Sim.
    Isaac/IsaacLab imports must stay inside conversion methods.

    TODO:
    - Integrate Isaac Sim asset converter for GLB/OBJ/STL/FBX to USD.
    - Add rigid body settings.
    - Add collision approximation / SDF collider configuration.
    - Add mass and physics material assignment.
    - Validate USD asset placement in Isaac Sim scene.
    """

    def __init__(self, output_dir: str = "data/output/usd") -> None:
        self.output_dir = Path(output_dir)

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

        mesh_path_obj = Path(mesh_path)

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

        output_usd_path = self.output_dir / output_name

        try:
            return self._convert_with_isaac(
                mesh_path=mesh_path_obj,
                output_usd_path=output_usd_path,
                material=material,
            )
        except ImportError as exc:
            return USDConversionResult(
                input_mesh_path=str(mesh_path_obj),
                output_usd_path=None,
                status="dependency_missing",
                error=(
                    "Isaac Sim / IsaacLab conversion APIs are not available "
                    f"in the current Python environment: {exc}"
                ),
                metadata={
                    "material": material,
                    "fallback": "conversion_not_executed",
                    "expected_output_path": str(output_usd_path),
                },
            )
        except Exception as exc:
            return USDConversionResult(
                input_mesh_path=str(mesh_path_obj),
                output_usd_path=None,
                status="error",
                error=str(exc),
                metadata={
                    "material": material,
                    "expected_output_path": str(output_usd_path),
                },
            )

    def _convert_with_isaac(
        self,
        mesh_path: Path,
        output_usd_path: Path,
        material: str,
    ) -> USDConversionResult:
        """
        Isaac-backed conversion.

        This is a placeholder wrapper. The exact conversion API depends on
        the Isaac Sim / IsaacLab version installed in the working environment.
        Keep imports here, not at module import time.
        """
        try:
            # Reference implementation should be adapted from:
            # sandbox/legacy_sources/scripts/tools/convert_mesh.py
            # sandbox/legacy_sources/scripts/tools/prepare_asset.py
            from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
        except Exception as exc:
            raise ImportError(exc) from exc

        cfg = MeshConverterCfg(
            asset_path=str(mesh_path),
            usd_dir=str(output_usd_path.parent),
            usd_file_name=output_usd_path.name,
            force_usd_conversion=True,
        )

        converter = MeshConverter(cfg)
        usd_path = getattr(converter, "usd_path", str(output_usd_path))

        return USDConversionResult(
            input_mesh_path=str(mesh_path),
            output_usd_path=str(usd_path),
            status="success",
            error=None,
            metadata={
                "material": material,
                "converter": "isaaclab.sim.converters.MeshConverter",
            },
        )
