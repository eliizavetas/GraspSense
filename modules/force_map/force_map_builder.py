from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


MATERIAL_COEFFICIENTS = {
    "paper": 0.35,
    "plastic": 0.55,
    "glass": 0.85,
    "metal": 1.00,
    "rubber": 0.75,
    "unknown": 0.50,
}


@dataclass
class ForceMapResult:
    material: str
    layers: int
    angular_bins: int
    force_map: Any
    min_force: float
    max_force: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if np is not None and hasattr(self.force_map, "tolist"):
            data["force_map"] = self.force_map.tolist()
        return data


class ForceMapBuilder:
    """
    Lightweight first version of the GraspSense force-map builder.

    Current implementation produces a structured placeholder force map.
    It is intentionally simple and importable without Isaac Sim.

    TODO:
    - Apply PCA to mesh vertices to obtain the principal object frame.
    - Discretize the object surface into height layers and angular bins.
    - Estimate local wall thickness via raycasting.
    - Add rim/edge refinement for reinforced object boundaries.
    - Project force-map values back onto mesh vertices.
    - Export force-map values as USD primvars for Isaac Sim runtime lookup.
    """

    def __init__(
        self,
        height_layers: int = 20,
        angular_bins: int = 36,
        force_clamp: float = 10.0,
    ) -> None:
        self.height_layers = height_layers
        self.angular_bins = angular_bins
        self.force_clamp = force_clamp

    def build(
        self,
        mesh_path: Optional[str] = None,
        material: str = "unknown",
        base_force: float = 1.0,
    ) -> ForceMapResult:
        material_key = material.lower() if material else "unknown"
        material_coeff = MATERIAL_COEFFICIENTS.get(
            material_key,
            MATERIAL_COEFFICIENTS["unknown"],
        )

        if np is None:
            raise ImportError(
                "numpy is required for ForceMapBuilder. "
                "Install it with: pip install numpy"
            )

        force_map = self._generate_placeholder_map(
            base_force=base_force,
            material_coeff=material_coeff,
        )

        return ForceMapResult(
            material=material_key,
            layers=self.height_layers,
            angular_bins=self.angular_bins,
            force_map=force_map,
            min_force=float(force_map.min()),
            max_force=float(force_map.max()),
            metadata={
                "mesh_path": mesh_path,
                "material_coefficient": material_coeff,
                "force_clamp": self.force_clamp,
                "implementation": "placeholder",
                "note": (
                    "This is a lightweight placeholder force map. "
                    "Real PCA, raycasting, wall-thickness estimation, "
                    "and USD primvar export are TODO."
                ),
            },
        )

    def _generate_placeholder_map(self, base_force: float, material_coeff: float):
        layer_axis = np.linspace(0.0, 1.0, self.height_layers).reshape(-1, 1)
        angular_axis = np.linspace(0.0, 2.0 * np.pi, self.angular_bins).reshape(1, -1)

        # Slightly stronger rim/base zones and mild angular variation.
        rim_bonus = 0.6 * np.exp(-((layer_axis - 1.0) ** 2) / 0.02)
        base_bonus = 0.4 * np.exp(-((layer_axis - 0.0) ** 2) / 0.03)
        angular_variation = 0.1 * (1.0 + np.sin(angular_axis))

        force_map = base_force * (1.0 + rim_bonus + base_bonus + angular_variation)
        force_map = force_map * material_coeff
        force_map = np.clip(force_map, 0.0, self.force_clamp)

        return force_map
