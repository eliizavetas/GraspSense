#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert mesh to USD using IsaacLab MeshConverter.")
parser.add_argument("--input", required=True, help="Input mesh path, e.g. object.glb")
parser.add_argument("--output", required=True, help="Output USD path")
parser.add_argument("--collision-approximation", default="sdf")
parser.add_argument("--mass", type=float, default=0.5)
parser.add_argument("--static-friction", type=float, default=0.6)
parser.add_argument("--dynamic-friction", type=float, default=0.4)
parser.add_argument("--restitution", type=float, default=0.0)
parser.add_argument("--make-instanceable", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdShade, UsdPhysics, PhysxSchema, Sdf

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path


collision_approximation_map = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
    "none": None,
}


def ensure_physics_material(stage, path, static_friction, dynamic_friction, restitution):
    # Create /World only for materials; the converted asset itself may live under /object.
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/PhysicsMaterials", "Xform")

    mat_prim = stage.GetPrimAtPath(path)
    if not mat_prim or not mat_prim.IsValid():
        mat_prim = stage.DefinePrim(path, "Material")

    UsdPhysics.MaterialAPI.Apply(mat_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

    def set_attr(name, value, type_name):
        attr = mat_prim.GetAttribute(name)
        if not attr:
            attr = mat_prim.CreateAttribute(name, type_name)
        attr.Set(value)

    set_attr("physxMaterial:staticFriction", float(static_friction), Sdf.ValueTypeNames.Float)
    set_attr("physxMaterial:dynamicFriction", float(dynamic_friction), Sdf.ValueTypeNames.Float)
    set_attr("physxMaterial:restitution", float(restitution), Sdf.ValueTypeNames.Float)

    return UsdShade.Material(mat_prim)


def find_asset_root(stage):
    # Prefer /World child if it exists.
    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        children = [
            child for child in world.GetChildren()
            if child.GetName() not in ["Looks", "PhysicsMaterials", "physicsScene", "PhysicsScene"]
        ]
        if children:
            return children[0]

    # Otherwise use first non-utility root prim, e.g. /object from GLB conversion.
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.GetName() not in ["World", "Looks", "PhysicsMaterials", "physicsScene", "PhysicsScene"]:
            return prim

    raise RuntimeError("Could not find asset root prim in converted USD.")


def main():
    mesh_path = Path(args_cli.input).resolve()
    output_path = Path(args_cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "input_mesh_path": str(mesh_path),
        "output_usd_path": str(output_path),
        "asset_root_path": None,
        "collision_approximation": args_cli.collision_approximation,
        "mass": args_cli.mass,
        "error": None,
    }

    try:
        if not check_file_path(str(mesh_path)):
            raise ValueError(f"Invalid mesh file path: {mesh_path}")

        cfg_class = collision_approximation_map.get(args_cli.collision_approximation)
        if cfg_class is None and args_cli.collision_approximation != "none":
            raise ValueError(f"Unsupported collision approximation: {args_cli.collision_approximation}")

        collision_cfg = cfg_class() if cfg_class is not None else None

        mesh_converter_cfg = MeshConverterCfg(
            mass_props=schemas_cfg.MassPropertiesCfg(mass=args_cli.mass),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(
                collision_enabled=args_cli.collision_approximation != "none"
            ),
            asset_path=str(mesh_path),
            force_usd_conversion=True,
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            make_instanceable=args_cli.make_instanceable,
            mesh_collision_props=collision_cfg,
        )

        mesh_converter = MeshConverter(mesh_converter_cfg)
        usd_path = Path(mesh_converter.usd_path)

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Could not open converted USD: {usd_path}")

        root = find_asset_root(stage)

        material = ensure_physics_material(
            stage,
            "/World/PhysicsMaterials/PM_Object",
            args_cli.static_friction,
            args_cli.dynamic_friction,
            args_cli.restitution,
        )

        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

        stage.GetRootLayer().Save()

        result["success"] = True
        result["output_usd_path"] = str(usd_path)
        result["asset_root_path"] = str(root.GetPath())

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2))
    simulation_app.close()
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
