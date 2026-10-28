"""Independent Graphdeco Gaussian PLY reader and property-preserving writers.

No AccGS dependency. All vertex properties are retained for existing splats;
newly born Gaussians use the original dtype, preserving SH layout and extras.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814
REQUIRED = {
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
}


def _sh_names(names: tuple[str, ...]):
    return sorted((p for p in names if p.startswith("f_rest_")),
                  key=lambda x: int(x.split("_")[-1]))


@dataclass(frozen=True)
class GaussianPly:
    path: Path
    vertex: np.ndarray
    comments: list[str]
    dc: np.ndarray
    rest: np.ndarray
    means: np.ndarray
    log_scales: np.ndarray
    quaternions: np.ndarray
    opacity_logits: np.ndarray
    max_sh_degree: int

    @classmethod
    def load(cls, path: Path):
        path = Path(path).expanduser().resolve(strict=True)
        original = PlyData.read(str(path))
        if len(original.elements) != 1 or original.elements[0].name != "vertex":
            raise ValueError("SpEdE requires vertex-only Gaussian PLY")
        vertex = original["vertex"].data
        fields = tuple(vertex.dtype.names or ())
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError(f"Missing Gaussian properties: {sorted(missing)}")
        if not len(vertex):
            raise ValueError("Empty Gaussian PLY")
        rest_names = _sh_names(fields)
        if len(rest_names) % 3:
            raise ValueError("Invalid SH coefficient layout")
        coeff_count = 1 + len(rest_names) // 3
        degree = int(round(np.sqrt(coeff_count) - 1))
        if not 0 <= degree <= 3 or (degree+1)**2 != coeff_count:
            raise ValueError("Only SH degrees 0–3 supported")
        if rest_names:
            flattened = np.column_stack([vertex[x] for x in rest_names])
            rest = flattened.astype(np.float32).reshape(len(vertex), 3, coeff_count - 1).transpose(0, 2, 1)
        else:
            rest = np.zeros((len(vertex), 0, 3), np.float32)
        return cls(
            path=path,
            vertex=vertex,
            comments=list(original.comments),
            dc=np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)[:, None, :],
            rest=rest,
            means=np.stack([vertex[x] for x in ("x","y","z")], axis=-1).astype(np.float32),
            log_scales=np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32),
            quaternions=np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32),
            opacity_logits=np.asarray(vertex["opacity"], np.float32)[:, None],
            max_sh_degree=degree,
        )


def write_vertices(reference: GaussianPly, destination: Path, vertices: np.ndarray, message: str):
    destination = Path(destination).expanduser().resolve()
    if destination == reference.path:
        raise ValueError("Refusing to overwrite input source PLY")
    if vertices.dtype != reference.vertex.dtype:
        raise ValueError("Output vertex dtype differs from source")
    if not len(vertices):
        raise ValueError("Cannot write empty Gaussian scene")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".ply",
                                         prefix=".spede-", delete=False) as f:
            staging = Path(f.name)
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            text=False, byte_order="<",
            comments=list(reference.comments) + [message],
        ).write(str(staging))
        os.replace(staging, destination)
    finally:
        if staging and staging.exists():
            staging.unlink()


def remove_rows(source: Path, ids: np.ndarray, output: Path):
    ply = GaussianPly.load(source)
    ids = np.unique(np.asarray(ids, np.int64).reshape(-1))
    if not len(ids) or ids[0] < 0 or ids[-1] >= len(ply.vertex):
        raise ValueError("Invalid or empty object ID selection")
    keep = np.ones(len(ply.vertex), dtype=bool)
    keep[ids] = False
    if not np.any(keep):
        raise ValueError("Removing all Gaussians is not supported")
    write_vertices(ply, output, ply.vertex[keep].copy(),
                   f"SpEdE physically removed {len(ids)} source Gaussian rows")
    return {
        "source_gaussians": len(ply.vertex),
        "removed_gaussians": len(ids),
        "remaining_gaussians": int(keep.sum()),
        "object_removed": True,
        "surface_repaired": False,
        "scene_ply": str(Path(output).resolve()),
    }


def append_planar_vertices(source: Path, output: Path, means: np.ndarray,
                           rgb: np.ndarray, tangent_radius: np.ndarray,
                           *, opacity: float = 0.92, thickness_fraction: float = 0.15,
                           quaternions: np.ndarray | None = None):
    ply = GaussianPly.load(source)
    xyz = np.asarray(means, np.float32)
    colors = np.asarray(rgb, np.float32)
    radius = np.asarray(tangent_radius, np.float32)
    if radius.ndim == 1:
        radius = np.repeat(radius[:, None], 2, axis=1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or colors.shape != xyz.shape or radius.shape != (len(xyz), 2):
        raise ValueError("Gaussian births require aligned Nx3 xyz/rgb and N or Nx2 radii")
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(colors).all():
        raise ValueError("Births must be finite and non-empty")
    if np.any(radius <= 0) or not np.isfinite(radius).all():
        raise ValueError("Gaussian radii must be finite and positive")
    if not (0 < opacity < 1):
        raise ValueError("Opacity must lie in (0,1)")
    born = np.zeros(len(xyz), dtype=ply.vertex.dtype)
    for i, key in enumerate(("x", "y", "z")):
        born[key] = xyz[:, i]
    for i in range(3):
        born[f"f_dc_{i}"] = (np.clip(colors[:, i], 0, 1) - 0.5) / SH_C0
    born["opacity"] = float(np.log(opacity / (1.0 - opacity)))
    born["scale_0"] = np.log(radius[:, 0])
    born["scale_1"] = np.log(radius[:, 1])
    born["scale_2"] = np.log(radius.min(axis=1) * thickness_fraction)
    qs = np.zeros((len(xyz), 4), np.float32)
    qs[:, 0] = 1
    if quaternions is not None:
        qs = np.asarray(quaternions, np.float32)
        if qs.shape != (len(xyz), 4):
            raise ValueError("Quaternions must have shape Nx4")
    for i in range(4):
        born[f"rot_{i}"] = qs[:, i]
    all_rows = np.concatenate([ply.vertex, born])
    write_vertices(ply, output, all_rows,
                   f"SpEdE planar patch: added {len(xyz)} Gaussians")
    return {"source_gaussians": int(len(ply.vertex)), "added_gaussians": int(len(xyz)),
            "total_gaussians": int(len(all_rows)), "scene_ply": str(Path(output).resolve())}
