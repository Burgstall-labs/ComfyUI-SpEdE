"""Gaussian PLY to the 32-byte ``.splat`` format most web viewers read.

Position, scale, RGBA and rotation per Gaussian; view-dependent colour (SH
bands above 0) is dropped. Used for the UI's 3D views (three.js +
GaussianSplats3D).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def ply_to_splat(src: Path, dst: Path) -> int:
    """Write ``dst`` from the Gaussians in ``src``; returns how many were kept."""
    v = PlyData.read(str(src))["vertex"].data
    scales = np.exp(np.c_[v["scale_0"], v["scale_1"], v["scale_2"]]).astype(np.float32)
    rgb = np.clip((0.5 + SH_C0 * np.c_[v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]) * 255, 0, 255)
    alpha = np.clip(255.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64))), 0, 255)
    quat = np.c_[v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]].astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    # Largest, most opaque first: progressive loading shows the scene early.
    order = np.argsort(-(scales.prod(axis=1) * alpha / 255.0))
    records = np.zeros(len(v), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    records["p"] = np.c_[v["x"], v["y"], v["z"]][order]
    records["s"] = scales[order]
    records["c"] = np.c_[rgb, alpha].astype(np.uint8)[order]
    records["r"] = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)[order]
    # Invisible Gaussians are not worth sending to the viewer.
    records = records[records["c"][:, 3] > 0]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    records.tofile(tmp)
    tmp.replace(dst)                      # a viewer loading mid-export sees the old file
    return len(records)
