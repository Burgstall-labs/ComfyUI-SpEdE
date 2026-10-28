#!/usr/bin/env python3
"""Create a deterministic, self-owned synthetic 3DGS/NPZ/mask test fixture.

No external datasets, people or model weights. Camera targets are an illustrated
proxy (not ground-truth renderer output); actual geometry must be judged with
a real gsplat GPU pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation


def _camera(position: np.ndarray, center: np.ndarray) -> np.ndarray:
    forward = center - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return np.linalg.inv(c2w).astype(np.float32)


def _project(points: np.ndarray, viewmat: np.ndarray, K: np.ndarray):
    camera = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = camera[:, 2]
    good = z > 0.01
    u = K[0, 0] * camera[:, 0] / np.maximum(z, 0.01) + K[0, 2]
    v = K[1, 1] * camera[:, 1] / np.maximum(z, 0.01) + K[1, 2]
    return np.column_stack([u[good], v[good]])


def _circle_surface(rng: np.random.Generator, count: int) -> np.ndarray:
    unit = rng.normal(size=(count, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    radii = rng.normal(0.43, 0.015, size=(count, 1))
    return unit * radii


def generate(output: Path, *, seed: int = 24, size: int = 192, views: int = 8) -> dict:
    if views < 4:
        raise ValueError("At least four real camera views are required")
    if size < 64:
        raise ValueError("Image size must be at least 64")
    output = Path(output).expanduser().resolve()
    mask_dir = output / "camera_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    object_xyz = _circle_surface(rng, 2600)
    floor_xz = rng.uniform(-1.6, 1.6, size=(2200, 2))
    floor_xyz = np.column_stack(
        [floor_xz[:, 0], np.full(len(floor_xz), -0.72), floor_xz[:, 1]]
    )
    coords = np.concatenate([object_xyz, floor_xyz]).astype(np.float32)
    total = len(coords)
    object_count = len(object_xyz)
    fields = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "synthetic_source_id",
    ]
    dtype = np.dtype([
        (name, "<i4" if name == "synthetic_source_id" else "<f4")
        for name in fields
    ])
    vertex = np.zeros(total, dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = coords[:, axis]
    # Graphdeco SH degree 0 coefficients, from RGB via C0.
    rgb = np.zeros((total, 3), np.float32)
    rgb[:object_count] = np.array([0.8, 0.16, 0.12])
    rgb[object_count:] = np.array([0.57, 0.55, 0.51])
    sh_dc = (rgb - 0.5) / 0.28209479177387814
    for i in range(3):
        vertex[f"f_dc_{i}"] = sh_dc[:, i]
    vertex["opacity"] = np.log(0.96 / 0.04)
    for name in ("scale_0", "scale_1", "scale_2"):
        vertex[name][:object_count] = np.log(0.026)
        vertex[name][object_count:] = np.log(0.058)
    vertex["rot_0"] = 1.0
    vertex["synthetic_source_id"] = np.arange(total, dtype=np.int32)
    scene_path = output / "synthetic_scene.ply"
    PlyData(
        [PlyElement.describe(vertex, "vertex")], text=False, byte_order="<",
        comments=["SpEdE synthetic demo fixture. All geometry procedural."],
    ).write(scene_path)

    target_images = []
    viewmats = []
    intrinsics = []
    labels = []
    for i in range(views):
        theta = 2 * np.pi * i / views
        position = np.array([
            2.4 * np.sin(theta), 0.8, 2.4 * np.cos(theta),
        ])
        viewmat = _camera(position, np.zeros(3))
        f = 0.5 * size / np.tan(np.deg2rad(65.0) / 2)
        K = np.array(
            [[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        pixels = _project(object_xyz, viewmat, K)
        hull = ConvexHull(pixels)
        polygon = [(float(x), float(y)) for x, y in pixels[hull.vertices]]
        im = Image.new("L", (size, size), 0)
        ImageDraw.Draw(im).polygon(polygon, fill=255)
        mask = binary_dilation(np.asarray(im) > 0, iterations=2)
        label = f"camera_{i}"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{label}.png")
        # Illustrated RGB proxies for Comfy mask-input experimentation only.
        proxy = np.full((size, size, 3), [145, 140, 130], dtype=np.uint8)
        proxy[mask] = np.array([215, 52, 44], dtype=np.uint8)
        target_images.append(proxy)
        labels.append(label)
        viewmats.append(viewmat)
        intrinsics.append(K)
    pack_path = output / "cameras.npz"
    np.savez_compressed(
        pack_path,
        targets=np.stack(target_images),
        labels=np.array(labels),
        viewmats=np.stack(viewmats),
        intrinsics=np.stack(intrinsics),
        frame_indices=np.arange(views, dtype=np.int32),
        face_size=np.int32(size),
    )
    known_path = output / "ground_truth_object_ids.npz"
    np.savez_compressed(
        known_path, object_ids=np.arange(object_count, dtype=np.int64)
    )
    manifest = {
        "purpose": "CPU asset tests and optional gsplat GPU smoke; not a photoreal quality test",
        "scene_ply": str(scene_path),
        "camera_targets_npz": str(pack_path),
        "mask_directory": str(mask_dir),
        "hero_view_id": labels[0],
        "camera_count": views,
        "total_gaussians": total,
        "object_gaussians": object_count,
        "known_object_ids_npz": str(known_path),
        "target_images": "illustrated proxies, not gsplat ground truth",
        "seed": seed,
    }
    (output / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/spede-fixture"))
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(generate(args.out, seed=args.seed, size=args.size, views=args.views), indent=2))


if __name__ == "__main__":
    main()
