# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Small geometry helpers every stage shares: world up, a ground frame,
projection to pixels, unprojection of depth, the depth of a plane per pixel.
(Each had three to eight copies across the engine, removal and hull code.)"""
from __future__ import annotations

import numpy as np


def world_up_from_cameras(camera_to_worlds) -> np.ndarray:
    """World up from real cameras, insensitive to their pitch.

    A photographer keeps the camera level (no roll) far more reliably than
    horizontal: the cameras' x axes are therefore horizontal, and up is the
    direction most orthogonal to all of them (the small-eigenvalue space of
    sum x x^T). Within that space the mean camera up (-Y) picks the direction.
    An orbit or any capture with cameras turned about the vertical pins up
    exactly, whatever their pitch; a strictly forward-facing capture (all x
    axes parallel, a 2-D null space) cannot tell pitch from gravity and gets
    the mean camera up, as before. The mean of -Y alone tilted by the pitch
    of every capture that looks down at a table."""
    c2ws = [np.asarray(c, np.float64) for c in camera_to_worlds]
    if not c2ws:
        raise ValueError("at least one camera is required to estimate world up")
    xs = np.stack([c[:3, 0] / max(float(np.linalg.norm(c[:3, 0])), 1e-12) for c in c2ws])
    mean_up = -np.mean([c[:3, 1] for c in c2ws], axis=0)
    if float(np.linalg.norm(mean_up)) < 1e-6:
        raise ValueError("camera up axes cancel out; cannot estimate world up")
    values, vectors = np.linalg.eigh(xs.T @ xs)        # ascending
    # Directions the x axes do not span: the two smallest; a single camera or
    # a straight-line capture leaves both free, an orbit leaves one.
    span = vectors[:, :2] if values[1] < 0.05 * max(values[2], 1e-12) else vectors[:, :1]
    up = span @ (span.T @ mean_up)
    if float(np.linalg.norm(up)) < 1e-6:
        up = mean_up
    return (up / float(np.linalg.norm(up))).astype(np.float32)


def ground_axes(world_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, axis_a, axis_b): an orthonormal frame with the ground plane
    spanned by axis_a and axis_b."""
    up = np.asarray(world_up, np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    axis_a = np.cross(up, np.eye(3)[int(np.argmin(np.abs(up)))])
    axis_a /= np.linalg.norm(axis_a)
    return up, axis_a, np.cross(up, axis_a)


def project_to_pixels(points: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) [N, 2] (continuous, pixel (0, 0) spans
    0..1) and camera depth z [N] of world points; u, v are NaN behind the
    camera."""
    view = np.asarray(viewmat, np.float64)
    k = np.asarray(intrinsic, np.float64)
    cam = np.asarray(points, np.float64) @ view[:3, :3].T + view[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(z > 1e-9, z, np.nan)
        uv = np.stack([cam[:, 0] / safe * k[0, 0] + k[0, 2], cam[:, 1] / safe * k[1, 1] + k[1, 2]], axis=1)
    return uv, z


def unproject_depth(depth: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray,
                    pixels: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """World points of a camera-z depth map: [H, W, 3] for the whole map, or
    [N, 3] for the given (ys, xs) pixels (``depth`` then has shape [N] or
    [H, W]). Pixel centres are at +0.5."""
    depth = np.asarray(depth, np.float64)
    k = np.asarray(intrinsic, np.float64)
    c2w = np.asarray(camera_to_world, np.float64)
    if pixels is None:
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        z = depth
    else:
        ys, xs = (np.asarray(p) for p in pixels)
        z = depth[ys, xs] if depth.ndim == 2 else depth
    cam = np.stack([(xs + 0.5 - k[0, 2]) / k[0, 0] * z, (ys + 0.5 - k[1, 2]) / k[1, 1] * z, z], axis=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def plane_depth_map(viewmat: np.ndarray, intrinsic: np.ndarray, width: int, height: int,
                    normal: np.ndarray, offset: float) -> np.ndarray:
    """Camera-z depth of the plane n.x = offset at each pixel; NaN where the
    ray misses it (behind the camera or parallel)."""
    k = np.asarray(intrinsic, np.float64)
    c2w = np.linalg.inv(np.asarray(viewmat, np.float64))
    n = np.asarray(normal, np.float64)
    ys, xs = np.mgrid[0:int(height), 0:int(width)]
    rays = np.stack([(xs + 0.5 - k[0, 2]) / k[0, 0], (ys + 0.5 - k[1, 2]) / k[1, 1], np.ones(xs.shape)], -1)
    denom = (rays @ c2w[:3, :3].T) @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(offset) - float(c2w[:3, 3] @ n)) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return np.where(t > 0, t, np.nan)
