# SPDX-License-Identifier: AGPL-3.0-or-later
# The largest-component cleanup, the rotation helpers and the silhouette
# registration idea are ported by the copyright holder (Burgstall Labs) from
# Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612
# (splatedit/generated_object.py); released under AGPL-3.0-or-later per the
# owner decision of 2026-10-08 (see NOTICE).
"""Replace: put a generated object where the removed one stood.

Qwen-Image 2.1 draws the new object into the cleaned hero view, SAM 3.1 cuts
it out and TripoSplat turns the cut-out into Gaussians in its own normalised
frame. This module grounds those Gaussians in the scene with what the scene
already knows (kitchen spike, 26 Oct 2026):

- the support plane of the ORIGINAL removed object: a robust plane through
  the unselected Gaussians in a ring around its footprint, world up from the
  capture cameras (as spede_core.support_trim);
- the yaw about the plane normal: the silhouette against Qwen's mask, front
  and back told apart by colour against Qwen's picture;
- the base on the plane, the horizontal position where Qwen drew it (the
  silhouette centroid) but at most ``max_shift`` from the old footprint's
  centre, a uniform scale that matches the silhouette's area.

Replace v2 (27 Oct 2026) is an inverse extract: the object goes into the
SOURCE scene without exactly the extracted object (``object_rows``), not
into Remove's result, whose dilated masks and carved support left a raised
fill patch with rims far larger than the old object. ``replace_fill`` fills
only the part of the old contact footprint the new object leaves bare (the
Remove fill laid flat when there is one, else plain discs coloured from the
surface round it), and ``size_scale`` sizes the new object from the old one
(the size modes). The contact shadow is procedural: flat dark Gaussians from
the blurred footprint of the object's lowest part, not Qwen's darkening
(which is in the wrong place once the object is in 3D).

Everything is numpy; the node passes a gsplat render where one is wanted
(``render``), the CPU tests use the point render here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry_utils import ground_axes
from .ply import SH_C0, GaussianPly, write_vertices
from .support_trim import _axes, _local_colour_distance, _robust_plane

# TripoSplat's normalised frame has its up along -Y (splatedit's ComfyTripoSplatGenerator).
TRIPOSPLAT_UP = np.array([0.0, -1.0, 0.0])


@dataclass(frozen=True)
class ObjectGaussians:
    means: np.ndarray           # (N, 3)
    log_scales: np.ndarray      # (N, 3)
    quaternions: np.ndarray     # (N, 4) wxyz
    opacity_logits: np.ndarray  # (N,)
    rgb: np.ndarray             # (N, 3) in [0, 1]

    @property
    def count(self) -> int:
        return len(self.means)

    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logits))

    def subset(self, keep) -> "ObjectGaussians":
        keep = np.asarray(keep)
        return ObjectGaussians(self.means[keep], self.log_scales[keep], self.quaternions[keep],
                               self.opacity_logits[keep], self.rgb[keep])

    @staticmethod
    def empty() -> "ObjectGaussians":
        return ObjectGaussians(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3)))

    @staticmethod
    def concat(*parts: "ObjectGaussians") -> "ObjectGaussians":
        return ObjectGaussians(*(np.concatenate([getattr(p, f) for p in parts])
                                 for f in ("means", "log_scales", "quaternions", "opacity_logits", "rgb")))


def ply_gaussians(ply: GaussianPly, rows=None) -> ObjectGaussians:
    """Rows of a loaded Graphdeco PLY (all by default), colour from the DC
    term only."""
    pick = slice(None) if rows is None else np.asarray(rows)
    q = ply.quaternions[pick].astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dc = ply.dc[pick].reshape(len(q), -1)[:, :3]
    return ObjectGaussians(ply.means[pick].astype(np.float64), ply.log_scales[pick].astype(np.float64), q,
                           ply.opacity_logits.reshape(-1)[pick].astype(np.float64),
                           np.clip(dc.astype(np.float64) * SH_C0 + 0.5, 0.0, 1.0))


def load_object_ply(path: Path) -> ObjectGaussians:
    """A Graphdeco PLY (TripoSplat's via SplatToFile3D is SH degree 0)."""
    return ply_gaussians(GaussianPly.load(path))


def keep_largest_component(obj: ObjectGaussians, *, voxel_fraction: float = 0.015,
                           min_opacity: float = 0.02) -> tuple[ObjectGaussians, dict]:
    """The largest 26-connected cluster of opaque Gaussians: generators also
    build flat cards from background seen through gaps in the object. The
    voxel is a fraction of the object's robust extent."""
    opaque = obj.opacity >= min_opacity
    kept = obj.subset(opaque)
    if kept.count == 0:
        raise ValueError("the generated object has no opaque Gaussians")
    extent = float(np.max(np.percentile(kept.means, 99, axis=0) - np.percentile(kept.means, 1, axis=0)))
    voxel = max(extent * float(voxel_fraction), 1e-9)
    cells = np.floor((kept.means - kept.means.min(axis=0)) / voxel).astype(np.int64)
    grid = np.zeros(cells.max(axis=0) + 1, bool)
    grid[tuple(cells.T)] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(grid, labels, range(1, count + 1))
    in_main = labels[tuple(cells.T)] == 1 + int(np.argmax(sizes))
    return kept.subset(in_main), {"input_gaussians": obj.count, "transparent_dropped": int((~opaque).sum()),
                                  "components": int(count), "kept_gaussians": int(in_main.sum()), "voxel": voxel}


# ---- rotations --------------------------------------------------------------------

def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction a to direction b."""
    a = np.asarray(a, np.float64) / max(float(np.linalg.norm(a)), 1e-12)
    b = np.asarray(b, np.float64) / max(float(np.linalg.norm(b)), 1e-12)
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perpendicular = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(a, [0.0, 1.0, 0.0])
        return rotation_about_axis(perpendicular, math.pi)
    return rotation_about_axis(axis, math.atan2(s, c))


def quaternion_from_matrix(r: np.ndarray) -> np.ndarray:
    """wxyz of a rotation matrix (Shepperd: the largest component first)."""
    r = np.asarray(r, np.float64)
    trace = np.trace(r)
    candidates = [1 + trace, 1 + r[0, 0] - r[1, 1] - r[2, 2], 1 - r[0, 0] + r[1, 1] - r[2, 2],
                  1 - r[0, 0] - r[1, 1] + r[2, 2]]
    i = int(np.argmax(candidates))
    s = 2.0 * math.sqrt(max(candidates[i], 1e-12))
    if i == 0:
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif i == 1:
        q = [(r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif i == 2:
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s]
    else:
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4]
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1.0 if q[0] >= 0 else -1.0)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=1)


# ---- the support plane ----------------------------------------------------------------

@dataclass(frozen=True)
class SupportPlane:
    """``normal`` (upward) · x = ``offset`` under the removed object."""
    normal: np.ndarray
    offset: float
    footprint_centre: np.ndarray     # on the plane
    footprint_size: float            # the old footprint's larger side (2–98 % box)
    object_height: float             # the removed object's height (1–99 %)
    spread: float
    ring: int
    tilt_deg: float
    # The old footprint's principal extents (2–98 %, along its own axes): the
    # size modes compare the new object's with them (``size_scale``).
    footprint_major: float = 0.0
    footprint_minor: float = 0.0

    @property
    def dominant_extent(self) -> float:
        """The removed object's dominant extent (``dominant_extent``)."""
        if self.footprint_major > 0:
            return dominant_extent(self.object_height, self.footprint_major, self.footprint_minor)
        return max(self.object_height, self.footprint_size)

    def height(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, np.float64) @ self.normal - self.offset

    def project(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, np.float64)
        return point - self.height(point[None])[0] * self.normal

    def as_report(self) -> dict:
        return {"normal": [float(v) for v in self.normal], "offset": float(self.offset),
                "footprint_centre": [float(v) for v in self.footprint_centre],
                "footprint_size": self.footprint_size, "object_height": self.object_height,
                "spread": self.spread, "ring_gaussians": self.ring, "tilt_deg": self.tilt_deg,
                "footprint_major": self.footprint_major, "footprint_minor": self.footprint_minor}


def support_plane(means: np.ndarray, opacity_logits: np.ndarray, object_ids: np.ndarray, world_up: np.ndarray, *,
                  ring_inner_fraction: float = 0.02, ring_outer_fraction: float = 0.15,
                  ring_height_fraction: float = 0.3, minimum_opacity: float = 0.3,
                  minimum_ring: int = 200) -> SupportPlane:
    """The surface the removed object stood on, from the SOURCE scene: the
    unselected opaque Gaussians in a ring just outside its footprint near its
    base height, a robust plane through them (support_trim's ring and fit)."""
    means = np.asarray(means, np.float64)
    ids = np.unique(np.asarray(object_ids, np.int64))
    up, e1, e2 = _axes(world_up)
    height = means @ up
    uv = np.stack([means @ e1, means @ e2], axis=1)
    low, high = np.percentile(height[ids], [1.0, 99.0])
    span = float(high - low)
    footprint = uv[ids]
    diagonal = float(np.linalg.norm(np.ptp(footprint, axis=0)))
    if span <= 0 or diagonal <= 0:
        raise ValueError("the removed object has no extent")
    others = np.ones(len(means), bool)
    others[ids] = False
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logits, np.float64).reshape(-1)))
    near = np.flatnonzero(others & (np.abs(height - low) <= ring_height_fraction * span) & (opacity >= minimum_opacity))
    gap, _ = cKDTree(footprint).query(uv[near])
    ring = near[(gap > ring_inner_fraction * diagonal) & (gap <= ring_outer_fraction * diagonal)]
    if len(ring) < minimum_ring:
        raise ValueError(f"only {len(ring)} scene Gaussians around the removed object's footprint: "
                         "no surface to stand the new object on")
    coef, spread, _inlier = _robust_plane(uv[ring], height[ring])
    # height = a u + b v + c  ->  n . x = d with n = up - a e1 - b e2
    normal = up - coef[0] * e1 - coef[1] * e2
    norm = float(np.linalg.norm(normal))
    lo, hi = np.percentile(footprint, 2, axis=0), np.percentile(footprint, 98, axis=0)
    centre_uv = (lo + hi) / 2
    centre = centre_uv[0] * e1 + centre_uv[1] * e2 + (coef[0] * centre_uv[0] + coef[1] * centre_uv[1] + coef[2]) * up
    major, minor = footprint_extents(footprint)
    return SupportPlane(normal / norm, float(coef[2] / norm), centre, float(np.max(hi - lo)), span,
                        float(spread), int(len(ring)), float(np.degrees(np.arccos(min(1.0, abs(normal @ up) / norm)))),
                        major, minor)


def dominant_extent(height: float, major: float, minor: float) -> float:
    """An object's size for the size modes: the larger of its height and its
    footprint's width, the width being sqrt(length x width) (the side of a
    square of the same area). Kitchen: the lego bulldozer is 1.66 long and
    0.75 wide; a round bowl as wide as it is long has 1.7 times its
    footprint and dwarfs it, one as wide as the square has the same."""
    return max(float(height), math.sqrt(max(float(major), 0.0) * max(float(minor), 0.0)))


def footprint_extents(uv: np.ndarray) -> tuple[float, float]:
    """A footprint's length and width (2–98 %) along its own principal axes,
    so that neither depends on how the object is turned."""
    uv = np.asarray(uv, np.float64)
    if len(uv) < 3:
        return 0.0, 0.0
    centred = uv - uv.mean(axis=0)
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    along = centred @ vectors[:, ::-1]
    extent = np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0)
    return float(extent[0]), float(extent[1])


def fill_rows(scene_means: np.ndarray, source_means: np.ndarray) -> np.ndarray:
    """Rows of the current scene that are not source Gaussians: the Remove
    fill's births and anything an all-around fill optimised or added. By
    position (bit for bit), not by row number: the all-around fill writes
    its trained Gaussians after the others, so rows past the kept source
    count are not all births there."""
    def rows(a):
        a = np.ascontiguousarray(np.asarray(a, np.float32))
        return a.view(np.dtype((np.void, a.dtype.itemsize * 3))).ravel()
    return ~np.isin(rows(scene_means), rows(source_means))


# ---- the hero camera: projection, silhouettes, renders -------------------------------------

def _project(points: np.ndarray, viewmat: np.ndarray, k: np.ndarray):
    cam = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / z[:, None] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
    return uv, z


def point_render(obj: ObjectGaussians, viewmat: np.ndarray, intrinsic: np.ndarray, width: int, height: int,
                 *, dilate: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """CPU stand-in for a splat render: the nearest opaque Gaussian's colour
    per pixel and a filled silhouette as alpha."""
    viewmat, k = np.asarray(viewmat, np.float64), np.asarray(intrinsic, np.float64)
    keep = obj.opacity >= 0.3
    uv, z = _project(obj.means[keep], viewmat, k)
    ok = (z > 0) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    px, zs, colours = uv[ok].astype(np.int64), z[ok], obj.rgb[keep][ok]
    order = np.argsort(-zs)                         # far first: the nearest is written last
    rgb = np.zeros((height, width, 3))
    rgb[px[order, 1], px[order, 0]] = colours[order]
    hit = np.zeros((height, width), bool)
    hit[px[:, 1], px[:, 0]] = True
    if dilate > 0 and hit.any():
        grown = ndimage.binary_dilation(hit, iterations=dilate)
        filled = ndimage.grey_dilation(rgb, size=(2 * dilate + 1, 2 * dilate + 1, 1))
        rgb = np.where(hit[..., None], rgb, filled)
        hit = grown
    return rgb, ndimage.binary_fill_holes(hit).astype(np.float64)


def gsplat_render(device: str = "cuda"):
    """A ``render`` for place_object on the GPU (the node's)."""
    def render(obj: ObjectGaussians, viewmat, intrinsic, width, height):
        import torch
        from gsplat import rasterization

        def t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        out, alpha, _meta = rasterization(
            means=t(obj.means), quats=t(obj.quaternions), scales=t(np.exp(obj.log_scales)), opacities=t(obj.opacity),
            colors=t(obj.rgb), viewmats=t(viewmat)[None], Ks=t(intrinsic)[None], width=int(width), height=int(height))
        return out[0].clamp(0, 1).cpu().numpy().astype(np.float64), alpha[0, ..., 0].cpu().numpy().astype(np.float64)
    return render


def _ray(px: np.ndarray, viewmat: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.linalg.inv(viewmat)
    d = c2w[:3, :3] @ np.array([(px[0] - k[0, 2]) / k[0, 0], (px[1] - k[1, 2]) / k[1, 1], 1.0])
    return c2w[:3, 3], d / np.linalg.norm(d)


def _ray_plane(px: np.ndarray, viewmat: np.ndarray, k: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray | None:
    origin, d = _ray(px, viewmat, k)
    denom = float(normal @ d)
    if abs(denom) < 1e-9:
        return None
    t = (offset - float(normal @ origin)) / denom
    return origin + t * d if t > 0 else None


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


# ---- placement ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Frame:
    """The object rotated into the scene, before scale and position: local
    points relative to its base centre (base on z=0 of the plane frame)."""
    rotation: np.ndarray
    local: np.ndarray          # (N, 3) rotated means minus the base centre
    height: float              # 1–99 % along the plane normal, unscaled


def _frame(means: np.ndarray, opaque: np.ndarray, normal: np.ndarray, base_rotation: np.ndarray, yaw_deg: float,
           origin: np.ndarray) -> _Frame:
    rotation = rotation_about_axis(normal, math.radians(yaw_deg)) @ base_rotation
    y = (means - origin) @ rotation.T
    _up, a1, a2 = ground_axes(normal)
    h = y[opaque] @ normal
    p1, p2 = y[opaque] @ a1, y[opaque] @ a2
    lo = np.percentile(h, 1.0)
    base = ((np.percentile(p1, 2) + np.percentile(p1, 98)) / 2 * a1
            + (np.percentile(p2, 2) + np.percentile(p2, 98)) / 2 * a2 + lo * normal)
    return _Frame(rotation, y - base, float(np.percentile(h, 99.0) - lo))


def _fit(frame: _Frame, silhouette, *, plane: SupportPlane, viewmat, k, target: np.ndarray, scale: float,
         max_shift: float, iterations: int) -> tuple[float, np.ndarray, bool, np.ndarray]:
    """Scale (silhouette area) and position on the plane (silhouette
    centroid, at most ``max_shift`` from the old footprint centre) for one
    rotation; ``silhouette(scale, base position)`` is the object's mask in
    the camera. Returns scale, base position, whether the shift was capped and
    the last silhouette."""
    area = float(target.sum())
    want = _centroid(target)
    position = plane.footprint_centre.copy()
    capped = False
    sil = None
    for _ in range(iterations):
        sil = silhouette(scale, position)
        got = float(sil.sum())
        if got < 1:
            # Off screen or behind the camera: back to the footprint, the old object's size.
            position, scale = plane.footprint_centre.copy(), plane.object_height / max(frame.height, 1e-9)
            continue
        scale *= float(np.clip(math.sqrt(area / got), 0.5, 2.0))
        sil = silhouette(scale, position)
        if not sil.any():
            continue
        # The centroids' rays cut at the object's mid-height: the in-plane move
        # that takes the rendered centroid onto Qwen's.
        mid = plane.offset + 0.5 * scale * frame.height
        a = _ray_plane(_centroid(sil), viewmat, k, plane.normal, mid)
        b = _ray_plane(want, viewmat, k, plane.normal, mid)
        if a is None or b is None:
            break
        shift = b - a
        shift -= (shift @ plane.normal) * plane.normal
        position = position + shift
        offset = position - plane.footprint_centre
        distance = float(np.linalg.norm(offset))
        capped = distance > max_shift
        if capped:
            position = plane.footprint_centre + offset * (max_shift / distance)
    sil = silhouette(scale, plane.project(position))
    return scale, plane.project(position), capped, sil


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


SIZE_MODES = ("natural", "match", "larger")


def size_scale(frame: _Frame, opaque: np.ndarray, plane: SupportPlane, mode: str, *, larger: float = 1.3,
               cover_boost: float = 1.25) -> tuple[float, dict]:
    """The uniform scale of a size mode for the object in ``frame`` (unscaled).

    "match": the new object's dominant extent (``dominant_extent``: the
    larger of its height and its footprint's width) becomes the removed
    object's. That is what reads as "as big as": a vase as tall as the old
    object was big, a bowl with the old footprint's area. Matching heights
    alone leaves a flat object tiny where a long one stood (the bowl at 0.41x
    on kitchen); matching the footprint's length makes a round object dwarf a
    long narrow one; matching footprints makes a vase a tower. A
    footprint-dominant object (its footprint at least as wide as it is tall)
    is also grown to cover the old footprint's length and width where that
    takes at most ``cover_boost`` more; a tall one is not (it would have to be
    much taller than anything that stood there), the fill covers the rest.
    "larger": ``larger`` times "match"."""
    if mode not in SIZE_MODES or mode == "natural":
        raise ValueError(f"size mode {mode!r}: one of {SIZE_MODES[1:]} has a scale of its own")
    _up, a1, a2 = ground_axes(plane.normal)
    local = frame.local[opaque]
    major, minor = footprint_extents(np.stack([local @ a1, local @ a2], axis=1))
    dominant = dominant_extent(frame.height, major, minor)
    scale = plane.dominant_extent / max(dominant, 1e-9)
    wide = math.sqrt(major * minor) >= frame.height
    cover = None
    if wide and minor > 0 and plane.footprint_major > 0:
        cover = max(plane.footprint_major / max(major, 1e-9), plane.footprint_minor / minor)
        scale = min(max(scale, cover), cover_boost * scale)
    if mode == "larger":
        scale *= float(larger)
    return float(scale), {"removed_dominant_extent": plane.dominant_extent,
                          "new_dominant_extent_unscaled": float(dominant),
                          "footprint_dominant": bool(wide),
                          "cover_scale": None if cover is None else float(cover)}


def place_object(obj: ObjectGaussians, *, plane: SupportPlane, viewmat: np.ndarray, intrinsic: np.ndarray,
                 mask: np.ndarray, image: np.ndarray | None = None, max_shift_fraction: float = 0.25,
                 size: str = "natural", generated_up: np.ndarray = TRIPOSPLAT_UP, render=None,
                 yaw_step_deg: float = 10.0, work_size: int = 256, max_points: int = 20_000,
                 photometric_margin: float = 0.05, iterations: int = 6) -> tuple[ObjectGaussians, dict]:
    """``obj`` placed in the scene: up along the plane normal, yaw from the
    silhouette (and colour), base on the plane, position and uniform scale
    from Qwen's ``mask`` (hero camera pixels, ``image`` the same view's RGB
    in 0..1 or 0..255). ``max_shift_fraction`` caps the move from the old
    footprint's centre, as a fraction of the old footprint's size.

    ``size`` "natural" keeps Qwen's size (the silhouette's area); "match" and
    "larger" set the scale from the removed object (``size_scale``), about
    the base position the natural fit found (where Qwen drew it). The natural fit is reported in
    every mode (``height_natural_vs_removed``)."""
    if size not in SIZE_MODES:
        raise ValueError(f"size must be one of {SIZE_MODES}")
    mask = np.asarray(mask, bool)
    if mask.sum() < 16:
        raise ValueError("the new object's mask is (almost) empty")
    height, width = mask.shape
    viewmat = np.asarray(viewmat, np.float64)
    k = np.asarray(intrinsic, np.float64)
    if image is not None:
        image = np.asarray(image, np.float64)[..., :3]
        if image.max() > 1.0:
            image = image / 255.0
    render = render or point_render
    max_shift = float(max_shift_fraction) * plane.footprint_size
    normal = plane.normal
    base_rotation = rotation_aligning(generated_up, normal)
    opaque = obj.opacity >= 0.3
    if opaque.sum() < 16:
        opaque = np.ones(obj.count, bool)
    origin = np.median(obj.means, axis=0)

    # Yaw: every step on a subsample at ``work_size`` with point silhouettes.
    factor = float(work_size) / max(height, width)
    small_k = k.copy()
    small_k[:2] *= factor
    small_shape = (max(1, round(height * factor)), max(1, round(width * factor)))
    small = ndimage.zoom(mask.astype(np.float32), (small_shape[0] / height, small_shape[1] / width), order=1) >= 0.5
    pick = np.flatnonzero(opaque)
    if len(pick) > max_points:
        pick = pick[np.linspace(0, len(pick) - 1, max_points).round().astype(np.int64)]

    def small_silhouette(frame):
        def silhouette(scale, position):
            return _points_mask(scale * frame.local[pick] + position)
        return silhouette

    def _points_mask(points):
        uv, z = _project(points, viewmat, small_k)
        ok = (z > 0) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < small_shape[1]) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < small_shape[0])
        out = np.zeros(small_shape, bool)
        px = uv[ok].astype(np.int64)
        out[px[:, 1], px[:, 0]] = True
        return ndimage.binary_fill_holes(ndimage.binary_dilation(out, iterations=1)) if out.any() else out

    def placed(frame: _Frame, scale: float, position: np.ndarray) -> ObjectGaussians:
        q = quaternion_from_matrix(frame.rotation)
        return ObjectGaussians(scale * frame.local + position, obj.log_scales + math.log(scale),
                               _quat_multiply(np.repeat(q[None], obj.count, axis=0), obj.quaternions),
                               obj.opacity_logits, obj.rgb)

    def photometric(frame, scale, position) -> float:
        rgb, alpha = render(placed(frame, scale, position), viewmat, k, width, height)
        seen = mask & (alpha > 0.5)
        return 1.0 if seen.sum() < 16 else float(np.abs(rgb[seen] - image[seen]).mean())

    def search(yaw):
        frame = _frame(obj.means, opaque, normal, base_rotation, yaw, origin)
        scale = plane.object_height / max(frame.height, 1e-9)
        scale, position, _capped, sil = _fit(frame, small_silhouette(frame), plane=plane, viewmat=viewmat, k=small_k,
                                             target=small, scale=scale, max_shift=max_shift, iterations=iterations)
        return _iou(sil, small), frame, scale, position

    candidates = {float(y): search(float(y)) for y in np.arange(0.0, 360.0, float(yaw_step_deg))}
    best = max(candidates, key=lambda y: (candidates[y][0], -y))
    error = None
    if image is not None:
        # A silhouette cannot tell the front from the back: the yaws close to
        # the best IoU are rendered, and the one most like Qwen's picture wins.
        short = [y for y, c in candidates.items() if c[0] >= candidates[best][0] - photometric_margin]
        scored = {y: photometric(*candidates[y][1:]) for y in short}
        best = min(scored, key=lambda y: (scored[y], -y))
        error = scored[best]
    # Refined to a degree by the silhouette alone, within a step of the winner.
    coarse = best
    for y in np.arange(coarse - yaw_step_deg + 1.0, coarse + yaw_step_deg, 1.0):
        y = float(y) % 360.0
        if y not in candidates:
            candidates[y] = search(y)
        if candidates[y][0] > candidates[best][0]:
            best = y

    # The final fit on every Gaussian with ``render``'s alpha at full size.
    _iou_small, frame, scale, _position = candidates[best]

    def full_silhouette(scale, position):
        return render(placed(frame, scale, position), viewmat, k, width, height)[1] > 0.5
    scale, position, capped, sil = _fit(frame, full_silhouette, plane=plane, viewmat=viewmat, k=k, target=mask,
                                        scale=scale, max_shift=max_shift, iterations=iterations)
    natural_scale = scale
    sizing = {}
    if size != "natural":
        # Scaled about the base where Qwen's drawing stands (the natural fit's): refitting the centroid at the
        # new size would pull a bigger object towards the camera (its centroid sits higher in the picture).
        scale, sizing = size_scale(frame, opaque, plane, size)
        sil = full_silhouette(scale, position)
    result = placed(frame, scale, position)
    _up, a1, a2 = ground_axes(normal)
    major, minor = footprint_extents(np.stack([frame.local[opaque] @ a1, frame.local[opaque] @ a2], axis=1))
    if image is not None:
        error = photometric(frame, scale, position)
    shift = position - plane.footprint_centre
    report = {
        "yaw_deg": best, "size": size, "scale": float(scale), "object_height": float(scale * frame.height),
        "height_vs_removed": float(scale * frame.height / max(plane.object_height, 1e-9)),
        "natural_scale": float(natural_scale),
        "height_natural_vs_removed": float(natural_scale * frame.height / max(plane.object_height, 1e-9)),
        "scale_vs_natural": float(scale / max(natural_scale, 1e-12)),
        "dominant_vs_removed": float(scale * dominant_extent(frame.height, major, minor)
                                     / max(plane.dominant_extent, 1e-9)),
        **sizing,
        "base_centre": [float(v) for v in position], "shift": float(np.linalg.norm(shift)),
        "shift_fraction": float(np.linalg.norm(shift) / max(plane.footprint_size, 1e-9)),
        "max_shift_fraction": float(max_shift_fraction), "shift_capped": bool(capped),
        "silhouette_iou": _iou(sil, mask), "area_ratio": float(sil.sum() / max(mask.sum(), 1)),
        "centroid_offset_px": [float(v) for v in (_centroid(sil) - _centroid(mask))] if sil.any() else None,
        "photometric_error": error, "yaws_tried": len(candidates),
    }
    return result, report


# ---- the scene round the new object -------------------------------------------------------

def _footprint_grid(uv: np.ndarray, step: float, pad: float):
    lo, hi = uv.min(axis=0) - pad, uv.max(axis=0) + pad
    shape = np.maximum(np.ceil((hi - lo) / step).astype(np.int64) + 1, 1)
    occupied = np.zeros(shape, bool)
    cells = np.clip(((uv - lo) / step).astype(np.int64), 0, shape - 1)
    occupied[cells[:, 0], cells[:, 1]] = True
    return lo, occupied


def _cells(uv: np.ndarray, lo: np.ndarray, step: float, shape) -> tuple[np.ndarray, np.ndarray]:
    cells = np.floor((uv - lo) / step).astype(np.int64)
    inside = (cells >= 0).all(axis=1) & (cells[:, 0] < shape[0]) & (cells[:, 1] < shape[1])
    return np.clip(cells, 0, np.asarray(shape) - 1), inside


def object_footprint(obj: ObjectGaussians, plane: SupportPlane, *, cells: int = 80):
    """The placed object's footprint on the plane: a grid over the plane
    axes (``cells`` across the object's width), closed and filled."""
    _up, a1, a2 = ground_axes(plane.normal)
    opaque = obj.opacity >= 0.3
    if opaque.sum() < 16:
        opaque = np.ones(obj.count, bool)
    uv = np.stack([obj.means[opaque] @ a1, obj.means[opaque] @ a2], axis=1)
    width = float(np.max(np.ptp(uv, axis=0)))
    step = max(width / cells, 1e-9)
    return uv, width, step, (a1, a2), opaque


def object_rows(source_means: np.ndarray, object_means: np.ndarray) -> np.ndarray:
    """The source rows an extracted object PLY holds (Extract writes source
    rows unchanged: matched by position, bit for bit, like ``fill_rows``)."""
    return np.flatnonzero(~fill_rows(source_means, object_means))


@dataclass(frozen=True)
class PlaneGrid:
    """Square cells on the support plane round the removed object's footprint."""
    lo: np.ndarray
    step: float
    shape: tuple
    a1: np.ndarray
    a2: np.ndarray

    def uv(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64)
        return np.stack([points @ self.a1, points @ self.a2], axis=1)

    def cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _cells(self.uv(points), self.lo, self.step, self.shape)

    def raster(self, points: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape, bool)
        cells, inside = self.cells(points)
        out[cells[inside, 0], cells[inside, 1]] = True
        return out

    def lookup(self, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        cells, inside = self.cells(points)
        return inside & grid[cells[:, 0], cells[:, 1]]


def _closed_footprint(grid: PlaneGrid, points: np.ndarray, plane: SupportPlane, low_fraction: float) -> np.ndarray:
    """Cells under the lower ``low_fraction`` of ``points`` (by height above
    the plane), closed and filled: where the object meets the surface."""
    h = plane.height(points)
    top = float(np.percentile(h, 99.0))
    low = points[h < low_fraction * top]
    if len(low) < 3:
        low = points
    # Each point stamps its neighbours first (sparse points leave gaps between cells), taken back after.
    grown = ndimage.binary_dilation(grid.raster(low), iterations=1)
    return ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_closing(grown, iterations=2)), iterations=1)


def replace_fill(source_means: np.ndarray, source_opacity_logits: np.ndarray, source_rgb: np.ndarray,
                 object_ids: np.ndarray, plane: SupportPlane, *, new_object: ObjectGaussians | None = None,
                 remove_fill: ObjectGaussians | None = None, cells: int = 100, low_fraction: float = 0.35,
                 grow_fraction: float = 0.03, band_fraction: float = 0.05, remove_band_fraction: float = 0.15,
                 lift_fraction: float = 0.002, ring_gap_fraction: float = 0.06, ring_fraction: float = 0.3,
                 surface_fraction: float = 0.01, blur_cells: float = 1.5, opacity: float = 0.98,
                 volume_margin_fraction: float = 0.03, volume_top_fraction: float = 1.05,
                 depth_fraction: float = 0.1, contact_fraction: float = 0.1, colour_neighbors: int = 32,
                 colour_best: int = 4, support_colour_spread: float = 4.0,
                 ) -> tuple[np.ndarray, ObjectGaussians, dict]:
    """Replace's base: the source scene without exactly the extracted object
    (``object_ids``), and a flat fill where the old object met the surface.

    Nobody saw the surface under the old object's contact footprint (its
    lower ``low_fraction``, closed, grown by ``grow_fraction`` of the
    footprint's size: kitchen, the mat under the bulldozer's tracks). Where
    ``new_object`` (placed) touches it (its own lowest ``contact_fraction``,
    closed, one cell in: under a vase's belly the ground shows from a low
    camera) it stays hidden; the rest gets a fill on the plane
    (``lift_fraction`` of the old height above it):

    - ``remove_fill``: the project's Remove fill Gaussians (Qwen's repaint
      of the hero view as Gaussians), those in the region up to
      ``remove_band_fraction`` of the old height above the plane, laid flat
      on it (the Remove sheet stands 2–6 % proud and has rims);
    - every cell they leave empty: one flat disc, coloured from the surface
      in a ring round the footprint (from ``ring_gap_fraction`` to
      ``ring_fraction`` of its size: right next to it the mat lies in the old
      object's shadow), only Gaussians on the surface (within
      ``surface_fraction`` of the old height of the plane, not the darker
      ones under it), each cell taking the nearest ring cell's colour,
      blurred by ``blur_cells``.

    The removal set is wider than the extract, because a new object covers
    the spot (kitchen, 27 Oct 2026: the bulldozer's leftovers showed as dark
    and yellow streaks in front of the vase and dark splats at its base).
    The old object's volume is its whole footprint (every height, closed,
    grown by ``volume_margin_fraction`` of its size) from
    ``depth_fraction`` of its height below the plane up to
    ``volume_top_fraction`` of its height above it. In it:

    - everything above the band (``band_fraction`` of the old height) goes,
      whatever its colour: parts of the old object the selection never took
      (elongated, opaque splats inside it) and floaters inside it;
    - in the band (and below the plane), rows inside the filled region go
      (the extract leaves the mat at the contact line in the tracks' shadow,
      as dark as the tracks, and the trim's bottoms of the tracks); in the
      rest of the footprint (under an overhang, under the new object), those
      whose colour is not the support's: nearer the old object's colours
      round them than the ring's, or further from the ring's than
      ``support_colour_spread`` times the ring's own scatter (as
      support_trim; ``colour_neighbors``/``colour_best``).

    Everything else, the mat round the footprint included, is the source
    scene as it was.

    Returns the source rows to drop (the object's and those), the fill
    Gaussians and a report."""
    source_means = np.asarray(source_means, np.float64)
    ids = np.unique(np.asarray(object_ids, np.int64))
    if len(ids) == 0:
        raise ValueError("the removed object has no Gaussians")
    count = len(source_means)
    is_object = np.zeros(count, bool)
    is_object[ids] = True
    height_old = plane.object_height
    size = max(plane.footprint_size, 1e-9)
    step = size / float(cells)
    _up, a1, a2 = ground_axes(plane.normal)
    uv_object = np.stack([source_means[ids] @ a1, source_means[ids] @ a2], axis=1)
    pad = ring_fraction * size + 4 * step
    lo = uv_object.min(axis=0) - pad
    shape = tuple(int(v) for v in np.ceil((uv_object.max(axis=0) + pad - lo) / step).astype(np.int64) + 1)
    grid = PlaneGrid(lo, step, shape, a1, a2)

    region = _closed_footprint(grid, source_means[ids], plane, low_fraction)
    grow = int(round(grow_fraction * size / step))
    if grow > 0:
        region = ndimage.binary_dilation(region, iterations=grow)
    covered = np.zeros(shape, bool)
    if new_object is not None:
        opaque = new_object.opacity >= 0.3
        points = new_object.means[opaque] if opaque.sum() >= 16 else new_object.means
        # Where it touches the plane, one cell in from its edge: the fill reaches under its rim and belly.
        covered = ndimage.binary_erosion(_closed_footprint(grid, points, plane, contact_fraction), iterations=1)
    hole = region & ~covered

    h = plane.height(source_means)
    band = band_fraction * height_old
    depth = depth_fraction * height_old
    opacity_source = 1.0 / (1.0 + np.exp(-np.asarray(source_opacity_logits, np.float64).reshape(-1)))
    rgb_source = np.asarray(source_rgb, np.float64)
    near_plane = (h >= -depth) & (h <= band) & ~is_object
    drop = is_object | (near_plane & grid.lookup(hole, source_means))
    # The old object's volume: its whole footprint (every height), closed, grown by the margin.
    volume = ndimage.binary_closing(ndimage.binary_dilation(grid.raster(source_means[ids]), iterations=1), iterations=2)
    volume = ndimage.binary_fill_holes(volume)
    margin = int(round(volume_margin_fraction * size / step))
    if margin > 0:
        volume = ndimage.binary_dilation(volume, iterations=margin)
    in_volume = grid.lookup(volume, source_means) & ~is_object
    above = in_volume & (h > band) & (h <= volume_top_fraction * height_old)
    drop |= above
    # In the band, outside the filled region: what is not the support's colour goes.
    on_surface = np.abs(h) <= max(surface_fraction * height_old, 3 * plane.spread)
    gap = ndimage.distance_transform_edt(~region) * step
    ring = (gap > ring_gap_fraction * size) & (gap <= ring_fraction * size)
    # The support's colours: the surface in the same ring round the whole volume (not under an overhang).
    gap_volume = ndimage.distance_transform_edt(~volume) * step
    palette = (gap_volume > ring_gap_fraction * size) & (gap_volume <= ring_fraction * size)
    ring_rows = np.flatnonzero(on_surface & ~is_object & (opacity_source >= 0.3) & grid.lookup(palette, source_means))
    contact = np.flatnonzero(in_volume & near_plane & ~drop)
    trusted = ids[(h[ids] > band) & (opacity_source[ids] >= 0.3)]
    leftover = np.zeros(0, np.int64)
    if len(contact) and len(ring_rows) >= colour_best + 1 and len(trusted) >= colour_best:
        uv_source = grid.uv(source_means)
        to_support = _local_colour_distance(rgb_source[contact], uv_source[contact], rgb_source[ring_rows],
                                            uv_source[ring_rows], neighbors=colour_neighbors, best=colour_best)
        to_object = _local_colour_distance(rgb_source[contact], source_means[contact], rgb_source[trusted],
                                           source_means[trusted], neighbors=colour_neighbors, best=colour_best)
        sample = ring_rows
        if len(sample) > 20000:
            sample = np.random.default_rng(0).choice(sample, 20000, replace=False)
        # The ring's own scatter (each sample's nearest ring colours include itself: one more, rescaled).
        scatter = _local_colour_distance(rgb_source[sample], uv_source[sample], rgb_source[ring_rows],
                                         uv_source[ring_rows], neighbors=colour_neighbors,
                                         best=colour_best + 1) * (colour_best + 1) / max(colour_best, 1)
        tolerance = support_colour_spread * float(np.percentile(scatter, 90.0))
        leftover = contact[(to_object <= to_support) | (to_support > tolerance)]
        drop[leftover] = True

    lift = lift_fraction * height_old
    parts, used_remove = [], 0
    filled = np.zeros(shape, bool)
    if remove_fill is not None and remove_fill.count:
        rh = plane.height(remove_fill.means)
        take = (rh >= -band) & (rh <= remove_band_fraction * height_old) & grid.lookup(hole, remove_fill.means)
        if take.any():
            flat = remove_fill.subset(take)
            means = flat.means - (plane.height(flat.means) - lift)[:, None] * plane.normal
            flat = ObjectGaussians(means, flat.log_scales, flat.quaternions, flat.opacity_logits, flat.rgb)
            parts.append(flat)
            used_remove = flat.count
            # A cell next to a laid-flat Gaussian counts as covered (they are wider than a cell).
            filled = ndimage.binary_dilation(grid.raster(flat.means[flat.opacity >= 0.3]), iterations=1) & hole

    # The surface round the footprint, per cell: the nearest observed colour, blurred.
    seen_rows = np.flatnonzero(on_surface & ~drop & (opacity_source >= 0.3))
    cells_seen, inside = grid.cells(source_means[seen_rows])
    seen_rows, cells_seen = seen_rows[inside], cells_seen[inside]
    keep = ring[cells_seen[:, 0], cells_seen[:, 1]]
    seen_rows, cells_seen = seen_rows[keep], cells_seen[keep]
    total = np.zeros((*shape, 3))
    weight = np.zeros(shape)
    np.add.at(total, (cells_seen[:, 0], cells_seen[:, 1]), rgb_source[seen_rows])
    np.add.at(weight, (cells_seen[:, 0], cells_seen[:, 1]), 1.0)
    seen = weight > 0
    need = hole & ~filled
    discs = 0
    if need.any():
        if not seen.any():
            raise ValueError("no surface round the removed object's footprint to colour the fill from")
        colour = total / np.maximum(weight, 1)[..., None]
        _d, nearest = ndimage.distance_transform_edt(~seen, return_indices=True)
        colour = colour[nearest[0], nearest[1]]
        if blur_cells > 0:
            colour = ndimage.gaussian_filter(colour, (blur_cells, blur_cells, 0))
        iu, iv = np.nonzero(need)
        cell_uv = np.stack([lo[0] + (iu + 0.5) * step, lo[1] + (iv + 0.5) * step], axis=1)
        # a1, a2 span the plane's directions: the offset along the normal puts the points on it.
        points = cell_uv[:, :1] * a1 + cell_uv[:, 1:] * a2 + (plane.offset + lift) * plane.normal
        discs = len(points)
        q = quaternion_from_matrix(rotation_aligning([0.0, 0.0, 1.0], plane.normal))
        alpha = float(np.clip(opacity, 1e-3, 0.999))
        parts.append(ObjectGaussians(points, np.log(np.tile([0.75 * step, 0.75 * step, 0.05 * step], (discs, 1))),
                                     np.tile(q, (discs, 1)), np.full(discs, math.log(alpha / (1 - alpha))),
                                     np.clip(colour[iu, iv], 0.0, 1.0)))
    fill = ObjectGaussians.concat(*parts) if parts else ObjectGaussians(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3)))
    cell_area = step * step
    return drop, fill, {
        "object_gaussians": int(len(ids)), "contact_shadow_rows_dropped": int((drop & ~is_object).sum()),
        "volume_rows_dropped": int(above.sum()), "leftover_contact_rows_dropped": int(len(leftover)),
        "hole_rows_dropped": int((near_plane & grid.lookup(hole, source_means)).sum()),
        "footprint_area": float(region.sum() * cell_area), "covered_by_new_object": float((region & covered).sum() * cell_area),
        "filled_area": float(hole.sum() * cell_area),
        "filled_fraction": float(hole.sum() / max(region.sum(), 1)),
        "from_remove_fill": int(used_remove), "remove_fill_area": float(filled.sum() * cell_area),
        "plane_fill_discs": int(discs), "cell": float(step), "band": float(band),
        "source": "remove fill + plane" if used_remove and discs else "remove fill" if used_remove else "plane",
    }


def contact_shadow(obj: ObjectGaussians, plane: SupportPlane, *, surface: np.ndarray | None = None,
                   surface_band: float = 0.0, strength: float = 0.3, low_fraction: float = 0.12,
                   pad_fraction: float = 0.25, blur_fraction: float = 0.05, lift_fraction: float = 0.01,
                   cells: int = 80, minimum_alpha: float = 0.03) -> ObjectGaussians:
    """Flat dark Gaussians on the support: the footprint of the object's lower
    ``low_fraction`` (by height), closed, blurred by ``blur_fraction`` of its
    width and spread ``pad_fraction`` of its width beyond it, at most
    ``strength`` opaque, ``lift_fraction`` of the object's height above the
    surface. ``surface`` (scene points) drapes the shadow over what is
    actually there, up to ``surface_band`` above the plane (a fill sheet the
    cleanup left); without it, it lies on the plane."""
    uv_all, width, step, (a1, a2), opaque = object_footprint(obj, plane, cells=cells)
    h = plane.height(obj.means)
    top = float(np.percentile(h[opaque], 99.0))
    low = opaque & (h < low_fraction * top)
    if low.sum() < 3:
        low = opaque
    uv = np.stack([obj.means[low] @ a1, obj.means[low] @ a2], axis=1)
    pad = pad_fraction * width
    lo, occupied = _footprint_grid(uv, step, pad)
    occupied = ndimage.binary_fill_holes(ndimage.binary_closing(occupied, iterations=3)).astype(np.float64)
    dark = float(strength) * np.clip(ndimage.gaussian_filter(occupied, blur_fraction * width / step) * 1.6, 0.0, 1.0)
    keep = dark > minimum_alpha
    iu, iv = np.nonzero(keep)
    alpha = np.clip(dark[keep], 1e-4, 0.999)
    # Kitchen vase (27 Oct 2026): at 0.75 (the spike's) and 0.5 the shadow read as a dark halo next to
    # Qwen's own soft one; 0.35 is about as dark as Qwen draws it right at the base.
    cell_uv = np.stack([lo[0] + (iu + 0.5) * step, lo[1] + (iv + 0.5) * step], axis=1)
    ground = np.zeros(len(cell_uv))
    if surface is not None and len(surface) and surface_band > 0:
        surface = np.asarray(surface, np.float64)
        sh = plane.height(surface)
        near = (sh >= -surface_band) & (sh <= surface_band)
        if near.any():
            heights = sh[near]
            distance, index = cKDTree(np.stack([surface[near] @ a1, surface[near] @ a2], axis=1)).query(
                cell_uv, k=8, distance_upper_bound=1.5 * step)
            found = np.isfinite(distance)
            top_here = np.where(found, heights[np.minimum(index, len(heights) - 1)], -np.inf).max(axis=1)
            ground = np.clip(np.where(np.isfinite(top_here), top_here, 0.0), 0.0, surface_band)
    # a1, a2 span the plane's directions: the offset along the normal puts the points on it.
    points = (cell_uv[:, :1] * a1 + cell_uv[:, 1:] * a2
              + (plane.offset + ground + lift_fraction * top)[:, None] * plane.normal)
    size = 0.8 * step
    q = quaternion_from_matrix(rotation_aligning([0.0, 0.0, 1.0], plane.normal))
    count = len(points)
    return ObjectGaussians(points, np.log(np.tile([size, size, 0.05 * size], (count, 1))), np.tile(q, (count, 1)),
                           np.log(alpha / (1 - alpha)), np.zeros((count, 3)))


def write_scene(scene: GaussianPly, keep: np.ndarray, added: ObjectGaussians, output: Path, message: str,
                moved: tuple[np.ndarray, np.ndarray] | None = None) -> int:
    """The scene's kept rows (``moved``: rows given new means) and ``added``
    (SH degree 0; higher bands zero) in the scene's own layout."""
    vertex = scene.vertex
    if moved is not None and len(moved[0]):
        vertex = vertex.copy()
        for i, c in enumerate("xyz"):
            vertex[c][moved[0]] = moved[1][:, i]
    born = np.zeros(added.count, dtype=scene.vertex.dtype)
    for i, c in enumerate("xyz"):
        born[c] = added.means[:, i]
    for i in range(3):
        born[f"scale_{i}"] = added.log_scales[:, i]
        born[f"f_dc_{i}"] = (added.rgb[:, i] - 0.5) / SH_C0
    for i in range(4):
        born[f"rot_{i}"] = added.quaternions[:, i]
    born["opacity"] = added.opacity_logits
    rows = np.concatenate([vertex[np.asarray(keep, bool)], born])
    write_vertices(scene, output, rows, message)
    return len(rows)
