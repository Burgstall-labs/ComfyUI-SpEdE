"""All-around fill: the removed object's hole patched so it holds from every direction.

The depth fill (``depth_fill``) is right in one camera, the hero; seen from
the other cameras its error was three times the hero's (0.10 against 0.034 on
kitchen). This module makes the fill agree with every camera:

1. plan: cameras that see the hole well from clearly different directions
   (the Qwen chain), and an orbit around the object through real cameras;
2. the Qwen chain cleans those cameras' photos (a ComfyUI node; the hero's
   high-resolution clean image is the first reference);
3. init: every chain camera adds depth births (its own MoGe depth, aligned to
   the scene) where the earlier cameras' fill disagrees with its clean image;
4. orbit warp: the hero's clean image reprojected with its aligned depth into
   every orbit frame, magenta where the hero saw nothing (CrossView-Warp
   conditioning for MiniMax H3, which draws the frames consistently);
5. frames that slipped off their path are dropped (``frame_usable``);
6. optimise the hole's Gaussians against all cameras at once: chain images and
   video frames inside the hole, every real photo outside it, the splats
   anchored near where they started (free, they drifted into the air where no
   camera looks and showed as fuzz from low side views).

Verified on kitchen (bulldozer, 10 chain cameras, 260-frame 1024 px orbit):
straight mat edges from all eight held-out cameras that see the hole.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from .geometry_utils import project_to_pixels

_LOG = logging.getLogger(__name__)

MAGENTA = np.array([255, 0, 255], np.uint8)


# ---- plan ---------------------------------------------------------------------

def camera_centre(view) -> np.ndarray:
    return np.linalg.inv(np.asarray(view.viewmat, np.float64))[:3, 3]


def choose_chain(rows: list[dict], cameras, hero: str, centre: np.ndarray, *, count: int = 10,
                 allowed: set[str] | None = None, min_visible: float = 0.8, min_in_frame: float = 0.95) -> dict:
    """The hero, then the best-seeing cameras that look at the object from
    directions at least 30 degrees from every camera chosen so far (20, then
    12, when that does not give ``count``)."""
    def direction(v):
        x = np.asarray(centre, np.float64) - camera_centre(cameras.by_id[v])
        return x / np.linalg.norm(x)

    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip(direction(a) @ direction(b), -1, 1))))

    chain = [hero]
    usable = [r for r in rows if r["view_id"] != hero and r["visible"] >= min_visible
              and r["in_frame"] >= min_in_frame and (allowed is None or r["view_id"] in allowed)]
    for min_angle in (30, 20, 12):
        for row in usable:
            v = row["view_id"]
            if v in chain:
                continue
            if all(angle(v, c) >= min_angle for c in chain):
                chain.append(v)
            if len(chain) == count:
                break
        if len(chain) == count:
            break
    return {"chain": chain, "angle_to_hero": {v: round(angle(v, hero), 1) for v in chain}}


def orbit_axes(cameras) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cameras' mean up and two horizontal axes."""
    up = -np.mean([np.linalg.inv(np.asarray(v.viewmat, np.float64))[:3, 1] for v in cameras.views if not v.is_virtual],
                  axis=0)
    up /= np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return up, e1, np.cross(up, e1)


def cylindrical(position: np.ndarray, centre: np.ndarray, axes) -> np.ndarray:
    """Azimuth (degrees), horizontal radius and height about the object."""
    up, e1, e2 = axes
    d = np.asarray(position, np.float64) - centre
    h = d @ up
    return np.array([np.degrees(np.arctan2(d @ e2, d @ e1)), np.linalg.norm(d - h * up), h])


def choose_orbit_stops(rows: list[dict], cameras, hero: str, centre: np.ndarray, *, spacing: float = 36.0,
                       min_visible: float = 0.5, min_in_frame: float = 0.95, direction: float = 1.0) -> list[str]:
    """Real cameras around the object, about ``spacing`` degrees of azimuth
    apart, from the hero round to the hero. The orbit's frames are posed
    between them, so every frame's camera is near a real one (H3 follows the
    warp; a frame far from every camera has little of the hero to warp)."""
    axes = orbit_axes(cameras)
    hero_az = cylindrical(camera_centre(cameras.by_id[hero]), centre, axes)[0]
    advance = {}
    for row in rows:
        v = row["view_id"]
        if v == hero or row["visible"] < min_visible or row["in_frame"] < min_in_frame:
            continue
        az = cylindrical(camera_centre(cameras.by_id[v]), centre, axes)[0]
        advance[v] = ((az - hero_az) * direction) % 360.0
    score = {r["view_id"]: r["score"] for r in rows}
    stops, at = [], 0.0
    while True:
        ahead = {v: a - at for v, a in advance.items() if a - at > 0.5 * spacing and v not in stops}
        if not ahead or 360.0 - at < 1.5 * spacing:
            break
        near = {v: d for v, d in ahead.items() if d <= 1.6 * spacing}
        if near:
            # Close to one spacing; between equally close ones the better view.
            v = min(near, key=lambda k: (round(abs(near[k] - spacing) / (0.25 * spacing)), -score.get(k, 0.0)))
        else:
            v = min(ahead, key=ahead.get)      # a gap: the next camera after it
        if 360.0 - advance[v] < 0.5 * spacing:
            break
        stops.append(v)
        at = advance[v]
    return stops + [hero]


def orbit_path(cameras, hero: str, stops: list[str], centre: np.ndarray, frames: int, *,
               direction: float = 1.0) -> dict:
    """World-to-camera matrices for ``frames`` frames from the hero through
    the stops: azimuth, radius and height interpolated about the object (an
    arc, not a chord through it), rotation slerped, frames spread by angle."""
    from scipy.spatial.transform import Rotation, Slerp

    ids = [hero] + list(stops)
    poses = [np.linalg.inv(np.asarray(cameras.by_id[v].viewmat, np.float64)) for v in ids]
    axes = orbit_axes(cameras)
    up, e1, e2 = axes
    cyl = np.array([cylindrical(p[:3, 3], centre, axes) for p in poses])
    az = [cyl[0, 0]]
    for k in range(1, len(cyl)):
        delta = (cyl[k, 0] - az[-1]) * direction % 360.0
        az.append(az[-1] + direction * (delta if delta > 1e-6 else 360.0))
    cyl[:, 0] = az
    span = np.abs(np.diff(cyl[:, 0]))
    keys = np.concatenate([[0], np.cumsum(span)]) / span.sum() * (frames - 1)
    rots = Rotation.from_matrix(np.stack([p[:3, :3] for p in poses]))
    legs = len(poses) - 1
    viewmats = []
    for f in range(frames):
        i = min(int(np.searchsorted(keys, f, side="right")) - 1, legs - 1)
        t = (f - keys[i]) / (keys[i + 1] - keys[i])
        c2w = np.eye(4)
        c2w[:3, :3] = Slerp([0, 1], rots[[i, i + 1]])([t]).as_matrix()[0]
        a, r, h = (1 - t) * cyl[i] + t * cyl[i + 1]
        a = np.radians(a)
        c2w[:3, 3] = centre + r * (np.cos(a) * e1 + np.sin(a) * e2) + h * up
        viewmats.append(np.linalg.inv(c2w))
    return {"stops": ids, "keys": keys.tolist(), "viewmats": [m.tolist() for m in viewmats]}


# ---- warp and frame check -----------------------------------------------------

def hero_depth(mask: np.ndarray, source_depth: np.ndarray, source_alpha: np.ndarray, aligned: np.ndarray,
               scale: int = 1) -> np.ndarray:
    """The hero's depth for the warp: the scene's own where it is solid, the
    aligned MoGe depth in the hole; upsampled for a ``scale`` x warp."""
    solid = (~mask) & (source_alpha > 0.9) & np.isfinite(source_depth) & (source_depth > 0)
    z = np.where(solid, source_depth, aligned).astype(np.float64)
    if scale > 1:
        good = np.isfinite(z) & (z > 0)
        _d, (iy, ix) = ndimage.distance_transform_edt(~good, return_indices=True)
        z = ndimage.zoom(z[iy, ix], scale, order=1)
        z[~ndimage.zoom(good.astype(np.uint8), scale, order=0).astype(bool)] = np.nan
    return z


class HeroWarp:
    """The hero's clean image as coloured 3D points, splatted into other cameras."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, viewmat: np.ndarray):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.asarray(rgb, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"hero image {rgb.shape[:2]} and depth {depth.shape} differ")
        K = np.asarray(intrinsic, np.float64)
        ok = np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(ok)
        z = depth[v, u]
        cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        c2w = np.linalg.inv(np.asarray(viewmat, np.float64))
        self.world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        self.colours = rgb[v, u, :3]
        self.size = rgb.shape[:2]

    def render(self, viewmat: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        H, W = self.size
        K = np.asarray(intrinsic, np.float64)
        vm = np.asarray(viewmat, np.float64)
        p = self.world @ vm[:3, :3].T + vm[:3, 3]
        good = p[:, 2] > 1e-6
        x = np.floor(p[good, 0] / p[good, 2] * K[0, 0] + K[0, 2]).astype(np.int64)
        y = np.floor(p[good, 1] / p[good, 2] * K[1, 1] + K[1, 2]).astype(np.int64)
        c, d = self.colours[good], p[good, 2]
        order = np.argsort(-d)                    # far first, near overwrites
        x, y, c = x[order], y[order], c[order]
        img = np.tile(MAGENTA, (H, W, 1))
        # 2x2 splats close the stretch cracks; the point's own pixel is
        # written last (the other way round shifted the warp by a pixel).
        for dy in (1, 0):
            for dx in (1, 0):
                xx, yy = x + dx, y + dy
                inb = (xx >= 0) & (yy >= 0) & (xx < W) & (yy < H)
                img[yy[inb], xx[inb]] = c[inb]
        return img


def frame_check(warp: np.ndarray, frame: np.ndarray, *, warp_scale: int = 1, max_mae: float = 0.08) -> dict:
    """Does a video frame stay on its path camera? Colour error against its
    warp where the warp has pixels, and the best 2D shift (a drifting camera
    shows as a shift); usable when both are small."""
    w = np.asarray(warp, np.float32) / (255.0 if np.asarray(warp).dtype == np.uint8 else 1.0)
    v = np.asarray(frame, np.float32) / (255.0 if np.asarray(frame).dtype == np.uint8 else 1.0)
    if v.shape != w.shape:
        from PIL import Image
        v = np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize(w.shape[1::-1]), np.float32) / 255
    known = ~((w[..., 0] > 0.95) & (w[..., 1] < 0.05) & (w[..., 2] > 0.95))
    known = ndimage.binary_erosion(known, iterations=3)
    if known.sum() < 1000:
        return {"known": int(known.sum()), "usable": False}
    wg, vg = ndimage.gaussian_filter(w.mean(2), 1.5), ndimage.gaussian_filter(v.mean(2), 1.5)
    best = (np.inf, 0, 0)
    for dy in range(-6, 7, 2):
        for dx in range(-6, 7, 2):
            e = float(np.abs(np.roll(vg, (dy, dx), axis=(0, 1)) - wg)[known].mean())
            best = min(best, (e, dy, dx))
    mae = float(np.abs(v - w).mean(2)[known].mean())
    shift = float(np.hypot(best[1], best[2]))
    return {"mae": round(mae, 4), "shift_px": shift, "known": int(known.sum()),
            "usable": bool(mae < max_mae and shift <= 2 * warp_scale)}


def object_hole(object_means: np.ndarray, viewmat, intrinsic, size: int, dilate: int = 12) -> np.ndarray:
    """The removed object's projection, closed, filled and grown: where a
    video frame says something about the fill."""
    uv, z = project_to_pixels(object_means, np.asarray(viewmat, np.float32), np.asarray(intrinsic, np.float32))
    hole = np.zeros((size, size), bool)
    ok = np.isfinite(uv).all(1) & (z > 0)
    xx, yy = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
    inb = (xx >= 0) & (yy >= 0) & (xx < size) & (yy < size)
    hole[yy[inb], xx[inb]] = True
    hole = ndimage.binary_fill_holes(ndimage.binary_closing(hole, iterations=4))
    return ndimage.binary_dilation(hole, iterations=int(dilate))


# ---- init from the chain --------------------------------------------------------

def _unit(a) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / 255 if a.max() > 1.5 else a


def multiview_init(*, source_ply: Path, hero_patch_ply: Path, cameras, chain: list[str], masks: dict,
                   edits: dict, object_ids: np.ndarray, moge_depth: Callable, out_dir: Path,
                   device: str = "cuda", log: Callable = _LOG.info, keep_covered: float | None = None,
                   carve: bool = True) -> dict:
    """Start from the hero's depth patch; for every other chain camera: its
    MoGe depth aligned to the current scene, carve the residue and the
    shadow halo that camera sees, and add births only where the current
    render disagrees with its clean image and no earlier camera's births draw.

    ``masks[v]``: the selection in camera v (already grown for non-hero
    cameras); ``edits[v]``: its clean image, float RGB at the camera size;
    ``moge_depth(rgb, camera) -> (depth, valid or None)``. ``keep_covered``:
    no births where the current render's alpha is above it (what is already
    covered is left to the optimisation to recolour). ``carve=False``: keep
    every Gaussian (the residue was already carved; carving against a noisy
    depth cut holes into the real surface)."""
    import torch

    from .depth_fill import (align_depth_to_scene, carve_residue_ids, carving_exposed, create_depth_patch,
                             max_observed_weight, shadow_halo)
    from .geometry_utils import plane_depth_map
    from .patch import fit_view_support_plane
    from .ply import GaussianPly, remove_rows
    from .renderer import GsplatRenderer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def rgbd(ply, v):
        r = GsplatRenderer(source_ply=ply, cameras=cameras, device=device)
        rgb, a, z = r.render_rgbd(v, edited=False)
        del r
        torch.cuda.empty_cache()
        return _unit(rgb), np.asarray(a), np.asarray(z)

    source = GaussianPly.load(source_ply)
    obj = source.means[object_ids]
    index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(source.means, np.float32))}

    def rows_to_source(ply):
        m = np.ascontiguousarray(GaussianPly.load(ply).means, np.float32)
        return np.array([index.get(r.tobytes(), -1) for r in m], np.int64)

    renderer = GsplatRenderer(source_ply=source_ply, cameras=cameras, device=device)
    seen_src = max_observed_weight(renderer, [v.view_id for v in cameras.views if not v.is_virtual], len(source.means))
    del renderer, source
    torch.cuda.empty_cache()

    current = Path(hero_patch_ply)
    steps = []
    for k, v in enumerate(chain[1:], start=1):
        cam = cameras.by_id[v]
        mask, edit = masks[v], edits[v]
        step = {"view": v}
        rgb0, alpha0, z0 = rgbd(current, v)
        pred, pvalid = moge_depth(edit, cam)
        scene_now = GaussianPly.load(current)
        rows = rows_to_source(current)
        births = rows < 0
        plane = None
        try:
            pl, _tol = fit_view_support_plane(scene_now, cam, mask)
            plane = plane_depth_map(cam.viewmat, cam.intrinsic, cam.width, cam.height, pl.normal, pl.offset)
        except (RuntimeError, ValueError) as e:
            step["plane"] = str(e)
        try:
            try:
                aligned = align_depth_to_scene(pred, z0, alpha0, mask, predicted_valid=pvalid, support_plane_depth=plane)
            except RuntimeError as e:
                if plane is None or "disagrees with known planar" not in str(e):
                    raise
                aligned = align_depth_to_scene(pred, z0, alpha0, mask, predicted_valid=pvalid)
        except RuntimeError as e:
            step["error"] = str(e)
            steps.append(step)
            log(f"  {v}: depth alignment refused: {e}")
            continue
        step["align"] = {x: aligned.report[x] for x in ("depth_model", "inlier_fraction", "median_relative_error")}
        halo = shadow_halo(mask, rgb0, edit)
        seen = np.where(births, 1.0, seen_src[np.maximum(rows, 0)]).astype(np.float32)
        if carve:
            ids, _crep = carve_residue_ids(scene_now, obj, cam, mask, aligned.depth, observed_weight=seen,
                                           halo=halo, halo_depth=z0)
        else:
            ids = np.zeros(0, np.int64)
        ids = ids[~births[ids]]                   # earlier cameras' births stay
        step["carved"], step["halo_px"] = int(len(ids)), int(halo.sum())
        carved = out_dir / f"{k}_{v}_carved.ply"
        if len(ids):
            remove_rows(current, ids, carved)
        else:
            shutil.copyfile(current, carved)
        rgb1, alpha1, z1 = rgbd(carved, v)
        rows_c = rows_to_source(carved)
        forbidden = None
        if (rows_c < 0).any():
            only = out_dir / f"{k}_{v}_births_only.ply"
            remove_rows(carved, np.flatnonzero(rows_c >= 0), only)
            _x, ab, _y = rgbd(only, v)
            forbidden = ab > 0.3
            os.remove(only)
        if keep_covered is not None:
            covered = alpha1 > keep_covered
            forbidden = covered if forbidden is None else forbidden | covered
        exposed = carving_exposed(alpha0, z0, alpha1, z1)
        nxt = out_dir / f"{k}_{v}_patched.ply"
        try:
            rep = create_depth_patch(carved, cam, edit, aligned, z1, alpha1, nxt, pixel_stride=1, max_births=200000,
                                     exposed=exposed, remaining_rgb=rgb1, forbidden=forbidden,
                                     halo=halo, halo_depth=z0)
            step["born"] = rep["born_gaussians"]
            current = nxt
        except RuntimeError as e:
            step["born"], step["patch"] = 0, str(e)
            current = carved
        for stale in (out_dir / f"{k - 1}_{chain[k - 1]}_carved.ply", out_dir / f"{k - 1}_{chain[k - 1]}_patched.ply"):
            if k > 1 and stale.exists() and stale != current:
                os.remove(stale)
        if current == nxt and carved.exists():
            os.remove(carved)
        steps.append(step)
        log(f"  {v}: " + json.dumps(step))
    final = out_dir / "mv_init.ply"
    shutil.copyfile(current, final)
    return {"ply": str(final), "steps": steps}


# ---- optimisation ---------------------------------------------------------------

def _ssim_map(a, b):
    import torch.nn.functional as F
    import torch
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    k = torch.ones(3, 1, 7, 7, device=a.device) / 49
    mu_a, mu_b = F.conv2d(a, k, padding=3, groups=3), F.conv2d(b, k, padding=3, groups=3)
    va = F.conv2d(a * a, k, padding=3, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, k, padding=3, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, k, padding=3, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + 1e-4) * (2 * cov + 9e-4)) / ((mu_a ** 2 + mu_b ** 2 + 1e-4) * (va + vb + 9e-4))
    return s[0].mean(0)


def held_out_chain(chain: list[str]) -> set[str]:
    """Two chain cameras (one for a short chain), never the hero, kept out of
    training and only evaluated."""
    if len(chain) >= 8:
        return {chain[len(chain) // 3], chain[2 * len(chain) // 3]}
    return {chain[len(chain) // 2]} if len(chain) >= 3 else set()


def _row_scaled_step(opt, P: dict, scales: dict) -> None:
    """``opt.step()``, then each parameter named in ``scales`` keeps only that
    per-row fraction of its step. Adam's moments do not depend on the learning
    rate, so this is the step at ``lr * scale`` row by row (0: frozen)."""
    import torch
    before = {k: P[k].detach().clone() for k in scales}
    opt.step()
    with torch.no_grad():
        for k, b in before.items():
            P[k].copy_(b + scales[k] * (P[k] - b))


def _shape_anchor(P: dict, rows, s0, q0, o0):
    """Squared change of the ``rows``' sizes, orientations and opacities from
    their start (summed over the rows), each in a unit like the centres'
    0.02 * extent: ln 2 per axis (twice the size), 20 degrees, 0.1 opacity."""
    import torch
    import torch.nn.functional as F
    ds = ((P["logs"][rows] - s0[rows]) / math.log(2.0)) ** 2
    dq = (1 - (F.normalize(P["quats"][rows], dim=-1) * q0[rows]).sum(-1) ** 2) / math.sin(math.radians(10)) ** 2
    do = ((torch.sigmoid(P["opl"][rows]) - o0[rows]) / 0.1) ** 2
    return ds.sum() + dq.sum() + do.sum()


def _constraint_stats(P: dict, m0, s0, q0, o0, births) -> dict:
    """How far the births and the trained kept Gaussians moved: drift (scene
    units), splat area (two largest axes) against the start, rotation
    (degrees) and opacity change; median and 90th percentile."""
    import torch
    import torch.nn.functional as F

    def q(x, qs=(50, 90)):
        return {f"p{k}": round(float(np.percentile(x, k)), 5) for k in qs} if len(x) else {}
    with torch.no_grad():
        drift = torch.linalg.norm(P["means"] - m0, dim=1).cpu().numpy()
        area = torch.exp(torch.sort(P["logs"], dim=1).values[:, 1:].sum(1)
                         - torch.sort(s0, dim=1).values[:, 1:].sum(1)).cpu().numpy()
        dot = (F.normalize(P["quats"], dim=-1) * q0).sum(-1).abs().clamp(max=1)
        rot = torch.rad2deg(2 * torch.acos(dot)).cpu().numpy()
        dop = (torch.sigmoid(P["opl"]) - o0).cpu().numpy()
    out = {}
    for name, rows in (("births", births), ("kept", ~births)):
        out[name] = {"rows": int(rows.sum()), "drift": q(drift[rows]), "area_ratio": q(area[rows]),
                     "area_gt2x_pct": round(100 * float((area[rows] > 2).mean()), 2) if rows.any() else None,
                     "area_gt4x_pct": round(100 * float((area[rows] > 4).mean()), 2) if rows.any() else None,
                     "rotation_deg": q(rot[rows]), "opacity_delta": q(dop[rows], (10, 50, 90))}
    return out


def multiview_optimize(*, source_ply: Path, init_ply: Path, cameras, photos: dict, masks: dict, chain: list[str],
                       edits: dict, object_ids: np.ndarray, video: list[dict], out_ply: Path,
                       iterations: int = 10000, anchor: bool = True, freeze_view_colour: bool = True,
                       device: str = "cuda", log: Callable = _LOG.info, progress: Callable | None = None,
                       preview_dir: Path | None = None, held_out: set | None = None,
                       freeze_geometry: bool = False, train_kept: bool = True,
                       protect: np.ndarray | None = None, geo_lr_births: float = 1.0, geo_lr_kept: float = 1.0,
                       births_anchor: float = 0.0, protect_kept: bool = False) -> dict:
    """Optimise the hole's Gaussians against every camera at once.

    Trainable: the births, and the kept Gaussians inside the removed object's
    box (grown 15 %) that project into a chain camera's grown selection.
    Targets: chain cameras' clean images and ``video`` frames (dicts with
    ``viewmat``, ``intrinsic``, ``image`` float RGB, ``hole`` bool) inside
    the hole; every camera's real photo outside its own grown selection.
    ``anchor``: splats stay inside the box and near their start, and the ones
    that still drift far are pruned. ``freeze_view_colour``: the SH rest
    terms keep their values (web viewers draw only the base colour, so trained
    view-dependent terms show there as haze). ``held_out``: chain cameras only
    evaluated (default ``held_out_chain(chain)``). ``freeze_geometry``: only
    colour and opacity train; positions, orientations and sizes stay where
    the births put them (free geometry turned the fill's surface into a
    cloud of discs at random angles: fuzz, with gaps). ``train_kept=False``:
    only the births train; the real Gaussians around the hole keep their
    colours (trained against the cleaned images they turned speckled).
    ``protect``: centres of Gaussians never pruned (the support surface's
    feather band: faint by design, and pruned as weak it left the fill with
    a hard edge). ``geo_lr_births`` / ``geo_lr_kept``: with free geometry,
    the births' and the kept Gaussians' positions, orientations and sizes
    train at these fractions of the rate (``_row_scaled_step``).
    ``births_anchor``: the births' sizes, orientations and opacities are
    also held near their start with this weight (``_shape_anchor``, per row
    as the centres' anchor). ``protect_kept``: the trained kept Gaussians are
    never pruned (many real ones are faint from the start). Kitchen, 24
    October: kept at 0, births at 0.3 with the anchor at 1e-2 removed the
    seams the frozen fill left against the kept surface, without the free
    fit's jittered, grainy disc cloud. The report's ``constraint_stats``
    say how far both moved."""
    import torch

    # ComfyUI runs nodes under inference mode: training needs autograd back.
    with torch.inference_mode(False), torch.enable_grad():
        return _multiview_optimize(**{k: v for k, v in locals().items() if k != "torch"})


def _multiview_optimize(*, source_ply, init_ply, cameras, photos, masks, chain, edits, object_ids, video, out_ply,
                        iterations, anchor, freeze_view_colour, device, log, progress, preview_dir,
                        held_out, freeze_geometry, train_kept, protect, geo_lr_births=1.0, geo_lr_kept=1.0,
                        births_anchor=0.0, protect_kept=False) -> dict:
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from PIL import Image

    from .ply import GaussianPly, write_vertices
    from .renderer import CameraView

    dev = torch.device(device)
    held = held_out_chain(chain) if held_out is None else set(held_out)
    hero = chain[0]
    train_chain = [v for v in chain if v not in held]
    views = []
    for cam in cameras.views:
        v = cam.view_id
        if cam.is_virtual or v not in masks:
            continue
        m = masks[v]
        photo = _unit(photos[v])
        inside = ndimage.binary_dilation(m, iterations=12)
        target, weight = photo.copy(), (~inside).astype(np.float32)
        if v in chain:
            edit = _unit(edits[v])
            target[inside] = edit[inside]
            if v not in held:
                weight[inside] = 2.0 if v == hero else 1.0
            else:
                weight[:] = 0.0
        views.append({"id": v, "cam": cam, "target": torch.from_numpy(target).to(dev),
                      "weight": torch.from_numpy(weight).to(dev), "mask": m})
    for f, item in enumerate(video):
        cam = CameraView(view_id=f"video@{item.get('frame', f)}", viewmat=np.asarray(item["viewmat"], np.float32),
                         intrinsic=np.asarray(item["intrinsic"], np.float32), width=item["image"].shape[1],
                         height=item["image"].shape[0], synthetic=True)
        views.append({"id": cam.view_id, "cam": cam, "target": torch.from_numpy(_unit(item["image"])).to(dev),
                      "weight": torch.from_numpy(item["hole"].astype(np.float32)).to(dev), "mask": item["hole"]})
    log(f"{len(views)} cameras ({len(video)} video frames); chain {chain}; held out {sorted(held)}")

    g = GaussianPly.load(init_ply)
    N = len(g.means)
    src = GaussianPly.load(source_ply)
    src_set = {r.tobytes() for r in np.ascontiguousarray(src.means, np.float32)}
    births = np.array([r.tobytes() not in src_set for r in np.ascontiguousarray(g.means, np.float32)])
    obj = src.means[object_ids]
    del src, src_set
    hits = np.zeros(N, bool)
    for vw in views:
        if vw["id"] not in chain:
            continue
        uv, z = project_to_pixels(g.means, vw["cam"].viewmat, vw["cam"].intrinsic)
        ok = np.isfinite(uv).all(1) & (z > 0)
        x = np.zeros(N, int)
        y = np.zeros(N, int)
        x[ok], y[ok] = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        inside = ndimage.binary_dilation(vw["mask"], iterations=12)
        H, W = inside.shape
        ok &= (x >= 0) & (y >= 0) & (x < W) & (y < H)
        hit = np.zeros(N, bool)
        hit[ok] = inside[y[ok], x[ok]]
        hits |= hit
    lo, hi = np.percentile(obj, 0.5, axis=0), np.percentile(obj, 99.5, axis=0)
    grow = (hi - lo) * 0.15
    in_box = np.all((g.means >= lo - grow) & (g.means <= hi + grow), axis=1)
    train = births | (hits & in_box & ~births & bool(train_kept))
    extent = float(np.linalg.norm(hi - lo))
    log(f"{N} Gaussians, {int(births.sum())} births, {int(train.sum())} trainable")

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    full = {"means": t(g.means), "quats": t(g.quaternions), "logs": t(g.log_scales),
            "opl": t(g.opacity_logits).reshape(-1), "dc": t(g.dc), "rest": t(g.rest)}
    ti = torch.from_numpy(np.flatnonzero(train)).to(dev)
    fi = torch.from_numpy(np.flatnonzero(~train)).to(dev)
    P = {k: torch.nn.Parameter(v[ti].clone()) for k, v in full.items()}
    frozen = {k: v[fi] for k, v in full.items()}
    del full
    box_lo = torch.tensor(lo - grow, dtype=torch.float32, device=dev)
    box_hi = torch.tensor(hi + grow, dtype=torch.float32, device=dev)
    m0 = P["means"].detach().clone()
    s0, o0 = P["logs"].detach().clone(), torch.sigmoid(P["opl"].detach())
    q0 = F.normalize(P["quats"].detach(), dim=-1)
    born = torch.from_numpy(births[train]).to(dev)
    scales = {}
    if not freeze_geometry and (geo_lr_births != 1.0 or geo_lr_kept != 1.0):
        row = torch.where(born, float(geo_lr_births), float(geo_lr_kept))[:, None]
        scales = {"means": row, "quats": row, "logs": row}
    geo = 0.0 if freeze_geometry else 1.0
    lrs = {"means": 2e-4 * extent * geo, "quats": 1e-3 * geo, "logs": 5e-3 * geo, "opl": 1e-2, "dc": 2.5e-3,
           "rest": 0.0 if freeze_view_colour else 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [P[k]], "lr": lr} for k, lr in lrs.items()])
    degree = g.max_sh_degree

    def render(vw):
        cam = vw["cam"]
        colours = torch.cat([torch.cat([frozen["dc"], P["dc"]]), torch.cat([frozen["rest"], P["rest"]])], 1)
        img, _a, _m = rasterization(
            means=torch.cat([frozen["means"], P["means"]]),
            quats=F.normalize(torch.cat([frozen["quats"], P["quats"]]), dim=-1),
            scales=torch.exp(torch.cat([frozen["logs"], P["logs"]])),
            opacities=torch.sigmoid(torch.cat([frozen["opl"], P["opl"]])), colors=colours,
            viewmats=torch.from_numpy(np.asarray(cam.viewmat, np.float32)).to(dev)[None],
            Ks=torch.from_numpy(np.asarray(cam.intrinsic, np.float32)).to(dev)[None],
            width=cam.width, height=cam.height, sh_degree=degree, packed=True, rasterize_mode="antialiased")
        return img[0, ..., :3]

    def evaluate(tag):
        res = {}
        with torch.no_grad():
            for vw in views:
                if vw["id"] not in chain:
                    continue
                img = render(vw).clamp(0, 1).cpu().numpy()
                tgt = vw["target"].cpu().numpy()
                m = vw["mask"]
                gr = lambda x: np.abs(x.mean(2) - ndimage.uniform_filter(x.mean(2), 5))[m].mean()
                res[vw["id"]] = {"mae": round(float(np.abs(img - tgt).mean(2)[m].mean()), 4),
                                 "texture": round(float(gr(img) / max(gr(tgt), 1e-6)), 2),
                                 "held_out": vw["id"] in held}
                if preview_dir is not None:
                    Path(preview_dir).mkdir(parents=True, exist_ok=True)
                    Image.fromarray((img * 255).astype(np.uint8)).save(Path(preview_dir) / f"{tag}_{vw['id']}.png")
        log(tag + " " + "  ".join(f"{v}{'*' if r['held_out'] else ''}: {r['mae']:.3f}/{r['texture']:.2f}"
                                  for v, r in res.items()))
        return res

    report = {"chain": chain, "held_out": sorted(held), "trainable": int(train.sum()), "births": int(births.sum()),
              "video_frames": len(video), "iterations": int(iterations), "anchor": bool(anchor),
              "freeze_view_colour": bool(freeze_view_colour), "geo_lr_births": float(geo_lr_births),
              "geo_lr_kept": float(geo_lr_kept), "births_anchor": float(births_anchor), "protect_kept": bool(protect_kept)}
    report["before"] = evaluate("before")
    train_views = [vw for vw in views if float(vw["weight"].sum()) > 0]
    weights = np.array([3.0 if vw["id"] in train_chain else 1.5 if vw["id"].startswith("video@") else 1.0
                        for vw in train_views])
    weights /= weights.sum()
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(int(iterations)):
        vw = train_views[rng.choice(len(train_views), p=weights)]
        img = render(vw)
        w = vw["weight"]
        l1 = (torch.abs(img - vw["target"]).mean(-1) * w).sum() / w.sum().clamp(min=1)
        ls = ((1 - _ssim_map(img, vw["target"])) * w).sum() / w.sum().clamp(min=1)
        o = torch.sigmoid(P["opl"])
        # Mild pressure towards opaque-or-gone: half-transparent fill splats
        # are what side views see as fuzz.
        loss = 0.8 * l1 + 0.2 * ls + 2e-3 * (o * (1 - o)).mean()
        if anchor:
            loss = loss + 1e-2 * (((P["means"] - m0) / (0.02 * extent)) ** 2).mean()
            if births_anchor > 0:
                loss = loss + births_anchor * _shape_anchor(P, born, s0, q0, o0) / max(len(born), 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if scales:
            _row_scaled_step(opt, P, scales)
        else:
            opt.step()
        with torch.no_grad():
            P["logs"].clamp_(max=math.log(extent * 0.05))
            if anchor:
                P["means"].copy_(torch.maximum(torch.minimum(P["means"], box_hi), box_lo))
            # No needles: the longest axis at most 4x the middle one.
            sv, si = torch.sort(P["logs"], dim=1)
            sv[:, 2] = torch.minimum(sv[:, 2], sv[:, 1] + math.log(4.0))
            P["logs"].scatter_(1, si, sv)
        if it % 250 == 0:
            log(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s")
        if progress is not None:
            progress(it + 1, int(iterations))
    report["after"] = evaluate("after")
    report["constraint_stats"] = _constraint_stats(P, m0, s0, q0, o0, born.cpu().numpy())

    with torch.no_grad():
        weak = torch.sigmoid(P["opl"]) < 0.05
        if anchor:
            far = torch.linalg.norm(P["means"] - m0, dim=1) > 0.08 * extent
            report["pruned_drifted"] = int((far & ~weak).sum())
            weak = weak | far
        if protect is not None and len(protect):
            keys = {r.tobytes() for r in np.ascontiguousarray(protect, np.float32)}
            start = np.ascontiguousarray(m0.cpu().numpy(), np.float32)
            kept = torch.from_numpy(np.array([r.tobytes() in keys for r in start])).to(dev)
            report["protected"] = int((weak & kept).sum())
            weak = weak & ~kept
        if protect_kept:
            report["protected_kept"] = int((weak & ~born).sum())
            weak = weak & born
        T = {k: v.detach()[~weak] for k, v in P.items()}
    report["pruned"] = int(weak.sum())
    log(f"pruned {report['pruned']} trained splats ({report.get('pruned_drifted', 0)} drifted); "
        f"{len(T['means'])} trained splats saved")
    tv = np.zeros(len(T["means"]), dtype=g.vertex.dtype)
    mn = T["means"].cpu().numpy()
    q = F.normalize(T["quats"], dim=-1).cpu().numpy()
    sc = T["logs"].cpu().numpy()
    for i, k in enumerate("xyz"):
        tv[k] = mn[:, i]
    for i in range(4):
        tv[f"rot_{i}"] = q[:, i]
    for i in range(3):
        tv[f"scale_{i}"] = sc[:, i]
    tv["opacity"] = T["opl"].cpu().numpy()
    d = T["dc"].cpu().numpy().reshape(len(tv), -1)
    for i in range(3):
        tv[f"f_dc_{i}"] = d[:, i]
    r = T["rest"].cpu().numpy()
    K = r.shape[1]
    for c in range(3):
        for j in range(K):
            tv[f"f_rest_{c * K + j}"] = r[:, j, c]
    for name in ("nx", "ny", "nz"):
        if name in (tv.dtype.names or ()):
            tv[name] = 0
    write_vertices(g, Path(out_ply), np.concatenate([g.vertex[~train], tv]),
                   f"SpEdE all-around fill, {int(iterations)} iterations")
    report["ply"] = str(out_ply)
    report["total_gaussians"] = int((~train).sum() + len(tv))
    report["seconds"] = round(time.time() - t0, 1)
    return report
