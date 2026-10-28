"""Progressive all-around fill: one camera at a time, each seeing the fill so far.

After the hero's depth fill, the cameras are visited in angular order around
the removed object, each next to one already done. For each camera:

1. The current scene is rendered into it. Inside the hole, pixels the render
   does not cover are marked magenta.
2. An image editor (Qwen-Image 2.1) cleans the render's hole: it fills the
   magenta and smooths the repair into the surrounding surfaces, with the
   nearest camera already done as a reference. Only the hole region of its
   result is used; the rest stays the render.
3. Births only where the render was not covered: that camera's depth (MoGe)
   aligned to the render's own depth around them. Nothing is carved.
4. A short optimisation of the hole's colours against every camera done so
   far (their clean images inside the hole, every real photo outside it).
   Geometry stays where the births put it: trained freely, the surface broke
   into a cloud of discs at random angles (bonsai: discs 58 degrees off their
   local surface against 0.3 for the hero's births alone; fuzz with gaps).

Each step thus sees mostly content earlier steps settled, and the editor only
has to fill thin bands and blend; it never invents the whole hole anew per
view, which is what made per-view edits disagree.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .multiview_fill import MAGENTA, multiview_init, multiview_optimize, object_hole

_LOG = logging.getLogger(__name__)

CLEAN_PROMPT = ("Image 1 is a render of a 3D scan that has been repaired where an object was removed. The repair "
                "shows as a blotchy area: lighter smudged patches, dark stains, streaks and magenta missing parts. "
                "Clean it up so the repaired area looks like a natural, continuous part of the scene, matching the "
                "surrounding surfaces, textures, lighting and shading in image 1 and consistent with image 2, "
                "which shows the same place from a nearby viewpoint. Do not add any objects. Keep the composition, "
                "the camera and everything that is not part of the repair exactly unchanged.")


def match_seam_tone(edited: np.ndarray, render: np.ndarray, region: np.ndarray, band: int = 12,
                    limits: tuple[float, float] = (0.8, 1.4)) -> tuple[np.ndarray, np.ndarray]:
    """Scale the edited image's colour channels so the band just inside
    ``region`` matches the real scene in the band just outside it. Qwen
    darkened its cleaned area by 10-20 % in most views; repeated over the
    steps, the whole fill came out darker than the cloth around it."""
    inner = region & ~ndimage.binary_erosion(region, iterations=band)
    outer = ndimage.binary_dilation(region, iterations=band) & ~region
    if inner.sum() < 100 or outer.sum() < 100:
        return edited, np.ones(3)
    e = np.asarray(edited, np.float64)
    gain = np.asarray(render, np.float64)[outer].mean(0) / np.maximum(e[inner].mean(0), 1.0)
    gain = np.clip(gain, *limits)
    out = e.copy()
    out[region] = e[region] * gain
    return out.clip(0, 255).astype(np.uint8), gain


def _membrane(values: np.ndarray, known: np.ndarray, iterations: int = 600) -> np.ndarray:
    """Harmonic interpolation: ``known`` pixels keep ``values``, the others
    become the smooth average of their neighbours (Jacobi iterations from a
    nearest-known start)."""
    _d, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    out = values[iy, ix].astype(np.float64)
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    for _ in range(iterations):
        avg = ndimage.convolve(out, k, mode="nearest")
        out = np.where(known, values, avg)
    return out


def match_seam_tone_field(edited: np.ndarray, render: np.ndarray, region: np.ndarray, band: int = 12,
                          limits: tuple[float, float] = (0.6, 1.6), work: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Like ``match_seam_tone`` but local: the ratio of the real scene just
    outside the seam to the edit just inside it is measured all along the
    seam and interpolated smoothly into the region, per channel. A table lit
    from the front is brighter there than at the back; one gain for the
    whole seam made the fill too dark in front and too bright behind."""
    h, w = region.shape
    f = max(1, h // work)
    sm = lambda a: a[::f, ::f]
    reg = sm(region)
    inner = reg & ~ndimage.binary_erosion(reg, iterations=max(1, band // f))
    outer = ndimage.binary_dilation(reg, iterations=max(1, band // f)) & ~reg
    if inner.sum() < 20 or outer.sum() < 20:
        return edited, np.ones(3)
    sigma = max(1.0, 2 * band / f)
    e = np.asarray(edited, np.float64)
    r = np.asarray(render, np.float64)
    seam = ndimage.binary_dilation(reg, iterations=1) & ~ndimage.binary_erosion(reg, iterations=1)
    gains = []
    out = e.copy()
    for c in range(3):
        def local(img, m):
            num = ndimage.gaussian_filter(np.where(m, sm(img[..., c]), 0.0), sigma)
            den = ndimage.gaussian_filter(m.astype(np.float64), sigma)
            return num / np.maximum(den, 1e-6), den
        ro, dro = local(r, outer)
        ei, dei = local(e, inner)
        ok = seam & (dro > 0.05) & (dei > 0.05)
        ratio = np.clip(ro / np.maximum(ei, 1.0), *limits)
        if ok.sum() < 10:
            gains.append(1.0)
            continue
        field = _membrane(np.where(ok, ratio, 1.0), ok)
        full = np.kron(field, np.ones((f, f)))[:h, :w]
        if full.shape != (h, w):
            full = np.pad(full, ((0, h - full.shape[0]), (0, w - full.shape[1])), mode="edge")
        out[..., c] = np.where(region, e[..., c] * full, e[..., c])
        gains.append(float(np.median(field[reg])))
    return out.clip(0, 255).astype(np.uint8), np.asarray(gains)


def fill_tone_target(rgb: np.ndarray, fill: np.ndarray, band: int = 24, work: int = 256,
                     amp_limits: tuple[float, float] = (0.3, 1.0)) -> np.ndarray:
    """What a rendered fill should look like next to the real surface around
    it: the surroundings' local mean colour interpolated smoothly over the
    fill (a membrane from the band around it), plus the fill's own texture
    scaled to the surroundings' texture strength. ``rgb``: the render
    (H x W x 3, 0..255), ``fill``: where it shows the fill. Returns the
    target image (float, equal to ``rgb`` outside the fill)."""
    h, w = fill.shape
    f = max(1, h // work)
    rgb = np.asarray(rgb, np.float64)
    small = rgb[::f, ::f]
    reg = fill[::f, ::f]
    b = max(2, band // f)
    ring = ndimage.binary_dilation(reg, iterations=b) & ~ndimage.binary_dilation(reg, iterations=2)
    out = rgb.copy()
    if ring.sum() < 20 or reg.sum() < 20:
        return out
    sigma = max(1.0, b / 2)

    def local(img, m):
        num = ndimage.gaussian_filter(np.where(m, img, 0.0), sigma)
        den = ndimage.gaussian_filter(m.astype(np.float64), sigma)
        return num / np.maximum(den, 1e-6)
    zoom = (h / reg.shape[0], w / reg.shape[1])
    targets, lows = [], []
    for c in range(3):
        ring_mean = local(small[..., c], ring)
        known = ring & (ndimage.gaussian_filter(ring.astype(np.float64), sigma) > 0.05)
        target = _membrane(np.where(known, ring_mean, 0.0), known, iterations=400)
        own = local(small[..., c], reg)
        targets.append(ndimage.zoom(target, zoom, order=1)[:h, :w])
        lows.append((ndimage.zoom(own, zoom, order=1)[:h, :w], ring_mean))
    ring_full = ndimage.zoom(ring.astype(np.uint8), zoom, order=0)[:h, :w].astype(bool)
    detail_fill = np.concatenate([(rgb[..., c] - lows[c][0])[fill] for c in range(3)])
    detail_ring = np.concatenate([(small[..., c] - lows[c][1])[ring] for c in range(3)])
    amp = float(np.clip(np.std(detail_ring) / max(np.std(detail_fill), 1e-6), *amp_limits)) if ring_full.any() else 1.0
    for c in range(3):
        out[..., c] = np.where(fill, targets[c] + amp * (rgb[..., c] - lows[c][0]), rgb[..., c])
    return out.clip(0, 255)


def correct_disc_tone(ply_path: Path, disc_means: np.ndarray, cameras, views: list[str], size: int, device: str,
                      iterations: int = 2, structure: bool = False) -> dict:
    """Render-space tone correction of the disc layer: in each view, the
    render of the fill is compared with ``fill_tone_target`` and each disc's
    base colour moves by the facing-weighted mean difference at its pixel.
    The renders include the view-dependent terms and the real surface's
    layering, which the base-colour harmonisation cannot see: after best-view
    texturing the bonsai fill showed as a lighter, high-contrast knit patch.
    ``structure``: keep the fill's own large-scale structure (mat edges, a
    gap) and only scale it by a smooth gain field measured along the seam
    (``match_seam_tone_field``) instead of replacing its low frequencies."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    from .geometry_utils import project_to_pixels
    from .ply import GaussianPly, write_vertices
    from .surface_fill import SH_C0, _disc_normals

    dev = torch.device(device)

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    report = []
    for it in range(iterations):
        g = GaussianPly.load(ply_path)
        index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(g.means, np.float32))}
        rows = np.array([index.get(r.tobytes(), -1) for r in np.ascontiguousarray(disc_means, np.float32)])
        rows = rows[rows >= 0]
        M = np.asarray(g.means, np.float64)[rows]
        normals = _disc_normals(g.quaternions[rows], g.log_scales[rows])
        P = dict(means=t(g.means), quats=F.normalize(t(g.quaternions), dim=-1), scales=torch.exp(t(g.log_scales)),
                 opacities=torch.sigmoid(t(g.opacity_logits)).reshape(-1))
        colours = torch.cat([t(g.dc), t(g.rest)], 1)
        marker = torch.zeros(len(g.means), 3, device=dev)
        marker[torch.from_numpy(rows).to(dev)] = 1
        delta = np.zeros((len(rows), 3))
        weight = np.zeros(len(rows))
        for v in views:
            cam = cameras.by_id[v]
            K = np.asarray(cam.intrinsic, np.float64).copy()
            K[:2] *= size / cam.width
            with torch.no_grad():
                img, alpha, _ = rasterization(**P, colors=colours, viewmats=t(cam.viewmat)[None], Ks=t(K)[None],
                                              width=size, height=size, sh_degree=g.max_sh_degree, packed=True,
                                              render_mode="RGB+ED")
                m, _a, _ = rasterization(**P, colors=marker, viewmats=t(cam.viewmat)[None], Ks=t(K)[None],
                                         width=size, height=size, sh_degree=None, packed=True)
            rgb = img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255
            depth = img[0, ..., 3].cpu().numpy()
            fill = (m[0, ..., 0].cpu().numpy() > 0.9) & (alpha[0, ..., 0].cpu().numpy() > 0.9)
            if fill.sum() < 200:
                continue
            if structure:
                target = match_seam_tone_field(rgb, rgb, fill, band=12 * size // 1024)[0].astype(np.float64)
            else:
                target = fill_tone_target(rgb, fill, band=24 * size // 1024)
            uv, z = project_to_pixels(M, cam.viewmat, K)
            ok = np.isfinite(uv).all(1) & (z > 0)
            x = np.where(ok, uv[:, 0], -1).astype(int)
            y = np.where(ok, uv[:, 1], -1).astype(int)
            ok &= (x >= 0) & (y >= 0) & (x < size) & (y < size)
            xc, yc = x.clip(0, size - 1), y.clip(0, size - 1)
            ok &= fill[yc, xc] & ~(z > depth[yc, xc] * 1.03 + 1e-6)
            centre = np.linalg.inv(np.asarray(cam.viewmat, np.float64))[:3, 3]
            to_cam = centre - M
            to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
            wv = np.where(ok, np.abs(np.sum(normals * to_cam, 1)), 0.0)
            delta += wv[:, None] * (target[yc, xc] - rgb[yc, xc]) / 255
            weight += wv
            del img, alpha, m
        seen = weight > 1e-6
        base = np.asarray(g.dc, np.float64)[rows, 0] * SH_C0 + 0.5
        base[seen] += delta[seen] / weight[seen, None]
        vtx = g.vertex.copy()
        for c in range(3):
            vtx[f"f_dc_{c}"][rows[seen]] = ((np.clip(base[seen, c], 0, 1) - 0.5) / SH_C0).astype(np.float32)
        tmp = Path(ply_path).with_name(Path(ply_path).stem + "_tone.ply")
        write_vertices(g, tmp, vtx, "SpEdE: disc tone matched to the surroundings in the renders")
        tmp.replace(ply_path)
        report.append({"corrected": int(seen.sum()),
                       "mean_abs_shift": round(float(np.abs(delta[seen] / weight[seen, None]).mean()), 4)})
        del P, colours, marker, g
        torch.cuda.empty_cache()
    return {"passes": report}


def top_camera(disc_means: np.ndarray, axes, cameras, size: int, margin: float = 1.4, radius: float | None = None,
               centre: np.ndarray | None = None):
    """A virtual camera straight above the support surface's discs, at the
    real cameras' median distance, framing them with ``margin``."""
    from types import SimpleNamespace

    up, e1, _e2 = (np.asarray(a, np.float64) for a in axes)
    if centre is None:
        centre = np.asarray(disc_means, np.float64).mean(0)
    if radius is None:
        d = np.asarray(disc_means, np.float64) - centre
        radius = float(np.percentile(np.linalg.norm(d - np.outer(d @ up, up), axis=1), 99))
    dist = float(np.median([np.linalg.norm(np.linalg.inv(np.asarray(v.viewmat, np.float64))[:3, 3] - centre)
                            for v in cameras.views if not v.is_virtual]))
    position = centre + dist * up
    z = -up
    x = e1 - (e1 @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ position
    f = 0.5 * size * dist / (margin * max(radius, 1e-6))
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    return SimpleNamespace(view_id="top", viewmat=vm, intrinsic=K, width=size, height=size, is_virtual=True)


def texture_from_top(ply: Path, disc_means: np.ndarray, axes, cameras, reference: np.ndarray,
                     clean: Callable, seed: int, size: int, device: str, out_dir: Path, ring: int = 40) -> dict:
    """The support surface's layout decided once, in one image: the scene is
    rendered from straight above, the disc area (but not its feather) is
    marked magenta, the editor paints it with the surroundings as context,
    and the discs take their colour from that single image. Per-view edits
    each drew the kitchen mats' edges and the gap between them in a
    different place, and any blend of them smeared; from above, the whole
    fill is seen squarely, at once, with nothing in front. ``ring`` (pixels
    at 1024): the painted area reaches this far past the discs onto the
    real surface, whose Gaussians there fade to the painted colours. The
    kitchen's orange mat had a dark stain (or the removed object's shadow)
    just around the fill, which the feather band faded into and which drew
    the seam from every side. 0: the disc area only, minus its feather band
    (kept as context). Returns the view dict for ``texture_discs_from_views``."""
    from PIL import Image

    from .surface_fill import texture_discs_from_views

    cam = top_camera(disc_means, axes, cameras, size)
    rgb, alpha, K, depth = _render(ply, cam, size, device, depth=True)
    from .geometry_utils import project_to_pixels
    uv, z = project_to_pixels(disc_means, cam.viewmat, K)
    ok = np.isfinite(uv).all(1) & (z > 0)
    px = uv[ok].astype(int)
    px = px[(px >= 0).all(1) & (px < size).all(1)]
    disc_area = np.zeros((size, size), bool)
    disc_area[px[:, 1], px[:, 0]] = True
    disc_area = ndimage.binary_fill_holes(ndimage.binary_closing(disc_area, iterations=4 * size // 1024 + 2))
    ring_px = int(ring) * size // 1024
    if ring_px:
        hole = ndimage.binary_dilation(disc_area, iterations=ring_px)
    else:
        # The feather band shows the real surface through the discs: context.
        hole = ndimage.binary_erosion(disc_area, iterations=12 * size // 1024)
    marked = rgb.copy()
    marked[hole | (alpha < 0.5)] = MAGENTA
    for attempt in range(3):
        q = clean(marked, reference, seed + 1000 * attempt)
        if q.shape[:2] != (size, size):
            q = _resize(q, size)
        if (_magenta_left(q) & hole).sum() <= 0.001 * max(hole.sum(), 1):
            break
    q = np.where(ndimage.binary_dilation(_magenta_left(q), iterations=3)[..., None], rgb, q)
    q, _gain = match_seam_tone_field(q, rgb, hole, band=12 * size // 1024)
    if ring_px:
        # Full weight over the discs, fading to none at the ring's end.
        w = np.clip(1.0 - ndimage.distance_transform_edt(~disc_area) / (ring_px + 1), 0.0, 1.0)
        w = ndimage.gaussian_filter(w.astype(np.float32), 2 * size / 1024)
    else:
        w = ndimage.gaussian_filter(disc_area.astype(np.float32), 3 * size / 1024)
    result = (q * w[..., None] + rgb * (1 - w[..., None]) + 0.5).clip(0, 255).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in (("render", rgb), ("marked", marked), ("clean", result)):
        Image.fromarray(img).save(out_dir / f"top_{name}.png")
    view = {"name": "top", "image": result, "viewmat": cam.viewmat, "intrinsic": K, "depth": depth,
            "region": disc_area}
    rep = texture_discs_from_views(ply, disc_means, [view])
    out = {"view": view, "textured": rep["textured"], "disc_area_px": int(disc_area.sum())}
    if ring_px:
        out["surface_recoloured"] = recolour_surface_under(ply, disc_means, q, w, cam.viewmat, K, depth)
    return out


def recolour_surface_under(ply: Path, disc_means: np.ndarray, painted: np.ndarray, weight: np.ndarray,
                           viewmat, K, depth: np.ndarray, depth_band: float = 0.03) -> int:
    """The real Gaussians seen in a painted top view (at the surface's
    depth, not above or below it) fade to its colours by ``weight`` (HxW,
    1: fully the painted colour). Returns how many changed."""
    from .geometry_utils import project_to_pixels
    from .ply import GaussianPly
    from .surface_fill import blend_colours_from_view

    g = GaussianPly.load(ply)
    M = np.asarray(g.means, np.float64)
    discs = {r.tobytes() for r in np.ascontiguousarray(disc_means, np.float32)}
    real = np.array([r.tobytes() not in discs for r in np.ascontiguousarray(g.means, np.float32)])
    del g
    h, w_ = weight.shape
    uv, z = project_to_pixels(M, viewmat, K)
    ok = real & np.isfinite(uv).all(1) & (z > 0)
    x = np.where(ok, uv[:, 0], -1).astype(int)
    y = np.where(ok, uv[:, 1], -1).astype(int)
    ok &= (x >= 0) & (y >= 0) & (x < w_) & (y < h)
    xc, yc = x.clip(0, w_ - 1), y.clip(0, h - 1)
    ok &= weight[yc, xc] > 0.01
    ok &= np.abs(z - depth[yc, xc]) < depth_band * depth[yc, xc]
    sel = np.flatnonzero(ok)
    if not len(sel):
        return 0
    view = {"name": "top_painted", "image": painted, "viewmat": viewmat, "intrinsic": K, "depth": depth,
            "region": weight > 0.01}
    return blend_colours_from_view(ply, M[sel].astype(np.float32), 1.0 - weight[yc[sel], xc[sel]], view)["blended"]


def _magenta_left(img: np.ndarray) -> np.ndarray:
    """Pixels an edit left magenta (or nearly)."""
    a = np.asarray(img, np.int32)
    return (a[..., 0] > 200) & (a[..., 1] < 80) & (a[..., 2] > 200)


def _magenta_tinted(img: np.ndarray, region: np.ndarray) -> float:
    """Share of ``region`` that is clearly purple: an edit that did not
    understand the marker sometimes paints a smooth purple wash instead."""
    a = np.asarray(img, np.int32)
    purple = (a[..., 0] - a[..., 1] > 50) & (a[..., 2] - a[..., 1] > 50)
    return float(purple[region].mean()) if region.any() else 0.0


def texture_top_tiles(ply: Path, disc_means: np.ndarray, disc_ramp: np.ndarray, axes, cameras, clean: Callable,
                      seed: int, size: int, device: str, out_dir: Path, *, px_per_disc: float = 3.0,
                      max_tiles: int = 4, overlap: float = 0.25) -> list[dict]:
    """Sharper detail for the support surface: after ``texture_from_top``
    has decided the layout in one image, the disc area is split into tiles
    seen from closer above, so the edit has about ``px_per_disc`` pixels per
    disc spacing (the single top view: 1.5). Each tile's render, with its
    not yet sharpened discs (but not the feather) magenta, is edited with
    the same tile rendered from the coarse layout as the reference, and its
    discs take their colour from the result. Tiles go in order of the most
    context first; overlaps show the tiles already done. Returns, per tile,
    its view dict and the discs it coloured (``rows``: indices into
    ``disc_means``)."""
    from PIL import Image

    from .geometry_utils import project_to_pixels
    from .surface_fill import texture_discs_from_views

    up, e1, e2 = (np.asarray(a, np.float64) for a in axes)
    D = np.asarray(disc_means, np.float64)
    spacing = float(np.median(cKDTree(D).query(D, k=2)[0][:, 1]))
    u, w = D @ e1, D @ e2
    target = px_per_disc / spacing
    tiles = []
    nu = int(np.clip(np.ceil(np.ptp(u) * (1 + overlap) * 1.15 * target / size), 1, max_tiles))
    nw = int(np.clip(np.ceil(np.ptp(w) * (1 + overlap) * 1.15 * target / size), 1, max_tiles))
    su, sw = np.ptp(u) / nu, np.ptp(w) / nw
    for i in range(nu):
        for j in range(nw):
            lo_u, lo_w = u.min() + i * su, w.min() + j * sw
            inside = ((u >= lo_u - overlap * su / 2) & (u <= lo_u + su + overlap * su / 2)
                      & (w >= lo_w - overlap * sw / 2) & (w <= lo_w + sw + overlap * sw / 2))
            core = (u >= lo_u) & (u <= lo_u + su) & (w >= lo_w) & (w <= lo_w + sw)
            if core.sum() >= 50:
                # Every tile at the same scale, framed on the tile itself
                # (not on its discs: an edge tile with few discs was framed
                # so close that the weave filled the frame and the editor
                # painted a smooth purple blob).
                cu, cw = lo_u + su / 2, lo_w + sw / 2
                h = float(np.median(D[inside] @ up))
                tiles.append((inside, core, cu * e1 + cw * e2 + h * up))
    done = np.zeros(len(D), bool)
    hole_disc = np.asarray(disc_ramp) >= 0.999
    results = []
    remaining = list(range(len(tiles)))
    step = 0
    while remaining:
        # Most context first: the smallest share of unpainted fill.
        k = min(remaining, key=lambda t: (tiles[t][0] & hole_disc & ~done).sum() / max(tiles[t][0].sum(), 1))
        remaining.remove(k)
        inside, core, tile_centre = tiles[k]
        cam = top_camera(D[inside], axes, cameras, size, margin=1.05, centre=tile_centre,
                         radius=0.5 * max(su, sw) * (1 + overlap))
        rgb, alpha, K, depth = _render(ply, cam, size, device, depth=True)
        uv, z = project_to_pixels(D, cam.viewmat, K)
        okp = np.isfinite(uv).all(1) & (z > 0)
        px = np.where(okp[:, None], uv, -1).astype(int)
        inimg = okp & (px >= 0).all(1) & (px < size).all(1)
        closing = 2 * size // 1024 + 2

        def area(sel):
            m = np.zeros((size, size), bool)
            q = px[sel & inimg]
            m[q[:, 1], q[:, 0]] = True
            return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=closing)) if m.any() else m
        hole = area(hole_disc & ~done)
        marked = rgb.copy()
        marked[hole] = MAGENTA
        for attempt in range(4):
            q = clean(marked, rgb, seed + 7 * step + 1000 * attempt)
            if q.shape[:2] != (size, size):
                q = _resize(q, size)
            left = _magenta_left(q) & hole
            tint = _magenta_tinted(q, hole) - _magenta_tinted(rgb, hole)
            if left.sum() <= 0.001 * max(hole.sum(), 1) and tint < 0.05:
                break
        else:
            # Still failed: this tile keeps the coarse layout.
            q = rgb.copy()
        # Kitchen, 21 October: one tile came back with a magenta triangle
        # in it and the discs took it. What is still magenta keeps the
        # coarse layout.
        left = ndimage.binary_dilation(_magenta_left(q), iterations=3)
        q = np.where(left[..., None], rgb, q)
        if hole.any():
            q, _gain = match_seam_tone_field(q, rgb, hole, band=12 * size // 1024)
        wgt = ndimage.gaussian_filter(hole.astype(np.float32), 2 * size / 1024)[..., None]
        result = (q * wgt + rgb * (1 - wgt) + 0.5).clip(0, 255).astype(np.uint8)
        for name, img in (("render", rgb), ("marked", marked), ("clean", result)):
            Image.fromarray(img).save(out_dir / f"tile{step:02d}_{name}.png")
        # This tile colours its own core's discs not done yet (overlaps are
        # context only), and the feather discs in it.
        rows = np.flatnonzero(core & ~done & inimg)
        view = {"name": f"tile{step:02d}", "image": result, "viewmat": cam.viewmat, "intrinsic": K,
                "depth": depth, "region": area(core)}
        if len(rows):
            texture_discs_from_views(ply, D[rows], [view])
        done[rows] = True
        results.append({"view": view, "rows": rows})
        step += 1
    return results


def visit_order(cameras, views: list[str], hero: str, centre: np.ndarray) -> list[tuple[str, str]]:
    """(camera, nearest camera already visited) in growing order from the
    hero: always the unvisited camera closest in viewing direction to one
    already visited."""
    def direction(v):
        c = np.linalg.inv(np.asarray(cameras.by_id[v].viewmat, np.float64))[:3, 3]
        d = np.asarray(centre, np.float64) - c
        return d / np.linalg.norm(d)
    dirs = {v: direction(v) for v in set(views) | {hero}}
    done, order = [hero], []
    left = [v for v in views if v != hero]
    while left:
        best = max(((float(dirs[v] @ dirs[d]), v, d) for v in left for d in done))
        _cos, v, d = best
        order.append((v, d))
        done.append(v)
        left.remove(v)
    return order


def _resize(img: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    from PIL import Image
    if img.dtype == bool:
        return np.asarray(Image.fromarray(img.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST if nearest else Image.LANCZOS))


def _render(ply: Path, cam, size: int, device: str, depth: bool = False, hide: np.ndarray | None = None):
    """``hide``: centres of Gaussians rendered fully transparent."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    from .ply import GaussianPly
    g = GaussianPly.load(ply)
    dev = torch.device(device)

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    K = np.asarray(cam.intrinsic, np.float64).copy()
    K[:2] *= size / cam.width
    opacities = torch.sigmoid(t(g.opacity_logits)).reshape(-1)
    if hide is not None and len(hide):
        index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(g.means, np.float32))}
        rows = np.array([index.get(r.tobytes(), -1) for r in np.ascontiguousarray(hide, np.float32)])
        opacities[torch.from_numpy(rows[rows >= 0]).to(dev)] = 0.0
        del index
    with torch.no_grad():
        img, alpha, _ = rasterization(
            means=t(g.means), quats=F.normalize(t(g.quaternions), dim=-1), scales=torch.exp(t(g.log_scales)),
            opacities=opacities, colors=torch.cat([t(g.dc), t(g.rest)], 1),
            viewmats=t(cam.viewmat)[None], Ks=t(K)[None], width=size, height=size, sh_degree=g.max_sh_degree,
            packed=True, rasterize_mode="antialiased", render_mode="RGB+ED" if depth else "RGB")
    rgb = (img[0, ..., :3].clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    a = alpha[0, ..., 0].cpu().numpy()
    d = img[0, ..., 3].cpu().numpy() if depth else None
    del g, img, alpha
    torch.cuda.empty_cache()
    return (rgb, a, K, d) if depth else (rgb, a, K)


def _hole_maker(cameras, obj: np.ndarray, selections: dict) -> Callable[[str, int], np.ndarray]:
    """``hole_at(v, s)``: the removed object's hole in camera v at s px (its
    centres projected and grown, or the camera's selection)."""
    def hole_at(v, s):
        cam = cameras.by_id[v]
        K = np.asarray(cam.intrinsic, np.float64).copy()
        K[:2] *= s / cam.width
        h = object_hole(obj, cam.viewmat, K, s, dilate=max(2, 16 * s // 1024))
        if v in selections:
            h |= _resize(np.asarray(selections[v], bool), s)
        return h
    return hole_at


def _support_surface(*, start_ply: Path, obj: np.ndarray, cameras, hero: str, hero_big: np.ndarray,
                     clean: Callable, seed: int, size: int, device: str, out_dir: Path, disc_density: float,
                     top_texture: bool, log: Callable, restore_outside: bool = False) -> dict:
    """The support surface (surface_fill.py) coloured from the hero, then, with
    ``top_texture``, from a painted top view and its tiles. Returns the
    filled scene's path, the surface report, the discs' centres and feather
    ramp, the band discs to protect from pruning, and the top view and tiles
    for the final texturing."""
    from .multiview_fill import orbit_axes
    from .ply import GaussianPly
    from .surface_fill import colour_discs_from_view, support_surface_fill

    filled = out_dir / "support_surface.ply"
    removed = Path(start_ply).with_name("removed_scene.ply")
    extra = {"restore_outside": True} if restore_outside else {}
    surface = support_surface_fill(scene_ply=Path(start_ply), object_means=obj, axes=orbit_axes(cameras),
                                   density=disc_density, out_ply=filled,
                                   observed_ply=removed if removed.exists() else None, log=log, **extra)
    hc = cameras.by_id[hero]
    Kh = np.asarray(hc.intrinsic, np.float64).copy()
    Kh[:2] *= size / hc.width
    surface["coloured_from_hero"] = colour_discs_from_view(filled, surface["discs"], hero_big, hc.viewmat, Kh)
    log(f"support surface: {surface['coloured_from_hero']} of {surface['discs']} discs coloured from the hero")
    _discs = GaussianPly.load(filled)
    disc_means = _discs.means[-surface["discs"]:].copy()
    disc_ramp = (1 / (1 + np.exp(-_discs.opacity_logits[-surface["discs"]:, 0]))) / 0.95
    del _discs
    # The feather band is faint by design: kept through the optimiser's
    # pruning of weak splats.
    protect = disc_means[disc_ramp < 0.999]
    top = None
    top_tiles = []
    if top_texture:
        top = texture_from_top(filled, disc_means, orbit_axes(cameras), cameras, hero_big, clean, seed - 1, size,
                               device, out_dir / "views")
        surface["top_textured"] = top["textured"]
        surface["top_surface_recoloured"] = top.get("surface_recoloured", 0)
        log(f"support surface textured from above: {top['textured']} discs")
        top_tiles = texture_top_tiles(filled, disc_means, np.clip(disc_ramp, 0, 1), orbit_axes(cameras), cameras,
                                      clean, seed - 101, size, device, out_dir / "views")
        surface["top_tiles"] = [len(t["rows"]) for t in top_tiles]
        log(f"support surface sharpened in {len(top_tiles)} tiles: {surface['top_tiles']} discs")
    return {"ply": filled, "surface": surface, "disc_means": disc_means, "disc_ramp": disc_ramp,
            "protect": protect, "top": top, "top_tiles": top_tiles, "Kh": Kh}


def _finish_surface(final: Path, *, surface: dict, disc_means: np.ndarray, disc_ramp: np.ndarray, top, top_tiles,
                    cameras, done: list[str], hero: str, clean_big: dict, region_at: Callable[[str], np.ndarray],
                    size: int, device: str, feather_crossfade: bool, log: Callable) -> None:
    """After the final optimisation: the discs coloured from the clean views
    that see them best (the top view first, then its tiles again), the
    feather band faded to the real surface, the feather ramp set again and
    the tone matched in the renders. ``region_at(u)``: camera u's cleaned
    region at ``size``."""
    from .multiview_fill import orbit_axes
    from .surface_fill import harmonise_disc_colours
    from .surface_fill import set_disc_feather, texture_discs_from_views
    views_tex = []
    for u in done:
        cu = cameras.by_id[u]
        _rgb, _a, Ku, du = _render(final, cu, size, device, depth=True)
        views_tex.append({"name": u, "image": clean_big[u], "viewmat": cu.viewmat, "intrinsic": Ku,
                          "depth": du, "region": region_at(u), "bonus": 1.5 if u == hero else 1.0})
    if top is not None:
        # The top view wins wherever it sees a disc; the others only
        # where it does not.
        top["view"]["bonus"] = 1e3
        views_tex.append(top["view"])
    surface["textured"] = texture_discs_from_views(final, disc_means, views_tex)
    for tile in top_tiles:
        # Each tile again colours the discs it coloured at the start.
        if len(tile["rows"]):
            texture_discs_from_views(final, disc_means[tile["rows"]], [tile["view"]])
    log(f"support surface textured from best views: {surface['textured']}")
    if feather_crossfade:
        from .surface_fill import crossfade_feather_to_surface
        under = top_camera(disc_means, orbit_axes(cameras), cameras, size)
        s_rgb, s_a, s_K, s_d = _render(final, under, size, device, depth=True, hide=disc_means)
        surface["feather_crossfade"] = crossfade_feather_to_surface(
            final, disc_means, np.clip(disc_ramp, 0, 1),
            {"name": "surface", "image": s_rgb, "viewmat": under.viewmat, "intrinsic": s_K, "depth": s_d,
             "region": s_a > 0.9})
        log(f"feather band faded to the real surface's colour: {surface['feather_crossfade']}")
    if top is None:
        # Both replace the fill's low frequencies with the surroundings'
        # interpolated over it: right for an even cloth, but they wipe
        # out large structure (kitchen: the mats' edges and the wood gap
        # became a smooth, glassy smear). The top view's edit is already
        # tone-matched at its seam and carries that structure.
        surface["harmonised"] = harmonise_disc_colours(final, disc_means, orbit_axes(cameras))
    surface["feather_reapplied"] = set_disc_feather(final, disc_means, np.clip(disc_ramp, 0, 1))
    surface["tone"] = correct_disc_tone(final, disc_means, cameras, done, min(size, 768), device,
                                        structure=top is not None)
    log(f"support surface tone matched in the renders: {surface['tone']}")
    log(f"support surface colours harmonised: {surface.get('harmonised', 'skipped (top view)')}")


def progressive_fill(*, source_ply: Path, start_ply: Path, cameras, hero: str, hero_clean: np.ndarray,
                     photos: dict, selections: dict, object_ids: np.ndarray, centre: np.ndarray,
                     clean: Callable[[np.ndarray, np.ndarray, int], np.ndarray], moge_depth: Callable,
                     out_dir: Path, size: int = 1024, seed: int = 241007, attempts: int = 3,
                     max_reframe: float = 0.05, step_iterations: int = 1500, final_iterations: int = 6000,
                     support_surface: bool = True, commit_texture: bool = False, top_texture: bool = True,
                     disc_density: float = 2.0, feather_crossfade: bool = True, device: str = "cuda", log: Callable = _LOG.info,
                     progress: Callable | None = None) -> dict:
    """``clean(marked_rgb, reference_rgb, seed) -> rgb`` (uint8 at ``size``);
    ``selections[v]``: the removed object's mask in camera v (camera size);
    ``hero_clean``: the hero's clean image (uint8, any size). Returns the
    report; the result is ``out_dir/progressive_scene.ply``. ``support_surface``:
    first rebuild the surface the object stood on (surface_fill.py).
    ``feather_crossfade``: the feather band's colour also fades to the real
    surface's (``crossfade_feather_to_surface``)."""
    from PIL import Image

    from .ply import GaussianPly

    out_dir = Path(out_dir)
    (out_dir / "views").mkdir(parents=True, exist_ok=True)
    src = GaussianPly.load(source_ply)
    obj = src.means[object_ids]
    del src
    native = cameras.by_id[hero].width
    views = [v.view_id for v in cameras.views if not v.is_virtual and v.view_id in photos]
    hole_at = _hole_maker(cameras, obj, selections)

    holes = {v: hole_at(v, native) for v in views}
    seen = [v for v in views if holes[v].any()]
    order = visit_order(cameras, seen, hero, centre)
    log(f"{len(order) + 1} cameras see the hole; order " + " ".join(f"{v}<{d}" for v, d in order))

    hero_big = _resize(np.asarray(hero_clean, np.uint8), size)
    Image.fromarray(hero_big).save(out_dir / "views" / f"00_{hero}_clean.png")
    clean_big = {hero: hero_big}
    edits = {hero: _resize(hero_big, native).astype(np.float32) / 255}
    regions = {hero: holes[hero]}
    current = Path(start_ply)
    surface = None
    protect = None
    if support_surface:
        ss = _support_surface(start_ply=current, obj=obj, cameras=cameras, hero=hero, hero_big=hero_big,
                              clean=clean, seed=seed, size=size, device=device, out_dir=out_dir,
                              disc_density=disc_density, top_texture=top_texture, log=log)
        surface, disc_means, disc_ramp, protect = ss["surface"], ss["disc_means"], ss["disc_ramp"], ss["protect"]
        top, top_tiles, Kh = ss["top"], ss["top_tiles"], ss["Kh"]
        current = ss["ply"]
        hc = cameras.by_id[hero]
        _r, _a, _K, hero_depth = _render(current, hc, size, device, depth=True)
        tex_views = [{"name": hero, "image": hero_big, "viewmat": hc.viewmat, "intrinsic": Kh, "depth": hero_depth,
                      "region": hole_at(hero, size), "bonus": 1.5}]
    steps = []
    for k, (v, near) in enumerate(order, start=1):
        cam = cameras.by_id[v]
        step = {"view": v, "reference": near}
        rgb, alpha, _K, depth = _render(current, cam, size, device, depth=True)
        hole = hole_at(v, size)
        region = ndimage.binary_dilation(hole, iterations=40 * size // 1024)
        unknown = hole & ndimage.binary_dilation(alpha < 0.9, iterations=6 * size // 1024)
        marked = rgb.copy()
        marked[unknown] = MAGENTA
        outside = ~ndimage.binary_dilation(region, iterations=10)
        best = None
        for attempt in range(attempts):
            q = clean(marked, clean_big[near], seed + 97 * k + attempt)
            if q.shape[:2] != (size, size):
                q = _resize(q, size)
            diff = ndimage.median_filter(np.abs(q.astype(np.float32) - rgb).mean(2) / 255, 5)
            moved = float((diff[outside] > 0.15).mean())
            if best is None or moved < best[1]:
                best = (q, moved)
            if moved < max_reframe:
                break
        q, moved = best
        q, gain = match_seam_tone_field(q, rgb, region, band=12 * size // 1024)
        step["tone_gain"] = [round(float(x), 3) for x in gain]
        w = ndimage.gaussian_filter(region.astype(np.float32), 4 * size / 1024)[..., None]
        result = (q * w + rgb * (1 - w) + 0.5).clip(0, 255).astype(np.uint8)
        step.update(unknown_px=int(unknown.sum()), reframe=round(moved, 4), attempts=attempt + 1)
        for name, img in (("render", rgb), ("marked", marked), ("clean", result)):
            Image.fromarray(img).save(out_dir / "views" / f"{k:02d}_{v}_{name}.png")
        clean_big[v] = result
        edits[v] = _resize(result, native).astype(np.float32) / 255
        regions[v] = _resize(region, native)

        # Births only where the render was not covered.
        sd = out_dir / f"step_{k:02d}_{v}"
        init = multiview_init(source_ply=source_ply, hero_patch_ply=current, cameras=cameras, chain=[near, v],
                              masks={v: _resize(hole, native)}, edits={v: edits[v]}, object_ids=object_ids,
                              moge_depth=moge_depth, out_dir=sd, device=device, log=log, keep_covered=0.9,
                              carve=False)
        step["geometry"] = init["steps"][0] if init["steps"] else {}
        done = [hero] + [o for o, _d in order[:k]]
        masks = {u: (regions[u] if u in regions else holes[u]) for u in views}
        rep = multiview_optimize(source_ply=source_ply, init_ply=Path(init["ply"]), cameras=cameras,
                                 photos=photos, masks=masks, chain=done, edits=edits, object_ids=object_ids,
                                 video=[], out_ply=sd / "optimised.ply", iterations=step_iterations,
                                 anchor=True, freeze_view_colour=True, device=device, log=lambda _t: None,
                                 held_out=set(), freeze_geometry=True,
                                 train_kept=False, protect=protect)
        step["after"] = {u: r["mae"] for u, r in rep["after"].items()}
        if surface is not None and commit_texture:
            # Commit the discs' colours to the clean views so far, each from
            # the ones that see it best, so the next camera renders settled
            # content. Off: tried 21 October, each pass's footprint blur went
            # through the editor (which keeps what it is shown) into the next
            # view and compounded; bonsai's knit became a flat dark blob and
            # kitchen's mat edges smeared further.
            from .surface_fill import texture_discs_from_views
            tex_views.append({"name": v, "image": result, "viewmat": cam.viewmat, "intrinsic": _K, "depth": depth,
                              "region": region})
            step["textured"] = texture_discs_from_views(sd / "optimised.ply", disc_means, tex_views)["textured"]
        if current.parent.name.startswith("step_"):
            shutil.rmtree(current.parent, ignore_errors=True)
        current = sd / "optimised.ply"
        for stale in sd.glob("*_patched.ply"):
            stale.unlink()
        steps.append(step)
        log(f"{k}/{len(order)} {v} (ref {near}): unknown {step['unknown_px']} px, reframe {moved:.3f}, "
            f"births {step['geometry'].get('born', 0)}, mae {step['after'].get(v, float('nan')):.3f}")
        if progress:
            progress(k, len(order) + 1)

    done = [hero] + [v for v, _d in order]
    masks = {u: (regions[u] if u in regions else holes[u]) for u in views}
    final = out_dir / "progressive_scene.ply"
    rep = multiview_optimize(source_ply=source_ply, init_ply=current, cameras=cameras, photos=photos, masks=masks,
                             chain=done, edits=edits, object_ids=object_ids, video=[], out_ply=final,
                             iterations=final_iterations, anchor=True, freeze_view_colour=True, device=device,
                             log=log, held_out=set(), preview_dir=out_dir / "previews", freeze_geometry=True,
                             train_kept=False, protect=protect)
    if current.parent.name.startswith("step_"):
        shutil.rmtree(current.parent, ignore_errors=True)
    if surface is not None:
        _finish_surface(final, surface=surface, disc_means=disc_means, disc_ramp=disc_ramp, top=top,
                        top_tiles=top_tiles, cameras=cameras, done=done, hero=hero, clean_big=clean_big,
                        region_at=lambda u: _resize(regions[u], size) if u in regions else hole_at(u, size),
                        size=size, device=device, feather_crossfade=feather_crossfade, log=log)
    if progress:
        progress(len(order) + 1, len(order) + 1)
    report = {"order": [hero] + [v for v, _d in order], "surface": surface, "steps": steps, "final": rep,
              "ply": str(final)}
    (out_dir / "progressive_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def cameras_at(cameras, size: int):
    """The cameras at ``size`` px: K's first two rows scaled (principal point
    included; pixel (0, 0) spans 0..1), width and height ``size``. For the
    camera pack, whose images are the photos' centre square downsampled."""
    import dataclasses
    from types import SimpleNamespace

    views = []
    for cam in cameras.views:
        K = np.asarray(cam.intrinsic, np.float64).copy()
        K[:2] *= size / cam.width
        views.append(dataclasses.replace(cam, intrinsic=K.astype(np.float32), width=size, height=size))
    return SimpleNamespace(views=views, by_id={v.view_id: v for v in views})


def independent_fill(*, source_ply: Path, start_ply: Path, cameras, hero: str, hero_clean: np.ndarray,
                     photos: dict, selections: dict, object_ids: np.ndarray, centre: np.ndarray,
                     clean: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                     clean_single: Callable[[np.ndarray, int], np.ndarray], out_dir: Path,
                     camera_image: Callable[[str], np.ndarray] | None = None, size: int = 1024,
                     seed: int = 241007, attempts: int = 3, max_reframe: float = 0.05,
                     final_iterations: int = 6000, top_texture: bool = True, restore_outside: bool = True,
                     disc_density: float = 2.0, feather_crossfade: bool = True, device: str = "cuda",
                     log: Callable = _LOG.info, progress: Callable | None = None, refit: bool = True) -> dict:
    """The all-around fill without the camera-by-camera loop: every camera
    that sees the hole is cleaned on its own, then one optimisation fits the
    fill to all of them.

    1. ``clean_single(photo, seed) -> rgb`` removes the object from each
       camera's own photo (``camera_image(v)``, uint8, any size; default the
       camera pack's image), with no reference. The result's tone is matched
       to the photo along the seam and it is blended into the photo with the
       feathered hole region, as the progressive fill does with its renders.
       The hero keeps ``hero_clean``.
    2. The support surface as in ``progressive_fill``, coloured from the
       hero, from a painted top view and its tiles (``clean(marked,
       reference, seed)``, the top view with the hero clean as reference).
       ``restore_outside``: the carve's rows past the discs are kept
       (``support_surface_fill(restore_outside=True)``).
    3. One optimisation of the fill against every clean camera. ``refit``
       (the default): at the cleans' ``size`` (the cameras scaled, the
       photos at that size, the masks grown so the optimiser's fixed 12 px
       dilation covers what it covers at the cameras' own size), the real
       Gaussians around the hole train their colour and opacity too, the
       discs move at 0.3 of the rate with their sizes, orientations and
       opacities anchored, and the finishing is skipped: it repaints the
       discs and resets their opacities, undoing the fit. ``refit=False``:
       the 23 October fill (the discs' colours and opacities only, at the
       cameras' size), then the progressive fill's finishing (top view and
       tiles, feather, tone).

    Kitchen, 24 October (scratch experiments): the frozen fit left V and arc
    seams against the kept surface (nothing reconciled the discs with the
    real Gaussians); training the kept ones with the discs' geometry free
    removed them but turned the fill into a grainy disc cloud; the refit's
    constraints keep the seams away without it, and fitting at 1024 rather
    than the camera pack's 512 sharpened it.

    Kitchen, 23 October (scratch experiment): about 6 min plus 75 s of Qwen
    against the progressive fill's 21 min, without the progressive result's
    ghost contours; the dark stain beside the fill was the carve, not the
    edits, and ``restore_outside`` removes it. Seeds as in
    ``progressive_fill``: camera k of the visit order
    starts at ``seed + 97 k``. Returns the report; the result is
    ``out_dir/independent_scene.ply``; per camera ``views/NN_<v>_input.png``
    (the photo), ``_qwen.png`` (the raw edit) and ``_clean.png`` (the
    target)."""
    import time

    from PIL import Image

    from .ply import GaussianPly

    t_all = time.time()
    timing = {}
    out_dir = Path(out_dir)
    (out_dir / "views").mkdir(parents=True, exist_ok=True)
    src = GaussianPly.load(source_ply)
    obj = src.means[object_ids]
    del src
    native = cameras.by_id[hero].width
    views = [v.view_id for v in cameras.views if not v.is_virtual and v.view_id in photos]
    hole_at = _hole_maker(cameras, obj, selections)
    holes = {v: hole_at(v, native) for v in views}
    seen = [v for v in views if holes[v].any()]
    order = visit_order(cameras, seen, hero, centre)
    done = [hero] + [v for v, _d in order]
    log(f"{len(done)} cameras see the hole; each cleaned on its own: " + " ".join(done))
    if camera_image is None:
        def camera_image(v):
            a = np.asarray(photos[v])
            return (a if a.max() > 1.5 else a * 255 + 0.5).clip(0, 255).astype(np.uint8)
    total = len(order) + 4

    hero_big = _resize(np.asarray(hero_clean, np.uint8), size)
    Image.fromarray(hero_big).save(out_dir / "views" / f"00_{hero}_clean.png")
    clean_big = {hero: hero_big}
    edits = {hero: _resize(hero_big, native).astype(np.float32) / 255}
    regions = {hero: holes[hero]}
    regions_big = {hero: _resize(holes[hero], size)}
    photos_big = {}
    steps = []
    for k, (v, _near) in enumerate(order, start=1):
        photo = np.asarray(camera_image(v), np.uint8)[..., :3]
        if photo.shape[:2] != (size, size):
            photo = _resize(photo, size)
        hole = hole_at(v, size)
        region = ndimage.binary_dilation(hole, iterations=40 * size // 1024)
        outside = ~ndimage.binary_dilation(region, iterations=10)
        best = None
        for attempt in range(attempts):
            q = clean_single(photo, seed + 97 * k + attempt)
            if q.shape[:2] != (size, size):
                q = _resize(q, size)
            diff = ndimage.median_filter(np.abs(q.astype(np.float32) - photo).mean(2) / 255, 5)
            moved = float((diff[outside] > 0.15).mean())
            if best is None or moved < best[1]:
                best = (q, moved)
            if moved < max_reframe:
                break
        raw, moved = best
        q, gain = match_seam_tone_field(raw, photo, region, band=12 * size // 1024)
        w = ndimage.gaussian_filter(region.astype(np.float32), 4 * size / 1024)[..., None]
        result = (q * w + photo * (1 - w) + 0.5).clip(0, 255).astype(np.uint8)
        for name, img in (("input", photo), ("qwen", raw), ("clean", result)):
            Image.fromarray(img).save(out_dir / "views" / f"{k:02d}_{v}_{name}.png")
        clean_big[v] = result
        edits[v] = _resize(result, native).astype(np.float32) / 255
        regions[v] = _resize(region, native)
        regions_big[v] = region
        photos_big[v] = photo
        steps.append({"view": v, "tone_gain": [round(float(x), 3) for x in gain], "reframe": round(moved, 4),
                      "attempts": attempt + 1})
        log(f"{k}/{len(order)} {v}: reframe {moved:.3f}, attempts {attempt + 1}")
        if progress:
            progress(k, total)
    timing["cleans"] = round(time.time() - t_all, 1)

    t = time.time()
    ss = _support_surface(start_ply=Path(start_ply), obj=obj, cameras=cameras, hero=hero, hero_big=hero_big,
                          clean=clean, seed=seed, size=size, device=device, out_dir=out_dir,
                          disc_density=disc_density, top_texture=top_texture, log=log,
                          restore_outside=restore_outside)
    surface = ss["surface"]
    timing["support_surface"] = round(time.time() - t, 1)
    if progress:
        progress(len(order) + 2, total)

    t = time.time()
    masks = {u: (regions[u] if u in regions else holes[u]) for u in views}
    final = out_dir / "independent_scene.ply"
    if refit:
        res = max(size, native)
        grow = int(round(12 * (res / native - 1)))
        fit_photos, fit_masks = {}, {}
        for u in views:
            if u == hero:
                p = np.asarray(camera_image(hero), np.uint8)[..., :3]
            elif u in photos_big:
                p = photos_big[u]
            else:
                a = np.asarray(photos[u])
                p = (a if a.max() > 1.5 else a * 255 + 0.5).clip(0, 255).astype(np.uint8)
            fit_photos[u] = _resize(p, res) if p.shape[:2] != (res, res) else p
            m = regions_big[u] if u in regions_big else holes[u]
            m = _resize(m, res) if m.shape != (res, res) else m
            fit_masks[u] = ndimage.binary_dilation(m, iterations=grow) if grow else m
        fit_edits = {u: (_resize(clean_big[u], res) if clean_big[u].shape[:2] != (res, res) else clean_big[u])
                     .astype(np.float32) / 255 for u in done}
        recipe = dict(freeze_geometry=False, train_kept=True, geo_lr_births=0.3, geo_lr_kept=0.0,
                      births_anchor=1e-2, protect_kept=True)
        rep = multiview_optimize(source_ply=source_ply, init_ply=ss["ply"], cameras=cameras_at(cameras, res),
                                 photos=fit_photos, masks=fit_masks, chain=done, edits=fit_edits,
                                 object_ids=object_ids, video=[], out_ply=final, iterations=final_iterations,
                                 anchor=True, freeze_view_colour=True, device=device, log=log, held_out=set(),
                                 preview_dir=out_dir / "previews", protect=ss["protect"], **recipe)
        rep["recipe"] = {**recipe, "target_size": res, "mask_grow_px": grow, "finishing": False}
        spacing = float(surface.get("disc_spacing") or 0)
        if spacing > 0:
            for part in rep.get("constraint_stats", {}).values():
                part["drift_spacings"] = {k: round(d / spacing, 3) for k, d in part.get("drift", {}).items()}
    else:
        rep = multiview_optimize(source_ply=source_ply, init_ply=ss["ply"], cameras=cameras, photos=photos,
                                 masks=masks, chain=done, edits=edits, object_ids=object_ids, video=[], out_ply=final,
                                 iterations=final_iterations, anchor=True, freeze_view_colour=True, device=device,
                                 log=log, held_out=set(), preview_dir=out_dir / "previews", freeze_geometry=True,
                                 train_kept=False, protect=ss["protect"])
    ss["ply"].unlink(missing_ok=True)
    timing["optimise"] = round(time.time() - t, 1)
    if progress:
        progress(len(order) + 3, total)

    t = time.time()
    if not refit:
        _finish_surface(final, surface=surface, disc_means=ss["disc_means"], disc_ramp=ss["disc_ramp"],
                        top=ss["top"], top_tiles=ss["top_tiles"], cameras=cameras, done=done, hero=hero,
                        clean_big=clean_big, region_at=lambda u: regions_big[u], size=size, device=device,
                        feather_crossfade=feather_crossfade, log=log)
    timing["finishing"] = round(time.time() - t, 1)
    timing["total"] = round(time.time() - t_all, 1)
    if progress:
        progress(total, total)
    report = {"method": "independent", "order": done, "surface": surface, "steps": steps, "final": rep,
              "timing_s": timing, "ply": str(final), "refit": rep.get("recipe"),
              "constraint_stats": rep.get("constraint_stats")}
    (out_dir / "independent_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
