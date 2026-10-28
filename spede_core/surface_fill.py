"""The surface a removed object stood on, rebuilt where nobody saw it.

An object standing on a table, floor or cloth hides the surface under it from
every camera. Guessing that surface per view (monocular depth births) gave
discs at different depths and angles from each camera: fuzz with gaps, and
the object's dark underside residue stayed visible through it. Here the
surface is fitted once, in 3D, to the real surface around the object's base:

1. The base: the lowest part of the removed object; its footprint on the
   ground plane (perpendicular to the cameras' mean up).
2. The surface: a quadratic height field fitted robustly to the kept
   Gaussians in a ring around the footprint, near the base's height.
3. Residue: kept Gaussians inside the footprint above the surface (the
   object's shadowed underside, stand remains) are removed.
4. Fill: a regular grid of flat discs on the fitted surface over the
   footprint, overlapping so no angle sees through, coloured with the ring's
   median colour; the optimisation recolours them from the clean images.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

_LOG = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814


def sh_rest_colour(rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """The colour the SH rest terms ``rest`` (K x 3, K = 3, 8 or 15, or one
    set per direction: N x K x 3) add for unit viewing directions ``dirs``
    (N x 3), as in 3DGS's eval_sh."""
    rest = np.asarray(rest, np.float64)
    K = rest.shape[-2]
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    basis = [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if K > 3:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        basis += [1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.31539156525252005 * (2 * zz - xx - yy),
                  -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy)]
        if K > 8:
            basis += [-0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
                      -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                      -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
                      -0.5900435899266435 * x * (xx - 3 * yy)]
    out = np.zeros((len(dirs), 3))
    for k, b in enumerate(basis[:K]):
        out += b * rest[..., k, :]
    return out


def _design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def fit_height_field(x: np.ndarray, y: np.ndarray, h: np.ndarray, iterations: int = 8) -> tuple[np.ndarray, float]:
    """Quadratic h(x, y) by iteratively reweighted least squares (Huber);
    returns the coefficients and the robust residual scale."""
    A = _design(x, y)
    w = np.ones(len(h))
    coef = np.zeros(6)
    for _ in range(iterations):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)
        r = h - A @ coef
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        k = 1.5 * scale
        # k/|r| is evaluated everywhere but only kept where |r| > k, so the
        # r == 0 division is discarded; silence its warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return coef, float(scale)


def surface_height(coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _design(x, y) @ coef


def surface_normal(coef: np.ndarray, x: np.ndarray, y: np.ndarray, axes) -> np.ndarray:
    up, e1, e2 = axes
    dhx = coef[1] + 2 * coef[3] * x + coef[4] * y
    dhy = coef[2] + coef[4] * x + 2 * coef[5] * y
    n = up[None] - dhx[:, None] * e1[None] - dhy[:, None] * e2[None]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def support_surface_fill(*, scene_ply: Path, object_means: np.ndarray, axes, out_ply: Path,
                         base_fraction: float = 0.15, margin: float = 0.04, ring: float = 0.35,
                         observed_ply: Path | None = None, feather: float = 0.05, density: float = 1.0,
                         feather_jitter: float = 0.6, restore_outside: bool = False, restore_margin: float = 1.5,
                         restore_height: float = 8.5, log=_LOG.info) -> dict:
    """``scene_ply``: the scene with the object removed; ``object_means``: the
    removed object's Gaussian centres; ``axes``: (up, e1, e2). ``margin`` and
    ``ring`` are fractions of the footprint's size. ``observed_ply``: the
    scene before any births (the hero's depth patch guessed its Gaussians
    from one view): Gaussians not in it are no evidence of the surface, and
    inside the fill region they are replaced. ``feather``: the disc layer
    continues this far (a fraction of the footprint) over the real surface
    around, its opacity falling to zero, so the seam blends instead of
    showing as a line. ``feather_jitter``: the band's width varies slowly
    along the outline by up to this fraction, so the seam is not an even
    offset of the object's outline, which the eye picks out. ``density``: discs per surrounding Gaussian spacing
    (2: half the spacing, four times the discs). At the surface's own
    spacing (kitchen: 5 mm apart, 5 mm wide) a disc is one colour over about
    a centimetre and the mats' weave cannot show however sharp the image it
    is coloured from. ``restore_outside``: residue the discs do not cover
    (farther than ``restore_margin`` surface spacings from every disc, in
    the plane) and lower than ``restore_height`` spacings above the disc
    surface is kept, not carved."""
    from .patch import _quaternions_from_frames
    from .ply import GaussianPly, append_planar_vertices, remove_rows

    up, e1, e2 = (np.asarray(a, np.float64) for a in axes)
    O = np.asarray(object_means, np.float64)
    hO = O @ up
    base, top = np.percentile(hO, 1), np.percentile(hO, 99)
    H = top - base
    # Stray selected Gaussians below the base (floor speckle) stay out.
    low = O[(hO >= base) & (hO <= base + base_fraction * H)]
    lx, ly = low @ e1, low @ e2
    xr, yr = np.percentile(lx, [1, 99]), np.percentile(ly, [1, 99])
    core = (lx >= xr[0]) & (lx <= xr[1]) & (ly >= yr[0]) & (ly <= yr[1])
    lx, ly = lx[core], ly[core]
    if lx.size < 3:
        raise RuntimeError(f"The removed object has no footprint ({len(O)} Gaussians selected): "
                           "run Remove again so the selection is redone")
    size = float(max(np.ptp(lx), np.ptp(ly)))
    if size <= 0:
        raise RuntimeError("The removed object has no footprint")

    # Footprint raster (cell = 1/96 of the footprint size).
    cell = size / 96
    x0, y0 = lx.min() - (margin + ring + 0.05) * size, ly.min() - (margin + ring + 0.05) * size
    nx = int(np.ceil((np.ptp(lx) + 2 * (margin + ring + 0.05) * size) / cell)) + 1
    ny = int(np.ceil((np.ptp(ly) + 2 * (margin + ring + 0.05) * size) / cell)) + 1
    foot = np.zeros((ny, nx), bool)
    foot[((ly - y0) / cell).astype(int), ((lx - x0) / cell).astype(int)] = True
    # Close the base's point raster at its own point spacing.
    gap = int(np.ceil(2 * np.sqrt(max(np.ptp(lx) * np.ptp(ly), 1e-12) / max(len(lx), 1)) / cell))
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(foot, iterations=max(4, gap)))
    fill_mask = ndimage.binary_dilation(foot, iterations=max(1, int(margin * 96)))
    ring_mask = ndimage.binary_dilation(fill_mask, iterations=max(1, int(ring * 96))) & ~fill_mask

    g = GaussianPly.load(scene_ply)
    M = np.asarray(g.means, np.float64)
    if observed_ply is not None:
        seen_set = {r.tobytes() for r in np.ascontiguousarray(GaussianPly.load(observed_ply).means, np.float32)}
        guessed = np.array([r.tobytes() not in seen_set for r in np.ascontiguousarray(g.means, np.float32)])
    else:
        guessed = np.zeros(len(M), bool)
    gx, gy, gh = M @ e1, M @ e2, M @ up
    ix, iy = ((gx - x0) / cell).astype(int), ((gy - y0) / cell).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
    ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    opacity = 1 / (1 + np.exp(-np.asarray(g.opacity_logits, np.float64).reshape(-1)))
    in_ring = inside & ring_mask[iyc, ixc] & (np.abs(gh - base) < 0.12 * H) & (opacity > 0.5) & ~guessed
    if in_ring.sum() < 50:
        raise RuntimeError(f"Too few surface Gaussians around the object's base ({int(in_ring.sum())})")
    coef, scale = fit_height_field(gx[in_ring], gy[in_ring], gh[in_ring])
    resid = gh[in_ring] - surface_height(coef, gx[in_ring], gy[in_ring])
    inl = np.abs(resid) < 3 * scale
    ring_pts = M[in_ring][inl]
    spacing = float(np.median(cKDTree(ring_pts).query(ring_pts, k=2)[0][:, 1]))
    spacing = float(np.clip(spacing, size / 400, size / 40))

    # The final fill region: the whole object's footprint (the base alone
    # missed the middle under a pot standing on legs: a hole there), limited
    # to where the fitted surface really exists (the scene's Gaussians on it,
    # holes closed), so it does not run over the table's edge into the air.
    surf = surface_height(coef, gx, gy)
    ox, oy = O @ e1, O @ e2
    oi, oj = ((ox - x0) / cell).astype(int), ((oy - y0) / cell).astype(int)
    ok = (oi >= 0) & (oj >= 0) & (oi < nx) & (oj < ny)
    whole = np.zeros((ny, nx), bool)
    whole[oj[ok], oi[ok]] = True
    whole = ndimage.binary_fill_holes(ndimage.binary_closing(whole, iterations=3))
    # Under the object's base nothing real was seen: what lies there is residue.
    on = (inside & (np.abs(gh - surf) < max(3 * scale, 2 * spacing)) & (opacity > 0.5) & ~guessed
          & ~ndimage.binary_dilation(foot, iterations=2)[iyc, ixc])
    # The surface's own height per cell (mats, steps and edges included),
    # carried smoothly under the object from around it. One quadratic for
    # the whole area cut through overlapping mats (kitchen) and floated as a
    # sheet over the lower one.
    hs_sum = np.zeros((ny, nx)); hs_w = np.zeros((ny, nx))
    np.add.at(hs_sum, (iyc[on], ixc[on]), gh[on] * opacity[on])
    np.add.at(hs_w, (iyc[on], ixc[on]), opacity[on])
    from .progressive_fill import _membrane
    known_h = hs_w >= 0.5
    hmap = _membrane(np.where(known_h, hs_sum / np.maximum(hs_w, 1e-9), 0.0), known_h, iterations=800)
    hmap = ndimage.gaussian_filter(hmap, 1.0)
    surf = np.where(inside, hmap[iyc, ixc], surf)
    # Where the surface exists: the convex hull of the Gaussians lying on it
    # (a table top is convex; the overhang beyond its edge is not in it).
    from scipy.spatial import Delaunay
    pts2 = np.unique(np.c_[ixc[on], iyc[on]], axis=0)
    support = foot.copy()
    if len(pts2) >= 3:
        jj, ii = np.mgrid[0:ny, 0:nx]
        support |= (Delaunay(pts2).find_simplex(np.c_[ii.ravel(), jj.ravel()]) >= 0).reshape(ny, nx)
    # Inside that, only where the real surface is missing or thin (under the
    # pot); cloth the cameras saw under the canopy stays.
    count = np.zeros((ny, nx))
    np.add.at(count, (iyc[on], ixc[on]), 1)
    dens = ndimage.gaussian_filter(count, 2)
    ring_dens = dens[ring_mask & (count > 0)]
    thin = dens < 0.3 * (np.median(ring_dens) if len(ring_dens) else dens.max())
    thin = ndimage.binary_opening(thin, iterations=2) | foot
    grow = max(1, int(margin * 96))
    fill_mask = ndimage.binary_dilation((whole | foot) & thin, iterations=grow) & support & thin

    # Residue: kept Gaussians inside the fill region above the surface, in a
    # thin slab below it (the object's dark underside), and big ones below
    # whose extent reaches it.
    in_fill = inside & fill_mask[iyc, ixc]
    # Below the surface, large dark Gaussians of the stool's inside reached up
    # through the new layer (2.5 sigma above their centre) as dark pits.
    reach = 2.5 * np.exp(np.asarray(g.log_scales, np.float64)).max(1)
    residue = (in_fill & (gh + reach > surf - 2 * max(scale, spacing)) & (gh < top + 0.05 * H)) | (in_fill & guessed)
    # Within the whole object's footprint, anything above the surface at the
    # object's height is left over from it (bits the selection missed, the
    # hero's guessed patch around the base): bonsai showed orange twigs and
    # pale fringes there.
    in_whole = inside & ndimage.binary_dilation(whole, iterations=2)[iyc, ixc]
    residue |= in_whole & (gh > surf + 3 * spacing) & (gh < top + 0.05 * H)
    residue |= in_whole & guessed & (gh > surf - 3 * spacing) & (gh < top + 0.05 * H)

    # Disc grid over the fill region.
    disc_spacing = spacing / max(density, 1e-3)
    xs = np.arange(x0, x0 + nx * cell, disc_spacing)
    ys = np.arange(y0, y0 + ny * cell, disc_spacing)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    fcells = max(1, int(round(feather * 96)))
    outside_d = ndimage.distance_transform_edt(~fill_mask)
    # The band's width: smooth noise with features about twice its width.
    noise = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal(fill_mask.shape), 2.0 * fcells,
                                    mode="wrap")
    noise = np.clip(noise / max(2.0 * float(noise.std()), 1e-12), -1.0, 1.0)
    width = fcells * (1.0 + float(np.clip(feather_jitter, 0.0, 0.9)) * noise)
    band = (outside_d > 0) & (outside_d <= width) & support
    gi = np.clip(((Y - y0) / cell).astype(int), 0, ny - 1)
    gj = np.clip(((X - x0) / cell).astype(int), 0, nx - 1)
    keep = fill_mask[gi, gj] | band[gi, gj]
    X, Y = X[keep], Y[keep]
    cx = np.clip((X - x0) / cell - 0.5, 0, nx - 1.001)
    cy = np.clip((Y - y0) / cell - 0.5, 0, ny - 1.001)
    # Distance and width at each disc, not its cell: a cell is 2 cm in the
    # kitchen, and per-cell steps drew the noisy edge as a staircase. Discs
    # in a fill cell stay opaque (the real surface under them is carved).
    d_at = ndimage.map_coordinates(outside_d, [cy, cx], order=1)
    w_at = ndimage.map_coordinates(width, [cy, cx], order=1)
    ramp = np.where(fill_mask[gi, gj][keep], 1.0, np.clip(1.0 - d_at / (w_at + 1), 0.0, 1.0))
    Hs = ndimage.map_coordinates(hmap, [cy, cx], order=1)
    pts = X[:, None] * e1 + Y[:, None] * e2 + Hs[:, None] * up
    restored = 0
    if restore_outside and residue.any():
        # The carve reaches past the discs (the whole object's footprint,
        # grown): there it took the real surface's upper layer and left the
        # darker, more opaque one under it bare. Kitchen, 23 October: the dark
        # red stain on the orange mat beside the fill was such a patch (7 500
        # scanned rows, 0.006-0.036 above the surface). Rows whose centre is
        # farther than ``restore_margin`` from every disc (in the surface's
        # plane) and lower than ``restore_height`` above the disc surface
        # there are kept. Both are in units of the surface's own Gaussian
        # spacing, so they follow the scene's scale: kitchen (spacing 0.0053)
        # 0.008 and 0.045, the rows the top view showed outside the disc area
        # (3 px margin at 1024), 98 % the same; bonsai (0.0066) 770 rows,
        # coloured like the cloth around them.
        cand = np.flatnonzero(residue)
        dist, near = cKDTree(np.c_[X, Y]).query(np.c_[gx[cand], gy[cand]])
        above = gh[cand] - Hs[near]
        back = (dist > restore_margin * spacing) & (above < restore_height * spacing)
        residue[cand[back]] = False
        restored = int(back.sum())
    carved = Path(out_ply).with_name(Path(out_ply).stem + "_carved.ply")
    if residue.any():
        remove_rows(scene_ply, np.flatnonzero(residue), carved)
    else:
        import shutil
        shutil.copyfile(scene_ply, carved)
    gy_, gx_ = np.gradient(ndimage.gaussian_filter(hmap, 2.0), cell)
    dhx = ndimage.map_coordinates(gx_, [cy, cx], order=1)
    dhy = ndimage.map_coordinates(gy_, [cy, cx], order=1)
    n = up[None] - dhx[:, None] * e1[None] - dhy[:, None] * e2[None]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t1 = np.cross(n, e2[None].repeat(len(n), 0))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    colour = np.median(np.asarray(g.dc, np.float64).reshape(len(M), -1)[in_ring][inl][:, :3], axis=0) * SH_C0 + 0.5
    rgb = np.repeat(np.clip(colour, 0, 1)[None], len(pts), 0)
    radius = np.full((len(pts), 2), 0.9 * disc_spacing)
    append_planar_vertices(carved, Path(out_ply), pts, rgb, radius, opacity=0.95, thickness_fraction=0.12,
                           quaternions=_quaternions_from_frames(t1, t2, n))
    carved.unlink()
    # The discs take the surrounding surface's view-dependent terms: without
    # them their base colour learned the surface's darkened look from most
    # angles, and web viewers (base colour only) showed the fill darker than
    # the cloth around it (bonsai: 45 against 60-74).
    if g.rest.shape[1]:
        _set_disc_rest(Path(out_ply), len(pts), np.median(g.rest[in_ring][inl], axis=0))
    set_disc_feather(Path(out_ply), pts.astype(np.float32), ramp)
    report = {"base_height": float(base), "footprint_size": size, "ring_gaussians": int(in_ring.sum()),
              "surface_residual": scale, "spacing": spacing, "disc_spacing": disc_spacing, "carved": int(residue.sum()),
              "discs": int(len(pts)), "feathered": int((ramp < 1).sum()),
              "feather_width_cells": [round(float(width[band].min()), 2) if band.any() else 0.0,
                                      round(float(width[band].max()), 2) if band.any() else 0.0],
              "colour": [float(c) for c in np.clip(colour, 0, 1)]}
    if restore_outside:
        report["restored_outside"] = restored
    log(f"support surface: ring {report['ring_gaussians']} Gaussians, residual {scale:.4f}, "
        f"spacing {spacing:.4f}, carved {report['carved']}, {len(pts)} discs")
    return report


def set_disc_feather(ply_path: Path, disc_means: np.ndarray, ramp: np.ndarray, opacity: float = 0.95) -> int:
    """Discs (found by their centres) get opacity ``opacity * ramp``; returns
    how many were found. Used after the optimisation too, which trains
    opacity and would otherwise close the feather."""
    from .ply import GaussianPly, write_vertices

    g = GaussianPly.load(ply_path)
    index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(g.means, np.float32))}
    rows = np.array([index.get(r.tobytes(), -1) for r in np.ascontiguousarray(disc_means, np.float32)])
    ok = rows >= 0
    a = np.clip(opacity * np.asarray(ramp, np.float64)[ok], 1e-4, 0.9999)
    v = g.vertex.copy()
    v["opacity"][rows[ok]] = np.log(a / (1 - a)).astype(np.float32)
    tmp = Path(ply_path).with_name(Path(ply_path).stem + "_feather.ply")
    write_vertices(g, tmp, v, "SpEdE: support surface feathered over the real surface")
    tmp.replace(ply_path)
    return int(ok.sum())


def crossfade_feather_to_surface(ply_path: Path, disc_means: np.ndarray, ramp: np.ndarray, surface_view: dict) -> dict:
    """The feather band's discs fade towards the real surface's own colour as
    well as in opacity: base colour ``ramp * own + (1 - ramp) * surface``,
    the surface colour taken from ``surface_view`` (a render of the scene
    without the discs, a ``texture_discs_from_views`` view dict). With the
    opacity alone, the band showed the fill's weave half-transparent over a
    real weave that did not line up: a ghosted double pattern ringing the
    seam."""
    ramp = np.clip(np.asarray(ramp, np.float64), 0.0, 1.0)
    sel = np.flatnonzero(ramp < 0.999)
    if not len(sel):
        return {"crossfaded": 0}
    rep = blend_colours_from_view(ply_path, np.asarray(disc_means)[sel], ramp[sel], surface_view)
    return {"crossfaded": rep["blended"], "seen": rep["seen"]}


def blend_colours_from_view(ply_path: Path, means: np.ndarray, own: np.ndarray, view: dict) -> dict:
    """Gaussians (found by their centres) get base colour ``own * theirs +
    (1 - own) * the view's``, the view's colour as ``texture_discs_from_views``
    reads it (footprint-averaged, view-dependent terms subtracted, occluded
    ones left as they are)."""
    from .ply import GaussianPly, write_vertices

    own = np.clip(np.asarray(own, np.float64), 0.0, 1.0)
    means = np.ascontiguousarray(np.asarray(means, np.float32))
    g = GaussianPly.load(ply_path)
    index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(g.means, np.float32))}
    rows = np.array([index.get(r.tobytes(), -1) for r in means])
    ok = rows >= 0
    before = np.asarray(g.dc, np.float64)[rows[ok], 0].copy()
    del g, index
    rep = texture_discs_from_views(ply_path, means[ok], [view])
    g = GaussianPly.load(ply_path)
    viewed = np.asarray(g.dc, np.float64)[rows[ok], 0]
    w = own[ok, None]
    v = g.vertex.copy()
    mixed = w * before + (1 - w) * viewed
    for c in range(3):
        v[f"f_dc_{c}"][rows[ok]] = mixed[:, c].astype(np.float32)
    tmp = Path(ply_path).with_name(Path(ply_path).stem + "_blend.ply")
    write_vertices(g, tmp, v, "SpEdE: colours blended towards a view")
    tmp.replace(ply_path)
    return {"blended": int(ok.sum()), "seen": rep.get("textured", 0)}


def _set_disc_rest(ply_path: Path, count: int, rest: np.ndarray) -> None:
    """The last ``count`` Gaussians get the SH rest terms ``rest`` (K x 3)."""
    from .ply import GaussianPly, _sh_names, write_vertices

    g = GaussianPly.load(ply_path)
    v = g.vertex.copy()
    names = _sh_names(tuple(v.dtype.names))
    K = len(names) // 3
    for c in range(3):
        for k in range(K):
            v[names[c * K + k]][len(v) - count:] = rest[k, c]
    tmp = Path(ply_path).with_name(Path(ply_path).stem + "_rest.ply")
    write_vertices(g, tmp, v, "SpEdE: support surface discs take the surrounding view-dependent terms")
    tmp.replace(ply_path)


def colour_discs_from_view(ply_path: Path, count: int, image: np.ndarray, viewmat, intrinsic) -> int:
    """The last ``count`` Gaussians (the discs) take their base colour from a
    clean image of a camera that sees them (``intrinsic`` at the image's
    size); discs outside it take the median of the others. The median DC of
    the surrounding Gaussians is darker than the surface renders (layers and
    view-dependent terms add up), and the per-view cleaning then kept that
    darkness. Returns how many discs were sampled."""
    from .geometry_utils import project_to_pixels
    from .ply import GaussianPly, write_vertices

    g = GaussianPly.load(ply_path)
    v = g.vertex.copy()
    rows = np.arange(len(v) - count, len(v))
    uv, z = project_to_pixels(g.means[rows], np.asarray(viewmat, np.float32), np.asarray(intrinsic, np.float32))
    h, w = image.shape[:2]
    ok = np.isfinite(uv).all(1) & (z > 0)
    x = np.where(ok, uv[:, 0], -1).astype(int)
    y = np.where(ok, uv[:, 1], -1).astype(int)
    ok &= (x >= 0) & (y >= 0) & (x < w) & (y < h)
    rgb = np.zeros((count, 3), np.float64)
    rgb[ok] = np.asarray(image, np.float64)[y[ok], x[ok], :3] / 255.0
    if ok.any():
        rgb[~ok] = np.median(rgb[ok], axis=0)
        # The image shows base colour plus the discs' view-dependent terms
        # from this camera: the base colour is what is left.
        if g.rest.shape[1]:
            centre = np.linalg.inv(np.asarray(viewmat, np.float64))[:3, 3]
            d = g.means[rows].astype(np.float64) - centre
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            rgb = rgb - np.stack([sh_rest_colour(g.rest[r], d[i:i + 1])[0] for i, r in enumerate(rows)])
        for i in range(3):
            v[f"f_dc_{i}"][rows] = ((rgb[:, i] - 0.5) / SH_C0).astype(np.float32)
        tmp = Path(ply_path).with_name(Path(ply_path).stem + "_coloured.ply")
        write_vertices(g, tmp, v, "SpEdE: support surface discs coloured from a clean view")
        tmp.replace(ply_path)
    return int(ok.sum())


def harmonise_disc_colours(ply_path: Path, disc_means: np.ndarray, axes, *, cell_factor: float = 4.0,
                           band: int = 4, out_ply: Path | None = None) -> dict:
    """Make the discs' base colour follow the real surface around them.

    Web viewers draw base colours only. The real cloth's base colour varies
    over the table (bright where the light falls, darker at the back); the
    discs, trained against the cleaned images through the view-dependent
    terms, came out about equally bright all over. Here, in the surface's
    plane, the surrounding real Gaussians' base colour is interpolated
    smoothly over the disc area (a membrane from the seam inwards) and the
    discs' own low-frequency base colour is replaced by it; their texture
    (the high frequencies) stays."""
    from scipy.ndimage import gaussian_filter

    from .progressive_fill import _membrane
    from .ply import GaussianPly, write_vertices

    up, e1, e2 = (np.asarray(a, np.float64) for a in axes)
    g = GaussianPly.load(ply_path)
    M = np.asarray(g.means, np.float64)
    key = {r.tobytes() for r in np.ascontiguousarray(disc_means, np.float32)}
    is_disc = np.array([r.tobytes() in key for r in np.ascontiguousarray(g.means, np.float32)])
    if is_disc.sum() < 100:
        return {"harmonised": 0}
    D = M[is_disc]
    spacing = float(np.median(cKDTree(D).query(D, k=2)[0][:, 1]))
    cell = cell_factor * spacing
    x, y, h = M @ e1, M @ e2, M @ up
    x0, y0 = x[is_disc].min() - (band + 2) * cell, y[is_disc].min() - (band + 2) * cell
    nx = int((x[is_disc].max() - x0) / cell) + band + 3
    ny = int((y[is_disc].max() - y0) / cell) + band + 3
    ix, iy = ((x - x0) / cell).astype(int), ((y - y0) / cell).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
    disc_cells = np.zeros((ny, nx), bool)
    disc_cells[iy[is_disc], ix[is_disc]] = True
    disc_cells = ndimage.binary_closing(disc_cells, iterations=1)
    ring_cells = ndimage.binary_dilation(disc_cells, iterations=band) & ~disc_cells
    # Height of the disc layer per cell, extended into the ring.
    hsum = np.zeros((ny, nx)); hcnt = np.zeros((ny, nx))
    np.add.at(hsum, (iy[is_disc], ix[is_disc]), h[is_disc]); np.add.at(hcnt, (iy[is_disc], ix[is_disc]), 1)
    hmap = _membrane(np.where(hcnt > 0, hsum / np.maximum(hcnt, 1), 0.0), hcnt > 0, iterations=200)
    opacity = 1 / (1 + np.exp(-np.asarray(g.opacity_logits, np.float64).reshape(-1)))
    rgb = np.asarray(g.dc, np.float64)[:, 0] * SH_C0 + 0.5
    ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    real = (~is_disc & inside & ring_cells[iyc, ixc] & (opacity > 0.5)
            & (np.abs(h - hmap[iyc, ixc]) < 3 * cell))
    if real.sum() < 50:
        return {"harmonised": 0}
    wsum = np.zeros((ny, nx))
    np.add.at(wsum, (iy[real], ix[real]), opacity[real])
    out_rgb = rgb.copy()
    stats = {}
    for c in range(3):
        csum = np.zeros((ny, nx))
        np.add.at(csum, (iy[real], ix[real]), opacity[real] * rgb[real, c])
        known = wsum > 0.5
        ring_field = _membrane(np.where(known, csum / np.maximum(wsum, 1e-9), 0.0), known, iterations=800)
        dsum = np.zeros((ny, nx)); dcnt = np.zeros((ny, nx))
        np.add.at(dsum, (iy[is_disc], ix[is_disc]), rgb[is_disc, c]); np.add.at(dcnt, (iy[is_disc], ix[is_disc]), 1)
        num = gaussian_filter(dsum, 2); den = gaussian_filter(dcnt, 2)
        disc_low = num / np.maximum(den, 1e-9)
        low = gaussian_filter(ring_field, 1)[iy[is_disc], ix[is_disc]]
        detail = rgb[is_disc, c] - disc_low[iy[is_disc], ix[is_disc]]
        # Texture strength like the real surface around: the cleaned images'
        # crisp knit gave the discs bright speckles the reconstructed cloth
        # does not have (a grey speckled patch in the purple).
        ring_detail = rgb[real, c] - (csum / np.maximum(wsum, 1e-9))[iy[real], ix[real]]
        amp = float(np.clip(np.std(ring_detail) / max(np.std(detail), 1e-6), 0.2, 1.0))
        out_rgb[is_disc, c] = low + amp * detail
        stats[c] = float(np.median(low - disc_low[iy[is_disc], ix[is_disc]]))
        stats[f"texture_{c}"] = amp
    v = g.vertex.copy()
    for c in range(3):
        v[f"f_dc_{c}"][is_disc] = ((np.clip(out_rgb[is_disc, c], 0, 1) - 0.5) / SH_C0).astype(np.float32)
    target = Path(out_ply) if out_ply else Path(ply_path)
    tmp = target.with_name(target.stem + "_harmonised.ply")
    write_vertices(g, tmp, v, "SpEdE: disc base colours follow the surrounding surface")
    tmp.replace(target)
    return {"harmonised": int(is_disc.sum()), "ring_gaussians": int(real.sum()),
            "median_correction_rgb": [round(stats[c], 4) for c in range(3)],
            "texture_scale_rgb": [round(stats[f"texture_{c}"], 3) for c in range(3)]}


def _disc_normals(quaternions: np.ndarray, log_scales: np.ndarray) -> np.ndarray:
    """Each Gaussian's thinnest axis (the disc's normal), unit length."""
    q = np.asarray(quaternions, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    return R[np.arange(len(R)), :, np.argmin(np.asarray(log_scales), 1)]


def texture_discs_from_views(ply_path: Path, disc_means: np.ndarray, views: list[dict], *,
                             sharpness: float = 8.0, depth_tolerance: float = 0.03) -> dict:
    """Disc base colours from the clean image that sees each disc best.

    Optimised against some twenty cleaned images that disagree in the
    detail (each view's editor drew its own stripes and knit), the discs
    learned their average: a blur. Here each disc takes its colour from the
    views that see it best (squarely and close: facing x pixels per unit),
    weighted ``(score / best) ** sharpness`` so neighbouring discs mostly
    agree on one view and switch smoothly. ``views``: dicts with ``image``
    (uint8 HxWx3), ``viewmat``, ``intrinsic`` (at the image's size),
    ``depth`` (the scene's rendered depth, HxW), ``region`` (bool HxW: where
    the image is a clean edit) and optional ``bonus`` (the hero's > 1).
    Returns counts per view."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    from .geometry_utils import project_to_pixels
    from .ply import GaussianPly, write_vertices

    g = GaussianPly.load(ply_path)
    index = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(g.means, np.float32))}
    rows = np.array([index.get(r.tobytes(), -1) for r in np.ascontiguousarray(disc_means, np.float32)])
    rows = rows[rows >= 0]
    if not len(rows):
        return {"textured": 0}
    M = np.asarray(g.means, np.float64)[rows]
    normals = _disc_normals(g.quaternions[rows], g.log_scales[rows])
    radius = np.exp(np.asarray(g.log_scales, np.float64)[rows]).max(1)
    n = len(rows)
    scores = np.zeros((len(views), n))
    colours = np.zeros((len(views), n, 3))
    for k, view in enumerate(views):
        vm = np.asarray(view["viewmat"], np.float64)
        K = np.asarray(view["intrinsic"], np.float64)
        img = np.asarray(view["image"], np.float64)[..., :3] / 255.0
        h, w = img.shape[:2]
        uv, z = project_to_pixels(M, vm, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        u = np.where(ok, uv[:, 0], -1.0)
        v = np.where(ok, uv[:, 1], -1.0)
        ok &= (u >= 2) & (v >= 2) & (u < w - 2) & (v < h - 2)
        xi, yi = u.astype(int).clip(0, w - 1), v.astype(int).clip(0, h - 1)
        ok &= np.asarray(view["region"], bool)[yi, xi]
        depth = np.asarray(view["depth"], np.float64)[yi, xi]
        ok &= ~(z > depth * (1 + depth_tolerance) + 1e-6)
        centre = np.linalg.inv(vm)[:3, 3]
        to_cam = centre - M
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.abs(np.sum(normals * to_cam, 1))
        pixels_per_unit = K[0, 0] / np.maximum(z, 1e-6)
        scores[k] = np.where(ok, facing * pixels_per_unit * float(view.get("bonus", 1.0)), 0.0)
        # Average over each disc's footprint rather than one pixel.
        footprint = float(np.median((radius * pixels_per_unit)[ok])) if ok.any() else 1.0
        sigma = max(0.0, 0.5 * footprint - 0.5)
        blurred = np.stack([gaussian_filter(img[..., c], sigma) if sigma else img[..., c] for c in range(3)], -1)
        for c in range(3):
            colours[k, :, c] = map_coordinates(blurred[..., c], [v - 0.5, u - 0.5], order=1, mode="nearest")
    best = scores.max(0)
    seen = best > 0
    weights = np.where(seen, (scores / np.maximum(best, 1e-12)) ** sharpness, 0.0)
    weights /= np.maximum(weights.sum(0), 1e-12)
    rgb = np.einsum("kn,knc->nc", weights, colours)
    if g.rest.shape[1]:
        # Each view showed base colour plus the disc's view-dependent terms
        # from that camera; subtract them weighted the same way.
        for k, view in enumerate(views):
            centre = np.linalg.inv(np.asarray(view["viewmat"], np.float64))[:3, 3]
            d = M - centre
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            sel = weights[k] > 1e-4
            if sel.any():
                rgb[sel] -= weights[k, sel, None] * sh_rest_colour(g.rest[rows[sel]], d[sel])
    v = g.vertex.copy()
    for c in range(3):
        v[f"f_dc_{c}"][rows[seen]] = ((np.clip(rgb[seen, c], 0, 1) - 0.5) / SH_C0).astype(np.float32)
    tmp = Path(ply_path).with_name(Path(ply_path).stem + "_textured.ply")
    write_vertices(g, tmp, v, "SpEdE: support surface discs textured from their best clean views")
    tmp.replace(ply_path)
    lead = np.argmax(scores, 0)
    return {"textured": int(seen.sum()), "discs": int(n),
            "lead_view_counts": {str(views[k].get("name", k)): int(((lead == k) & seen).sum())
                                 for k in range(len(views)) if ((lead == k) & seen).any()}}
