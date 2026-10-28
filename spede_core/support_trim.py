# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trim the support surface an extracted object was selected with.

The selection is shared with REMOVE, where a little of the table under the
object is welcome: its contact shadow goes with the object. An extracted
object must not carry it. Kitchen (26 Oct): the lego bulldozer came out with
a thin fringe of the pink/grey table mat under the bucket and tracks, already
in the semantic selection (the masks' 8 px dilation and the hull reach the
mat at the contact line).

The support is a plane under the object: up from the capture cameras, its
height and tilt fitted to the unselected Gaussians in a ring around the
object's footprint. A selected Gaussian is dropped only when it lies within a
thin band above that plane (or below it) *and* its colour is nearer the
ring's colours than the object's own. Track and bucket bottoms touch the
plane too; their colour keeps them.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .ply import SH_C0


def _axes(world_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(world_up, np.float64)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return up, e1, np.cross(up, e1)


def _robust_plane(uv: np.ndarray, height: np.ndarray, *, iterations: int = 8,
                  inlier_mads: float = 2.5) -> tuple[np.ndarray, float, np.ndarray]:
    """height ~ a*u + b*v + c by least squares on the inliers (MAD rejection).

    Returns (a, b, c), the inliers' robust spread (1.4826 MAD) and the
    inlier mask. The ring holds the mat's edges and things standing on it;
    the rejection keeps the plane on the surface itself.
    """
    design = np.c_[uv, np.ones(len(uv))]
    inlier = np.ones(len(uv), bool)
    spread = 0.0
    for _ in range(iterations):
        coef = np.linalg.lstsq(design[inlier], height[inlier], rcond=None)[0]
        residual = height - design @ coef
        centre = np.median(residual[inlier])
        spread = float(1.4826 * np.median(np.abs(residual[inlier] - centre)))
        updated = np.abs(residual - centre) <= inlier_mads * max(spread, 1e-9)
        if updated.sum() < 3 or np.array_equal(updated, inlier):
            break
        inlier = updated
    return coef, spread, inlier


def _local_colour_distance(colours: np.ndarray, positions: np.ndarray,
                           palette_colours: np.ndarray, palette_positions: np.ndarray,
                           *, neighbors: int, best: int) -> np.ndarray:
    """How well a colour matches the palette *near it*: the mean colour
    distance to the ``best`` closest colours among its ``neighbors``
    nearest palette Gaussians in space.

    Local, because a global palette always has a near match somewhere: the
    bulldozer's shaded interior and the mat's warm bounce light both hold
    the peach of the fringe, while next to the bucket the object is yellow.
    """
    neighbors = int(min(neighbors, len(palette_colours)))
    best = int(min(best, neighbors))
    _, index = cKDTree(palette_positions).query(positions, k=neighbors)
    index = np.asarray(index).reshape(len(positions), neighbors)
    distance = np.linalg.norm(palette_colours[index] - colours[:, None, :], axis=2)
    return np.sort(distance, axis=1)[:, :best].mean(axis=1)


def support_contact_trim(
    means: np.ndarray,
    log_scales: np.ndarray,
    dc: np.ndarray,
    opacity_logits: np.ndarray,
    object_ids: np.ndarray,
    world_up: np.ndarray,
    *,
    ring_inner_fraction: float = 0.02,
    ring_outer_fraction: float = 0.15,
    ring_height_fraction: float = 0.3,
    ring_minimum_opacity: float = 0.3,
    minimum_ring_gaussians: int = 200,
    band_mads: float = 3.0,
    band_scales: float = 3.0,
    maximum_band_fraction: float = 0.04,
    object_palette_bands: float = 3.0,
    colour_neighbors: int = 32,
    colour_best: int = 4,
    colour_ratio: float = 1.0,
    upper_band_factor: float = 2.0,
    upper_colour_ratio: float = 0.5,
    support_colour_spread: float = 4.0,
    hard_cut_bands: float = 2.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Selected rows to keep, and a report of the plane, band and counts.

    ``means``/``log_scales``/``dc``/``opacity_logits`` are the whole source
    scene's (``dc`` as SH degree 0 coefficients, (N, 3) or (N, 1, 3)).
    Lengths given as fractions are of the object's footprint diagonal (ring)
    or height (band cap). The band is ``band_mads`` times the ring plane's
    spread or ``band_scales`` times its Gaussians' median largest scale,
    whichever is larger, capped at ``maximum_band_fraction`` of the object's
    height. A Gaussian in the band is support when its colour distance to the
    support near it (``_local_colour_distance`` over the plane's Gaussians,
    by footprint position) is under ``colour_ratio`` times that to the object
    near it (over the object above ``object_palette_bands`` bands, by 3-D
    position); up to ``upper_band_factor`` bands, under
    ``upper_colour_ratio`` times. Nothing is trimmed without a ring of at
    least ``minimum_ring_gaussians`` (an object held in the air, or on a
    surface the capture never saw around it).
    """
    means = np.asarray(means, np.float64)
    ids = np.unique(np.asarray(object_ids, np.int64))
    report: dict[str, object] = {"enabled": True, "input_gaussians": int(len(ids)),
                                 "removed_gaussians": 0, "applied": False}
    if len(ids) < 10:
        report["reason"] = "too few selected Gaussians"
        return ids, report
    up, e1, e2 = _axes(world_up)
    height = means @ up
    uv = np.stack([means @ e1, means @ e2], axis=1)
    object_height = height[ids]
    low, high = np.percentile(object_height, [1.0, 99.0])
    span = float(high - low)
    footprint = uv[ids]
    diagonal = float(np.linalg.norm(np.ptp(footprint, axis=0)))
    if span <= 0 or diagonal <= 0:
        report["reason"] = "degenerate object extent"
        return ids, report

    # The ring: unselected, opaque Gaussians just outside the footprint, near
    # the object's base height (not the wall behind it or the shelf above).
    others = np.ones(len(means), bool)
    others[ids] = False
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logits, np.float64).reshape(-1)))
    near = np.flatnonzero(
        others
        & (np.abs(height - low) <= float(ring_height_fraction) * span)
        & (opacity >= float(ring_minimum_opacity))
    )
    gap, _ = cKDTree(footprint).query(uv[near])
    ring = near[(gap > ring_inner_fraction * diagonal) & (gap <= ring_outer_fraction * diagonal)]
    report["ring_gaussians"] = int(len(ring))
    if len(ring) < int(minimum_ring_gaussians):
        report["reason"] = f"only {len(ring)} support Gaussians around the footprint"
        return ids, report

    coef, spread, inlier = _robust_plane(uv[ring], height[ring])
    surface = ring[inlier]
    above = height - (uv @ coef[:2] + coef[2])
    scales = np.exp(np.asarray(log_scales, np.float64)[surface]).max(axis=1)
    band = max(float(band_mads) * spread, float(band_scales) * float(np.median(scales)))
    band = min(band, float(maximum_band_fraction) * span)

    colours = np.clip(0.5 + SH_C0 * np.asarray(dc, np.float64).reshape(len(means), 3), 0.0, 1.0)
    object_above = above[ids]
    candidates = ids[object_above <= float(upper_band_factor) * band]
    trusted = ids[object_above > float(object_palette_bands) * band]
    report.update({
        "world_up": [float(v) for v in up],
        "plane_height": float(coef[2]),
        "plane_height_at_object": float(np.median(uv[ids] @ coef[:2] + coef[2])),
        "plane_tilt_deg": float(np.degrees(np.arctan(np.linalg.norm(coef[:2])))),
        "plane_spread": float(spread),
        "plane_inliers": int(len(surface)),
        "band": float(band),
        "object_height": span,
        "candidate_gaussians": int(len(candidates)),
        "object_palette_gaussians": int(len(trusted)),
    })
    if len(candidates) == 0 or len(trusted) < 10:
        report["reason"] = "nothing selected at the support plane" if len(candidates) == 0 \
            else "no object above the support plane to compare colours with"
        return ids, report

    # Support colours by footprint position (the mat under the bucket, not
    # the wood beside it); object colours by 3-D position.
    to_support = _local_colour_distance(
        colours[candidates], uv[candidates], colours[surface], uv[surface],
        neighbors=colour_neighbors, best=colour_best,
    )
    to_object = _local_colour_distance(
        colours[candidates], means[candidates], colours[trusted], means[trusted],
        neighbors=colour_neighbors, best=colour_best,
    )
    # On the plane, nearer the support's colour than the object's is enough;
    # up to ``upper_band_factor`` bands the colour must be clearly the
    # support's (kitchen: peach mat splats a centimetre up round the bucket
    # teeth, among the teeth's own yellow ones).
    in_band = above[candidates] <= band
    ratio = np.where(in_band, float(colour_ratio), float(upper_colour_ratio))
    # Nearer the support than the object is not enough on its own: a part
    # that exists only at the plane (dark track links under a yellow body)
    # has no colour of its own above it to be compared with. The colour must
    # also be one the support itself shows, within ``support_colour_spread``
    # times the support's own 90th-percentile scatter.
    sample = surface
    if len(sample) > 20000:
        sample = np.random.default_rng(0).choice(sample, 20000, replace=False)
    scatter = _local_colour_distance(
        colours[sample], uv[sample], colours[surface], uv[surface],
        neighbors=colour_neighbors, best=colour_best + 1,
    ) * (colour_best + 1) / max(colour_best, 1)
    tolerance = float(support_colour_spread) * float(np.percentile(scatter, 90.0))
    support = (to_support < ratio * to_object) & (to_support <= tolerance)
    # Within ``hard_cut_bands`` bands of the plane everything goes, whatever
    # its colour (mat in the tracks' shadow is as dark as the tracks).
    support |= above[candidates] <= float(hard_cut_bands) * band
    removed = candidates[support]
    kept = np.setdiff1d(ids, removed)
    report.update({
        "applied": True,
        "removed_gaussians": int(len(removed)),
        "kept_gaussians": int(len(kept)),
        "removed_fraction": float(len(removed) / max(len(ids), 1)),
        "support_colour_tolerance": tolerance,
        "rejected_as_not_support_colour": int(((to_support < ratio * to_object) & (to_support > tolerance)).sum()),
        "parameters": {
            "ring_inner_fraction": float(ring_inner_fraction),
            "ring_outer_fraction": float(ring_outer_fraction),
            "ring_height_fraction": float(ring_height_fraction),
            "band_mads": float(band_mads),
            "band_scales": float(band_scales),
            "maximum_band_fraction": float(maximum_band_fraction),
            "object_palette_bands": float(object_palette_bands),
            "colour_neighbors": int(colour_neighbors),
            "colour_best": int(colour_best),
            "colour_ratio": float(colour_ratio),
            "upper_band_factor": float(upper_band_factor),
            "upper_colour_ratio": float(upper_colour_ratio),
            "support_colour_spread": float(support_colour_spread),
            "hard_cut_bands": float(hard_cut_bands),
        },
        "removed_source_ids_sample": [int(v) for v in removed[:200]],
    })
    return kept, report
