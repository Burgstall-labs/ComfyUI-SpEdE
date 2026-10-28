# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Recover thin/open object surfaces that semantic masks miss.

The semantic multiview hull is the trusted core.  Surface rescue considers
nearby source Gaussians that were *not* selected semantically and accepts only
those that:
  1. fit a locally planar manifold of trusted object Gaussians in world space;
  2. receive repeated, angularly diverse capture-view evidence at the source
     model's visible depth layer (or actually contribute to the rendered pixel);
  3. project into the object's filled/dilated silhouette envelope in those
     views, without too many visible contradictions.

This is aimed at spokes, grilles, thin tubes and other open structures where
background is visible through the object.  It is intentionally conservative:
semantic selection remains authority and rescue only adds source Gaussians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .depth_clean import select_diverse_views
from .geometry_utils import project_to_pixels
from .roi import dilate_mask
from .visual_hull import scaled_dilation


@dataclass(frozen=True)
class SurfaceRescueEvidence:
    candidate_ids: np.ndarray
    local_plane_distance: np.ndarray
    local_plane_tolerance: np.ndarray
    planarity_ratio: np.ndarray
    local_manifold_rank: np.ndarray
    envelope_support_views: np.ndarray
    raw_semantic_support_views: np.ndarray
    contradiction_views: np.ndarray
    visible_views: np.ndarray
    rescue: np.ndarray

    @property
    def rescued_ids(self) -> np.ndarray:
        return self.candidate_ids[self.rescue]


def surface_rescue_decision(
    envelope_support_views: np.ndarray,
    contradiction_views: np.ndarray,
    *,
    minimum_support_views: int = 2,
    maximum_contradiction_views: int = 0,
) -> np.ndarray:
    """Final rescue vote after world-space local-surface filtering."""
    support = np.asarray(envelope_support_views, np.int32)
    contradiction = np.asarray(contradiction_views, np.int32)
    if support.shape != contradiction.shape:
        raise ValueError("support and contradiction arrays must align")
    if minimum_support_views < 1:
        raise ValueError("minimum_support_views must be positive")
    if maximum_contradiction_views < 0:
        raise ValueError("maximum_contradiction_views must be non-negative")
    return (
        (support >= int(minimum_support_views))
        & (contradiction <= int(maximum_contradiction_views))
    )


def object_envelope(mask: np.ndarray, dilation_px_512: int = 3) -> np.ndarray:
    """Fill enclosed semantic holes, then add a small resolution-scaled band."""
    mask = np.asarray(mask, bool)
    filled = ndimage.binary_fill_holes(mask)
    return dilate_mask(filled, scaled_dilation(int(dilation_px_512), mask.shape))


def _visible_mass_for_candidates(contribution, local_index: np.ndarray, count: int) -> np.ndarray:
    ids = np.asarray(contribution.gaussian_ids, np.int64).reshape(-1)
    weights = np.asarray(contribution.weights, np.float64).reshape(-1)
    valid = (ids >= 0) & (ids < len(local_index)) & (weights > 0)
    if not bool(valid.any()):
        return np.zeros(count, np.float64)
    local = local_index[ids[valid]]
    selected = local >= 0
    if not bool(selected.any()):
        return np.zeros(count, np.float64)
    return np.bincount(
        local[selected],
        weights=weights[valid][selected],
        minlength=count,
    ).astype(np.float64)


def _local_surface_candidates(
    scene,
    semantic_ids: np.ndarray,
    *,
    neighbor_count: int,
    minimum_neighbors: int,
    neighbor_radius_fraction: float,
    plane_scale_tolerance: float,
    plane_span_tolerance: float,
    maximum_planarity_ratio: float,
    line_like_second_axis_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate source rows and their local trusted-surface fit diagnostics."""
    trusted = np.unique(np.asarray(semantic_ids, np.int64))
    if trusted.size < max(3, minimum_neighbors):
        return (
            np.zeros(0, np.int64),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int8),
        )

    means = np.asarray(scene.means, np.float64)
    trusted_points = means[trusted]
    lo = np.percentile(trusted_points, 1.0, axis=0)
    hi = np.percentile(trusted_points, 99.0, axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    object_size = float(np.linalg.norm(extent))
    radius = max(float(neighbor_radius_fraction) * object_size, 1e-9)

    is_trusted = np.zeros(len(scene.vertex), bool)
    is_trusted[trusted] = True
    expanded = (means >= lo - radius) & (means <= hi + radius)
    pool = np.flatnonzero(expanded.all(axis=1) & ~is_trusted)
    if pool.size == 0:
        return (
            pool.astype(np.int64),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int8),
        )

    k = min(max(int(neighbor_count), int(minimum_neighbors)), len(trusted))
    tree = cKDTree(trusted_points)
    distances, nearest = tree.query(means[pool], k=k)
    if k == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]
    close = distances[:, min(k - 1, int(minimum_neighbors) - 1)] <= radius
    pool = pool[close]
    distances = distances[close]
    nearest = nearest[close]
    if pool.size == 0:
        return (
            pool.astype(np.int64),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int8),
        )

    neighbors = trusted_points[nearest]                    # [M,K,3]
    center = neighbors.mean(axis=1)
    centered = neighbors - center[:, None, :]
    covariance = np.einsum("mki,mkj->mij", centered, centered)
    covariance /= max(k - 1, 1)
    values, vectors = np.linalg.eigh(covariance)           # ascending
    delta = means[pool] - center
    normal = vectors[:, :, 0]
    plane_distance = np.abs(np.einsum("mi,mi->m", delta, normal))
    second_ratio = values[:, 1] / np.maximum(values[:, 2], 1e-12)
    line_like = second_ratio < float(line_like_second_axis_ratio)
    principal = vectors[:, :, 2]
    along = np.einsum("mi,mi->m", delta, principal)
    line_distance = np.linalg.norm(
        delta - along[:, None] * principal,
        axis=1,
    )
    manifold_distance = np.where(line_like, line_distance, plane_distance)
    manifold_rank = np.where(line_like, 1, 2).astype(np.int8)

    max_scales = np.max(np.exp(np.asarray(scene.log_scales, np.float64)), axis=1)
    local_scale = np.maximum(
        max_scales[pool],
        np.median(max_scales[trusted[nearest]], axis=1),
    )
    neighbor_span = np.maximum(distances[:, -1], 1e-12)
    tolerance = np.maximum(
        float(plane_scale_tolerance) * local_scale,
        float(plane_span_tolerance) * neighbor_span,
    )
    planarity = values[:, 0] / np.maximum(values[:, 1] + values[:, 2], 1e-12)
    good = (
        (manifold_distance <= tolerance)
        & (planarity <= float(maximum_planarity_ratio))
    )
    return (
        pool[good].astype(np.int64),
        manifold_distance[good].astype(np.float32),
        tolerance[good].astype(np.float32),
        planarity[good].astype(np.float32),
        manifold_rank[good],
    )


def collect_surface_rescue_evidence(
    scene,
    semantic_ids: np.ndarray,
    views,
    masks_by_view: dict[str, np.ndarray],
    renderer,
    *,
    neighbor_count: int = 32,
    minimum_neighbors: int = 12,
    neighbor_radius_fraction: float = 0.08,
    plane_scale_tolerance: float = 3.0,
    plane_span_tolerance: float = 0.12,
    maximum_planarity_ratio: float = 0.20,
    line_like_second_axis_ratio: float = 0.05,
    envelope_dilation_px_512: int = 3,
    relative_depth_tolerance: float = 0.005,
    gaussian_scale_tolerance: float = 3.0,
    minimum_depth_alpha: float = 0.20,
    minimum_visible_mass: float = 0.01,
    minimum_support_views: int = 2,
    maximum_contradiction_views: int = 0,
    minimum_camera_baseline_deg: float = 5.0,
    maximum_cameras: int = 32,
) -> tuple[SurfaceRescueEvidence, dict[str, object]]:
    """Return conservative source-Gaussian rescue evidence."""
    candidate_ids, plane_distance, plane_tolerance, planarity, manifold_rank = _local_surface_candidates(
        scene,
        semantic_ids,
        neighbor_count=neighbor_count,
        minimum_neighbors=minimum_neighbors,
        neighbor_radius_fraction=neighbor_radius_fraction,
        plane_scale_tolerance=plane_scale_tolerance,
        plane_span_tolerance=plane_span_tolerance,
        maximum_planarity_ratio=maximum_planarity_ratio,
        line_like_second_axis_ratio=line_like_second_axis_ratio,
    )
    count = len(candidate_ids)
    empty_i = np.zeros(count, np.int16)
    if count == 0:
        evidence = SurfaceRescueEvidence(
            candidate_ids=candidate_ids,
            local_plane_distance=plane_distance,
            local_plane_tolerance=plane_tolerance,
            planarity_ratio=planarity,
            local_manifold_rank=np.zeros(count, np.int8),
            envelope_support_views=empty_i.copy(),
            raw_semantic_support_views=empty_i.copy(),
            contradiction_views=empty_i.copy(),
            visible_views=empty_i.copy(),
            rescue=np.zeros(count, bool),
        )
        return evidence, {
            "enabled": True,
            "candidate_gaussians": 0,
            "rescued_gaussians": 0,
            "reason": "no non-semantic Gaussians fit a trusted local surface",
        }

    means = np.asarray(scene.means[candidate_ids], np.float64)
    max_scale = np.max(np.exp(np.asarray(scene.log_scales[candidate_ids], np.float64)), axis=1)
    selected_views = select_diverse_views(
        [view for view in views if view.view_id in masks_by_view],
        np.median(np.asarray(scene.means[np.asarray(semantic_ids, np.int64)]), axis=0),
        minimum_baseline_deg=minimum_camera_baseline_deg,
        maximum_views=maximum_cameras,
    )

    envelope_support = np.zeros(count, np.int16)
    raw_support = np.zeros(count, np.int16)
    contradiction = np.zeros(count, np.int16)
    visible_views = np.zeros(count, np.int16)
    local_index = np.full(len(scene.vertex), -1, np.int32)
    local_index[candidate_ids] = np.arange(count, dtype=np.int32)
    per_view: list[dict[str, object]] = []

    for view in selected_views:
        raw_mask = np.asarray(masks_by_view[view.view_id], bool)
        envelope = object_envelope(raw_mask, envelope_dilation_px_512)

        uv, z = project_to_pixels(means, view.viewmat, view.intrinsic)
        finite = np.isfinite(uv).all(axis=1)
        px = np.full(count, -1, np.int64)
        py = np.full(count, -1, np.int64)
        px[finite] = np.floor(uv[finite, 0]).astype(np.int64)
        py[finite] = np.floor(uv[finite, 1]).astype(np.int64)
        in_frame = (
            finite
            & (z > 1e-9)
            & (px >= 0)
            & (px < int(view.width))
            & (py >= 0)
            & (py < int(view.height))
        )

        source_depth, source_alpha = renderer.render_depth(view.view_id, edited=False)
        contribution = renderer.contribution_map(view.view_id, edited=False, blend=True)
        mass = _visible_mass_for_candidates(contribution, local_index, count)
        contributes = mass >= float(minimum_visible_mass)

        rows = np.flatnonzero(in_frame)
        support_here = np.zeros(count, bool)
        raw_here = np.zeros(count, bool)
        contradiction_here = np.zeros(count, bool)
        if rows.size:
            depth = np.asarray(source_depth[py[rows], px[rows]], np.float64)
            alpha = np.asarray(source_alpha[py[rows], px[rows]], np.float64)
            valid_depth = (
                np.isfinite(depth)
                & (depth > 0)
                & np.isfinite(alpha)
                & (alpha >= float(minimum_depth_alpha))
            )
            vr = rows[valid_depth]
            if vr.size:
                depth = depth[valid_depth]
                tolerance = np.maximum(
                    float(relative_depth_tolerance) * depth,
                    float(gaussian_scale_tolerance) * max_scale[vr],
                )
                same_layer = np.abs(z[vr] - depth) <= tolerance
                actually_visible = same_layer | contributes[vr]
                inside_envelope = envelope[py[vr], px[vr]]
                inside_raw = raw_mask[py[vr], px[vr]]
                support_here[vr] = actually_visible & inside_envelope
                raw_here[vr] = actually_visible & inside_raw
                contradiction_here[vr] = actually_visible & ~inside_envelope

        envelope_support += support_here.astype(np.int16)
        raw_support += raw_here.astype(np.int16)
        contradiction += contradiction_here.astype(np.int16)
        visible_views += contributes.astype(np.int16)
        per_view.append({
            "view_id": view.view_id,
            "candidate_in_frame": int(in_frame.sum()),
            "envelope_support": int(support_here.sum()),
            "raw_semantic_support": int(raw_here.sum()),
            "contradictions": int(contradiction_here.sum()),
            "contributing_candidates": int(contributes.sum()),
        })

    rescue = surface_rescue_decision(
        envelope_support,
        contradiction,
        minimum_support_views=minimum_support_views,
        maximum_contradiction_views=maximum_contradiction_views,
    )
    evidence = SurfaceRescueEvidence(
        candidate_ids=candidate_ids,
        local_plane_distance=plane_distance,
        local_plane_tolerance=plane_tolerance,
        planarity_ratio=planarity,
        local_manifold_rank=manifold_rank,
        envelope_support_views=envelope_support,
        raw_semantic_support_views=raw_support,
        contradiction_views=contradiction,
        visible_views=visible_views,
        rescue=rescue,
    )
    rescued = evidence.rescued_ids
    hole_only = rescue & (raw_support == 0)
    report = {
        "enabled": True,
        "candidate_gaussians": int(count),
        "rescued_gaussians": int(rescue.sum()),
        "rescued_fraction_of_candidates": float(rescue.mean()),
        "rescued_with_zero_raw_sam_support": int(hole_only.sum()),
        "rescued_with_2plus_envelope_views": int(
            (rescue & (envelope_support >= 2)).sum()
        ),
        "rescued_with_3plus_envelope_views": int(
            (rescue & (envelope_support >= 3)).sum()
        ),
        "rescued_from_line_manifold": int((rescue & (manifold_rank == 1)).sum()),
        "rescued_from_plane_manifold": int((rescue & (manifold_rank == 2)).sum()),
        "judged_camera_count": int(len(selected_views)),
        "parameters": {
            "neighbor_count": int(neighbor_count),
            "minimum_neighbors": int(minimum_neighbors),
            "neighbor_radius_fraction": float(neighbor_radius_fraction),
            "plane_scale_tolerance": float(plane_scale_tolerance),
            "plane_span_tolerance": float(plane_span_tolerance),
            "maximum_planarity_ratio": float(maximum_planarity_ratio),
            "line_like_second_axis_ratio": float(line_like_second_axis_ratio),
            "envelope_dilation_px_512": int(envelope_dilation_px_512),
            "relative_depth_tolerance": float(relative_depth_tolerance),
            "gaussian_scale_tolerance": float(gaussian_scale_tolerance),
            "minimum_depth_alpha": float(minimum_depth_alpha),
            "minimum_visible_mass": float(minimum_visible_mass),
            "minimum_support_views": int(minimum_support_views),
            "maximum_contradiction_views": int(maximum_contradiction_views),
            "minimum_camera_baseline_deg": float(minimum_camera_baseline_deg),
            "maximum_cameras": int(maximum_cameras),
        },
        "rescued_source_ids_sample": [int(x) for x in rescued[:100].tolist()],
        "views": [view.view_id for view in selected_views],
        "per_view": per_view,
    }
    return evidence, report
