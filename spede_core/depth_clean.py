# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Conservative model-depth cleanup for extracted Gaussian objects.

A selected Gaussian is suspicious when its centre repeatedly sits clearly in
front of the source model's rendered expected-depth surface from independent
capture cameras.  The tolerance follows both scene depth and the Gaussian's
own scale, so large valid splats are not judged with a scene-unit magic number.

This is a cleanup gate after semantic multiview object extraction, not a
replacement for semantic selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry_utils import project_to_pixels


@dataclass(frozen=True)
class DepthCleanEvidence:
    source_ids: np.ndarray
    surface_support_views: np.ndarray
    front_violation_views: np.ndarray
    depth_judged_views: np.ndarray
    visible_views: np.ndarray | None
    remove: np.ndarray

    @property
    def removed_ids(self) -> np.ndarray:
        return self.source_ids[self.remove]

    @property
    def kept_ids(self) -> np.ndarray:
        return self.source_ids[~self.remove]


def depth_clean_decision(
    surface_support_views: np.ndarray,
    front_violation_views: np.ndarray,
    *,
    visible_views: np.ndarray | None = None,
    minimum_front_views: int = 2,
    maximum_surface_support_views: int = 0,
    remove_never_visible: bool = False,
) -> np.ndarray:
    """Return the conservative removal mask from accumulated view evidence."""
    support = np.asarray(surface_support_views, np.int32)
    front = np.asarray(front_violation_views, np.int32)
    if support.shape != front.shape:
        raise ValueError("surface support and front-violation arrays must align")
    if minimum_front_views < 1:
        raise ValueError("minimum_front_views must be positive")
    if maximum_surface_support_views < 0:
        raise ValueError("maximum_surface_support_views must be non-negative")

    remove = (
        (front >= int(minimum_front_views))
        & (support <= int(maximum_surface_support_views))
    )
    if remove_never_visible:
        if visible_views is None:
            raise ValueError("remove_never_visible requires visible_views evidence")
        visible = np.asarray(visible_views, np.int32)
        if visible.shape != support.shape:
            raise ValueError("visible_views must align with depth evidence")
        remove |= (visible == 0) & (support == 0)
    return remove


def _visible_mass_for_selected(contribution, local_index: np.ndarray, count: int) -> np.ndarray:
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


def select_diverse_views(
    views,
    object_center: np.ndarray,
    *,
    minimum_baseline_deg: float = 5.0,
    maximum_views: int = 48,
):
    """Greedy farthest-point subset of cameras around the selected object."""
    views = list(views)
    if not views or maximum_views <= 0:
        return []
    center = np.asarray(object_center, np.float64)
    directions = []
    valid_views = []
    for view in views:
        c2w = np.linalg.inv(np.asarray(view.viewmat, np.float64))
        delta = c2w[:3, 3] - center
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        directions.append(delta / length)
        valid_views.append(view)
    if not valid_views:
        return []

    directions = np.asarray(directions, np.float64)
    # Stable start: lexical view id, then farthest-point coverage.
    first = min(range(len(valid_views)), key=lambda i: str(valid_views[i].view_id))
    chosen = [first]
    remaining = set(range(len(valid_views))) - {first}
    baseline = float(minimum_baseline_deg)

    while remaining and len(chosen) < int(maximum_views):
        best = None
        best_key = None
        for index in remaining:
            dots = np.clip(directions[chosen] @ directions[index], -1.0, 1.0)
            nearest_deg = float(np.degrees(np.arccos(np.max(dots))))
            key = (nearest_deg, str(valid_views[index].view_id))
            if best_key is None or key > best_key:
                best, best_key = index, key
        if best is None or best_key[0] < baseline:
            break
        chosen.append(best)
        remaining.remove(best)

    return [valid_views[i] for i in chosen]


def collect_depth_clean_evidence(
    scene,
    object_ids: np.ndarray,
    views,
    renderer,
    *,
    relative_depth_tolerance: float = 0.005,
    gaussian_scale_tolerance: float = 3.0,
    minimum_depth_alpha: float = 0.20,
    minimum_front_views: int = 2,
    maximum_surface_support_views: int = 0,
    remove_never_visible: bool = False,
    minimum_visible_mass: float = 0.01,
    minimum_camera_baseline_deg: float = 5.0,
    maximum_cameras: int = 48,
) -> tuple[DepthCleanEvidence, dict[str, object]]:
    """Judge selected source Gaussians against source-model depth in cameras.

    A view contributes one of three depth observations for an in-frame
    Gaussian centre when the source depth is valid:
      - surface support: centre agrees with expected depth within tolerance;
      - front violation: centre is clearly nearer than expected depth;
      - behind/occluded: neither, and therefore not evidence for removal.

    Tolerance per Gaussian/view:
        max(relative_depth_tolerance * rendered_depth,
            gaussian_scale_tolerance * max_world_scale)

    Optional actual visibility uses the source contributor map. It is computed
    only when remove_never_visible is requested because contributor maps are
    more expensive than depth renders.
    """
    ids = np.unique(np.asarray(object_ids, np.int64).reshape(-1))
    if ids.size == 0:
        raise ValueError("object_ids is empty")
    if int(ids.min()) < 0 or int(ids.max()) >= len(scene.vertex):
        raise ValueError("object_ids contain rows outside the source scene")
    if relative_depth_tolerance < 0 or gaussian_scale_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    if not 0 <= minimum_depth_alpha <= 1:
        raise ValueError("minimum_depth_alpha must be in [0,1]")

    points = np.asarray(scene.means[ids], np.float64)
    max_scale = np.max(np.exp(np.asarray(scene.log_scales[ids], np.float64)), axis=1)
    input_views = list(views)
    views = select_diverse_views(
        input_views,
        np.median(points, axis=0),
        minimum_baseline_deg=minimum_camera_baseline_deg,
        maximum_views=maximum_cameras,
    )
    count = len(ids)
    support = np.zeros(count, np.int16)
    front = np.zeros(count, np.int16)
    judged = np.zeros(count, np.int16)
    visible = np.zeros(count, np.int16) if remove_never_visible else None
    per_view: list[dict[str, object]] = []

    local_index = None
    if remove_never_visible:
        local_index = np.full(len(scene.vertex), -1, np.int32)
        local_index[ids] = np.arange(count, dtype=np.int32)

    for view in views:
        uv, z = project_to_pixels(points, view.viewmat, view.intrinsic)
        finite_uv = np.isfinite(uv).all(axis=1)
        px = np.full(count, -1, np.int64)
        py = np.full(count, -1, np.int64)
        px[finite_uv] = np.floor(uv[finite_uv, 0]).astype(np.int64)
        py[finite_uv] = np.floor(uv[finite_uv, 1]).astype(np.int64)
        in_frame = (
            finite_uv
            & (z > 1e-9)
            & (px >= 0)
            & (px < int(view.width))
            & (py >= 0)
            & (py < int(view.height))
        )
        depth, alpha = renderer.render_depth(view.view_id, edited=False)
        rows = np.flatnonzero(in_frame)
        if rows.size:
            surface = np.asarray(depth[py[rows], px[rows]], np.float64)
            surface_alpha = np.asarray(alpha[py[rows], px[rows]], np.float64)
            valid = (
                np.isfinite(surface)
                & (surface > 0)
                & np.isfinite(surface_alpha)
                & (surface_alpha >= float(minimum_depth_alpha))
            )
            valid_rows = rows[valid]
            if valid_rows.size:
                surface = surface[valid]
                tolerance = np.maximum(
                    float(relative_depth_tolerance) * surface,
                    float(gaussian_scale_tolerance) * max_scale[valid_rows],
                )
                delta = z[valid_rows] - surface
                is_support = np.abs(delta) <= tolerance
                is_front = delta < -tolerance
                judged[valid_rows] += 1
                support[valid_rows[is_support]] += 1
                front[valid_rows[is_front]] += 1
        if remove_never_visible:
            contribution = renderer.contribution_map(
                view.view_id, edited=False, blend=True
            )
            mass = _visible_mass_for_selected(contribution, local_index, count)
            visible += (mass >= float(minimum_visible_mass)).astype(np.int16)

        per_view.append(
            {
                "view_id": view.view_id,
                "in_frame": int(in_frame.sum()),
                "depth_judged": int(valid_rows.size) if rows.size else 0,
                "surface_supported": (
                    int(is_support.sum()) if rows.size and valid_rows.size else 0
                ),
                "front_violations": (
                    int(is_front.sum()) if rows.size and valid_rows.size else 0
                ),
            }
        )

    remove = depth_clean_decision(
        support,
        front,
        visible_views=visible,
        minimum_front_views=minimum_front_views,
        maximum_surface_support_views=maximum_surface_support_views,
        remove_never_visible=remove_never_visible,
    )
    evidence = DepthCleanEvidence(
        source_ids=ids,
        surface_support_views=support,
        front_violation_views=front,
        depth_judged_views=judged,
        visible_views=visible,
        remove=remove,
    )

    report = {
        "enabled": True,
        "input_gaussians": int(count),
        "removed_gaussians": int(remove.sum()),
        "remaining_gaussians": int((~remove).sum()),
        "removed_fraction": float(remove.mean()),
        "surface_supported_gaussians": int((support >= 1).sum()),
        "front_violation_ge_1": int((front >= 1).sum()),
        "front_violation_ge_2": int((front >= 2).sum()),
        "never_depth_judged": int((judged == 0).sum()),
        "never_surface_supported": int((support == 0).sum()),
        "never_visible": (
            None if visible is None else int((visible == 0).sum())
        ),
        "removed_never_visible_enabled": bool(remove_never_visible),
        "input_camera_count": int(len(input_views)),
        "judged_camera_count": int(len(views)),
        "parameters": {
            "relative_depth_tolerance": float(relative_depth_tolerance),
            "gaussian_scale_tolerance": float(gaussian_scale_tolerance),
            "minimum_depth_alpha": float(minimum_depth_alpha),
            "minimum_front_views": int(minimum_front_views),
            "maximum_surface_support_views": int(maximum_surface_support_views),
            "minimum_visible_mass": float(minimum_visible_mass),
            "minimum_camera_baseline_deg": float(minimum_camera_baseline_deg),
            "maximum_cameras": int(maximum_cameras),
        },
        "front_histogram": {
            str(k): int((front == k).sum())
            for k in range(int(front.max()) + 1)
        },
        "support_histogram": {
            str(k): int((support == k).sum())
            for k in range(int(support.max()) + 1)
        },
        "removed_source_ids_sample": [
            int(x) for x in evidence.removed_ids[:100].tolist()
        ],
        "views": [v.view_id for v in views],
        "per_view": per_view,
    }
    return evidence, report
