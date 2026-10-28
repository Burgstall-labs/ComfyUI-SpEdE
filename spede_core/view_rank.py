# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Which cameras see a selected object best (for choosing the hero).

The selection is a mask in one camera. Its Gaussians are the ones that make
up those pixels (most of their rendered weight inside the mask), so a thin
object selects its spokes, not the wall behind them. Each camera is then
scored by how much of that surface it sees unoccluded, how much of it is in
the frame, and how large it is.
"""
from __future__ import annotations

import numpy as np

from .roi import ContributionMap


def selected_surface_ids(
    contribution: ContributionMap,
    mask: np.ndarray,
    *,
    min_weight: float = 0.05,
    min_inside: float = 0.5,
    return_weights: bool = False,
):
    """Gaussians that make up the masked pixels: at least ``min_weight`` in
    some masked pixel and at least ``min_inside`` of their rendered weight
    inside the mask (big background splats that also touch it are left out)."""
    ids = np.asarray(contribution.gaussian_ids)
    weights = np.asarray(contribution.weights, np.float64)
    mask = np.asarray(mask, bool)
    valid = ids >= 0
    if not valid.any():
        return np.zeros(0, np.int64)
    size = int(ids[valid].max()) + 1
    total = np.bincount(ids[valid], weights=weights[valid], minlength=size)
    inside_px = valid & mask[..., None]
    inside = np.bincount(ids[inside_px], weights=weights[inside_px], minlength=size)
    strong = np.zeros(size, bool)
    strong_px = inside_px & (weights >= float(min_weight))
    strong[np.unique(ids[strong_px])] = True
    keep = strong & (inside >= float(min_inside) * np.maximum(total, 1e-12))
    ids_out = np.flatnonzero(keep).astype(np.int64)
    return (ids_out, inside[ids_out]) if return_weights else ids_out


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(np.asarray(weights, np.float64)[order])
    return float(np.asarray(values)[order][np.searchsorted(cum, q / 100.0 * cum[-1])])


def click_points(mask: np.ndarray, count: int, *, reach: float = 0.6) -> list[list[float]]:
    """Positive clicks inside the dense core of a projected selection: the
    largest connected part, eroded, points spread by farthest-point order but
    within ``reach`` of the core's largest distance from its centre (fringe
    splats from the selection's edge must not pull SAM onto the table)."""
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    core = ndimage.binary_erosion(mask, iterations=3)
    if core.sum() < count:
        core = mask
    labels, n = ndimage.label(core)
    if n == 0:
        return []
    largest = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    ys, xs = np.nonzero(largest)
    cy, cx = ys.mean(), xs.mean()
    distance = np.hypot(ys - cy, xs - cx)
    # A ring or a two-pixel sliver has no pixel near its centre: keep the
    # closest ones rather than none.
    near = distance <= max(float(reach) * distance.max(), distance.min()) + 1e-9
    ys, xs, distance = ys[near], xs[near], distance[near]
    chosen = [int(np.argmin(distance))]
    for _ in range(min(count, len(ys)) - 1):
        d = np.min([(ys - ys[c]) ** 2 + (xs - xs[c]) ** 2 for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    return [[float(xs[c]), float(ys[c]), 1] for c in chosen]


def in_frame_fraction(means: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> float:
    """Share of the points in front of the camera and inside the image."""
    means = np.asarray(means, np.float64)
    if len(means) == 0:
        return 0.0
    view = np.asarray(viewmat, np.float64)
    k = np.asarray(intrinsic, np.float64)
    cam = means @ view[:3, :3].T + view[:3, 3]
    front = cam[:, 2] > 1e-6
    u = np.full(len(means), -1.0)
    v = np.full(len(means), -1.0)
    u[front] = cam[front, 0] / cam[front, 2] * k[0, 0] + k[0, 2]
    v[front] = cam[front, 1] / cam[front, 2] * k[1, 1] + k[1, 2]
    inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return float(inside.mean())


def view_score(visible: float, in_frame: float, area_fraction: float, *, full_size_area: float = 0.05) -> float:
    """Visible share x in-frame share x a size term that stops growing once
    the object covers ``full_size_area`` of the image."""
    size = min(1.0, max(0.0, float(area_fraction)) / float(full_size_area)) ** 0.5
    return float(visible) * float(in_frame) * size


def rank_views(
    renderer,
    cameras,
    hero_view_id: str,
    hero_mask: np.ndarray,
    *,
    depth_tolerance: float = 0.15,
    point_count: int = 4,
    max_depth_spread: float = 1.5,
) -> dict[str, object]:
    """Score every real camera for the selection in ``hero_view_id``; each
    view also gets click points on its visible part (to move the selection)."""
    contribution = renderer.contribution_map(hero_view_id, edited=False, blend=True)
    selected, inside_weight = selected_surface_ids(contribution, hero_mask, return_weights=True)
    if selected.size == 0:
        return {"status": "empty_selection", "hero_view_id": hero_view_id, "views": []}
    scene = renderer._scene(False)
    keep = np.zeros(len(scene.vertex), bool)
    keep[selected] = True
    others = np.flatnonzero(~keep)
    selected_means = scene.means[selected]
    # One object is one surface: its Gaussians lie at similar depths from the
    # hero. An object the splat only painted into far background splats (a
    # bike at the edge of an object-centric capture) spreads over depth.
    hero = cameras.by_id[hero_view_id]
    hero_view = np.asarray(hero.viewmat, np.float64)
    depth = (selected_means @ hero_view[:3, :3].T + hero_view[:3, 3])[:, 2]
    low = weighted_percentile(depth, inside_weight, 10)
    high = weighted_percentile(depth, inside_weight, 90)
    depth_spread = float(high / max(low, 1e-6))
    rows = []
    for view in cameras.views:
        if view.is_virtual:
            continue
        _img, only_alpha, meta, _scene = renderer._render_raw(
            view.view_id, edited=False, excluded_gaussian_ids=others, render_mode="RGB+ED")
        only_depth = np.asarray(meta["depth"])
        full_depth, _alpha = renderer.render_depth(view.view_id, edited=False)
        projected = np.asarray(only_alpha) >= 0.5
        area = int(projected.sum())
        visible_px = projected & (only_depth <= np.asarray(full_depth) * (1.0 + depth_tolerance) + 1e-6)
        visible = float(visible_px.sum()) / area if area else 0.0
        frame = in_frame_fraction(selected_means, view.viewmat, view.intrinsic, view.width, view.height)
        area_fraction = area / float(view.width * view.height)
        points = click_points(visible_px & (np.asarray(only_alpha) >= 0.9), point_count)
        rows.append({
            "view_id": view.view_id,
            "score": view_score(visible, frame, area_fraction),
            "visible": visible,
            "in_frame": frame,
            "area_fraction": area_fraction,
            "points": points,
        })
    rows.sort(key=lambda row: (-round(row["score"], 2), -row["area_fraction"]))
    return {"status": "ranked", "hero_view_id": hero_view_id, "selected_gaussians": int(selected.size),
            "depth_spread": depth_spread, "one_surface": bool(depth_spread <= max_depth_spread),
            "recommended": rows[0]["view_id"] if rows else None, "views": rows}


def three_camera_to_view(position, quaternion, fov_deg: float, aspect: float, width: int = 768):
    """A three.js perspective camera (x right, y up, looking down -z; the
    viewer shows the splat in its own world frame) as a pinhole CameraView
    (x right, y down, z forward)."""
    from .renderer import CameraView

    x, y, z, w = (float(v) for v in quaternion)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    c2w = np.eye(4)
    c2w[:3, :3] = rotation @ np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = np.asarray(position, np.float64)
    height = max(1, int(round(width / float(aspect))))
    fy = 0.5 * height / np.tan(np.radians(float(fov_deg)) / 2.0)
    k = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraView("explore", np.linalg.inv(c2w).astype(np.float32), k.astype(np.float32), int(width), height,
                      synthetic=True)


def hero_score(visible: bool, cos_to_user: float, user_distance: float, camera_distance: float,
               edge_margin: float) -> float:
    """How good a real camera is as the hero for a point the user picked:
    it must see the point; then it should look from about where the user
    looked (direction) and not be much farther away (the object stays large),
    and the point should not sit at the image edge."""
    if not visible or edge_margin <= 0.0:
        return 0.0
    direction = 0.5 + 0.5 * float(cos_to_user)
    closeness = min(1.0, float(user_distance) / max(float(camera_distance), 1e-9)) ** 0.5
    edge = min(1.0, float(edge_margin) / 0.1)
    return direction * closeness * edge


def front_depth(depth: np.ndarray, alpha: np.ndarray, pixel, *, patch_radius: int = 2,
                minimum_alpha: float = 0.5, percentile: float = 20.0) -> float | None:
    """Depth of the front-most surface in a small patch around a click. The
    rendered expected depth blends foreground and background at an object's
    edge (a median can land behind the object): a low percentile of the
    patch keeps the click on the object in front."""
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    h, w = depth.shape
    r = max(0, int(patch_radius))
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    patch_depth, patch_alpha = depth[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    good = (np.isfinite(patch_depth) & (patch_depth > 0) & np.isfinite(patch_alpha)
            & (patch_alpha > float(minimum_alpha)))
    if not bool(good.any()):
        return None
    return float(np.percentile(patch_depth[good], float(percentile), method="lower"))   # a real rendered depth


def pick_world_point(
    renderer,
    explore_view,
    pixel,
    *,
    minimum_alpha: float = 0.5,
    patch_radius: int = 2,
) -> dict[str, object]:
    """World-space surface point under a click in an arbitrary viewer camera.

    Uses the splat's own rendered expected depth over a small patch (stable
    around thin Gaussian holes); its front-most surface, not the median, so an
    edge click stays on the object (front_depth).
    """
    renderer.add_view(explore_view)
    depth, alpha = renderer.render_depth(explore_view.view_id, edited=False)
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    z = front_depth(np.asarray(depth), np.asarray(alpha), pixel, patch_radius=patch_radius, minimum_alpha=minimum_alpha)
    if z is None:
        return {"status": "nothing_there", "reason": "no splat surface under the click"}
    k = np.asarray(explore_view.intrinsic, np.float64)
    c2w = np.linalg.inv(np.asarray(explore_view.viewmat, np.float64))
    point_cam = np.array([
        (u + 0.5 - k[0, 2]) / k[0, 0] * z,
        (v + 0.5 - k[1, 2]) / k[1, 1] * z,
        z,
    ])
    point = c2w[:3, :3] @ point_cam + c2w[:3, 3]
    return {
        "status": "picked",
        "point": [float(x) for x in point],
        "depth": z,
        "pixel": [float(pixel[0]), float(pixel[1])],
    }


def pick_hero(renderer, cameras, explore_view, pixel, *, depth_tolerance: float = 0.05) -> dict[str, object]:
    """The world point under the user's click in a free viewer camera, and
    the real camera to select it in (with the click's pixel there)."""
    renderer.add_view(explore_view)
    depth, alpha = renderer.render_depth(explore_view.view_id, edited=False)
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    z = front_depth(np.asarray(depth), np.asarray(alpha), pixel)
    if z is None:
        return {"status": "nothing_there", "reason": "no surface under the click"}
    k = np.asarray(explore_view.intrinsic, np.float64)
    c2w = np.linalg.inv(np.asarray(explore_view.viewmat, np.float64))
    point = c2w[:3, :3] @ np.array([(u + 0.5 - k[0, 2]) / k[0, 0] * z, (v + 0.5 - k[1, 2]) / k[1, 1] * z, z]) + c2w[:3, 3]
    user_eye = c2w[:3, 3]
    user_dir = (point - user_eye) / np.linalg.norm(point - user_eye)
    rows = []
    for view in cameras.views:
        if view.is_virtual:
            continue
        vm = np.asarray(view.viewmat, np.float64)
        cam = vm[:3, :3] @ point + vm[:3, 3]
        if cam[2] <= 1e-6:
            continue
        kk = np.asarray(view.intrinsic, np.float64)
        pu, pv = cam[0] / cam[2] * kk[0, 0] + kk[0, 2], cam[1] / cam[2] * kk[1, 1] + kk[1, 2]
        margin = min(pu, pv, view.width - pu, view.height - pv) / float(max(view.width, view.height))
        if margin <= 0:
            continue
        view_depth, view_alpha = renderer.render_depth(view.view_id, edited=False)
        seen = view_depth[int(pv), int(pu)]
        visible = bool(view_alpha[int(pv), int(pu)] > 0.5 and seen >= cam[2] * (1.0 - depth_tolerance))
        eye = np.linalg.inv(vm)[:3, 3]
        cam_dir = (point - eye) / np.linalg.norm(point - eye)
        score = hero_score(visible, float(user_dir @ cam_dir), float(np.linalg.norm(point - user_eye)),
                           float(np.linalg.norm(point - eye)), margin)
        rows.append({"view_id": view.view_id, "score": score, "visible": visible, "pixel": [float(pu), float(pv)],
                     "cos_to_user": float(user_dir @ cam_dir), "distance": float(np.linalg.norm(point - eye))})
    rows.sort(key=lambda r: -r["score"])
    if not rows or rows[0]["score"] <= 0:
        return {"status": "no_camera", "reason": "no real camera sees the picked point", "point": point.tolist()}
    return {"status": "picked", "point": point.tolist(), "view_id": rows[0]["view_id"], "pixel": rows[0]["pixel"],
            "candidates": rows[:6]}
