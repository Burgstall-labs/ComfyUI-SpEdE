# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Object selection as a visual hull, for REPLACE/REMOVE deletion.

Contributor maps only record the front layers of an object, so selecting the
Gaussians that dominate a mask leaves the rest of the object in place. The
hull instead starts from every Gaussian centre inside the user's hero mask
frustum and carves away those that fall outside the object's silhouette in
the other real views (SAM masks that passed the silhouette vote). What stays
is the object including its interior; the background behind it in the hero
frustum is carved away by views that see it beside the object.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .roi import dilate_mask


@dataclass(frozen=True)
class HullView:
    view_id: str
    viewmat: np.ndarray
    intrinsic: np.ndarray
    silhouette: np.ndarray
    # The source scene's rendered depth/alpha in this camera. With them a view
    # abstains for points hidden behind something nearer (clutter in front of
    # the object) instead of voting them out.
    depth: np.ndarray | None = None
    alpha: np.ndarray | None = None


@dataclass(frozen=True)
class VisualHullResult:
    object_ids: np.ndarray
    candidate_ids: np.ndarray
    views_seen: np.ndarray
    views_inside: np.ndarray
    per_view_carved: dict[str, int]


def project_points(
    points: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel indices (x, y) of points and whether they land in front, in frame."""
    homogeneous = np.c_[np.asarray(points, np.float64), np.ones(len(points))]
    camera = homogeneous @ np.asarray(viewmat, np.float64).T
    z = camera[:, 2]
    safe_z = np.where(z > 1e-9, z, 1.0)
    k = np.asarray(intrinsic, np.float64)
    u = k[0, 0] * camera[:, 0] / safe_z + k[0, 2]
    v = k[1, 1] * camera[:, 1] / safe_z + k[1, 2]
    in_frame = (z > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    x = np.clip(np.floor(u), 0, width - 1).astype(np.int64)
    y = np.clip(np.floor(v), 0, height - 1).astype(np.int64)
    return x, y, in_frame


def scaled_dilation(dilate_px: int, shape: tuple[int, ...]) -> int:
    """A dilation given for a 512 px image, at this image's size."""
    return int(round(int(dilate_px) * max(int(shape[0]), int(shape[1])) / 512.0))


def required_votes(seen: np.ndarray, view_tolerance: float, *, lenient: bool = True) -> np.ndarray:
    """How many of the ``seen`` views must have a point inside: all but
    ``floor(tol * seen)``; with a tolerance above zero and ``lenient``, from
    three views on all but at least one (one wrong mask never carves by
    itself). Tolerance 0 stays strict: every view."""
    seen = np.asarray(seen, np.int64)
    allowed = np.floor(float(view_tolerance) * seen + 1e-9).astype(np.int64)
    if lenient and view_tolerance > 0:
        allowed = np.where(seen >= 3, np.maximum(allowed, 1), allowed)
    return (seen - allowed).astype(np.int32)


def silhouette_votes(
    points: np.ndarray,
    views: list[HullView],
    *,
    dilate_px: int,
    view_tolerance: float,
    occlusion_tolerance: float = 0.07,
    lenient: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Which points lie inside the (dilated) silhouettes of the views that see them.

    A point is kept when it projects inside in enough of the views that see
    it: all but ``floor(view_tolerance * seen)`` of them, and from three
    views on always all but at least one (one wrong mask or an 8 px pose
    error never carves the object by itself). A view with depth abstains for
    a point that lands outside its silhouette while something opaque is
    nearer there (the point is hidden: a pot behind a pasta bag), so clutter
    cannot vote the object out; ``hidden`` counts those views. ``dilate_px``
    is for a 512 px image and scales with the view's size.
    Returns (keep, seen, inside, per-view outside masks, hidden).
    """
    if not 0.0 <= view_tolerance < 1.0:
        raise ValueError("view_tolerance must be in [0, 1)")
    points = np.asarray(points)
    seen = np.zeros(len(points), np.int32)
    inside = np.zeros(len(points), np.int32)
    hidden_count = np.zeros(len(points), np.int32)
    per_view_outside: dict[str, np.ndarray] = {}
    for view in views:
        raw = np.asarray(view.silhouette, dtype=bool)
        silhouette = dilate_mask(raw, scaled_dilation(dilate_px, raw.shape))
        vh, vw = silhouette.shape
        vx, vy, visible = project_points(points, view.viewmat, view.intrinsic, vw, vh)
        hit = visible.copy()
        hit[visible] = silhouette[vy[visible], vx[visible]]
        hidden = np.zeros(len(points), bool)
        if view.depth is not None and view.alpha is not None:
            vm = np.asarray(view.viewmat, np.float64)
            z = np.asarray(points, np.float64) @ vm[2, :3] + vm[2, 3]
            rendered = np.asarray(view.depth)[vy, vx]
            opaque = np.asarray(view.alpha)[vy, vx] > 0.5
            hidden = visible & ~hit & opaque & (rendered > 0) & (rendered < z * (1.0 - float(occlusion_tolerance)))
        seen += visible & ~hidden
        inside += hit
        hidden_count += hidden
        per_view_outside[view.view_id] = visible & ~hit & ~hidden
    required = required_votes(seen, view_tolerance, lenient=lenient)
    return inside >= required, seen, inside, per_view_outside, hidden_count


def visual_hull_object_ids(
    means: np.ndarray,
    *,
    hero_viewmat: np.ndarray,
    hero_intrinsic: np.ndarray,
    hero_mask: np.ndarray,
    views: list[HullView],
    dilate_px: int,
    view_tolerance: float,
    min_seen_fraction: float = 0.0,
    relaxed_view_tolerance: float | None = None,
    world_up: np.ndarray | None = None,
) -> VisualHullResult:
    """Gaussians inside the hero-mask frustum and inside the views' silhouettes.

    A candidate is kept when it projects inside the (dilated) silhouette in at
    least ``ceil((1 - view_tolerance) * seen)`` of the views where it lands in
    frame. Views where it is out of frame don't count, so a silhouette can only
    remove what it actually sees. ``view_tolerance`` absorbs a few imperfect
    masks; ``dilate_px`` absorbs pose and mask-edge error. A candidate seen by
    fewer than ``min_seen_fraction`` of the views is not the object: too few
    silhouettes could carve it (on bonsai, 10k large faint background
    Gaussians behind the object were seen by 2 of 20 views and deleted).

    With ``relaxed_view_tolerance`` and ``world_up``, a second pass also
    takes candidates that pass the looser tolerance, but only within the
    height range (1st-99th percentile along ``world_up``) of the strict
    object: a few imperfect part masks cut the bonsai pot's rim, which stayed
    as a dark slab, while below the object's base the looser vote reaches the
    stand under it.
    """
    if not 0.0 <= view_tolerance < 1.0:
        raise ValueError("view_tolerance must be in [0, 1)")
    hero = np.asarray(hero_mask, dtype=bool)
    height, width = hero.shape
    x, y, in_frame = project_points(means, hero_viewmat, hero_intrinsic, width, height)
    inside_hero = in_frame.copy()
    inside_hero[in_frame] = hero[y[in_frame], x[in_frame]]
    candidates = np.flatnonzero(inside_hero)
    points = np.asarray(means)[candidates]
    keep, seen, inside, per_view_outside, hidden = silhouette_votes(
        points, views, dilate_px=dilate_px, view_tolerance=view_tolerance
    )
    if not 0.0 <= min_seen_fraction <= 1.0:
        raise ValueError("min_seen_fraction must be in [0, 1]")
    # A view that saw the point's place (even hidden behind clutter) counts:
    # abstaining must not drop the very points it protected.
    enough_views = (seen + hidden) >= np.ceil(min_seen_fraction * len(views) - 1e-9)
    keep &= enough_views
    if relaxed_view_tolerance is not None and world_up is not None and keep.any():
        if not view_tolerance <= relaxed_view_tolerance < 1.0:
            raise ValueError("relaxed_view_tolerance must be in [view_tolerance, 1)")
        up = np.asarray(world_up, np.float64)
        up = up / np.linalg.norm(up)
        height = np.asarray(points, np.float64) @ up
        low, high = np.percentile(height[keep], [1.0, 99.0])
        relaxed = inside >= required_votes(seen, relaxed_view_tolerance)
        keep |= relaxed & enough_views & (height >= low) & (height <= high)
    carved = {vid: int((outside & ~keep).sum()) for vid, outside in per_view_outside.items()}
    return VisualHullResult(
        object_ids=candidates[keep].astype(np.int64),
        candidate_ids=candidates.astype(np.int64),
        views_seen=seen,
        views_inside=inside,
        per_view_carved=carved,
    )


def hull_views_from_segment_report(report: dict, cameras, *, exclude: set[str] | None = None,
                                   renderer=None) -> list[HullView]:
    """The silhouettes that passed the multi-part vote in a segment-views report
    (with the source scene's depth when a renderer is given)."""
    from PIL import Image

    exclude = exclude or set()
    views = []
    for view_id in report["voting_views"]:
        if view_id in exclude:
            continue
        camera = cameras.by_id[view_id]
        with Image.open(report["views"][view_id]["mask_path"]) as mask:
            silhouette = np.asarray(mask.convert("L")) > 127
        if silhouette.shape != (camera.height, camera.width):
            raise ValueError(f"silhouette for {view_id} does not match its camera size")
        depth = alpha = None
        if renderer is not None:
            depth, alpha = renderer.render_depth(view_id, edited=False)
        views.append(HullView(view_id, camera.viewmat, camera.intrinsic, silhouette, depth, alpha))
    return views


def consistent_views(renderer, views: list[HullView], object_ids: np.ndarray, *, dilate_px: int = 8,
                     min_containment: float = 0.75) -> tuple[list[HullView], dict[str, float]]:
    """Views whose mask agrees with the object: at least ``min_containment`` of
    the mask lies on the object set's own rendered silhouette there. A mask on
    another object (SAM's pick in a view where the object is barely visible),
    or one that runs far past it (SAM's masks of the bulldozer from four low
    cameras took in part of the table: containment 0.54-0.71, the good ones
    0.83-1.0), must neither peel nor count as a residual. Strays must be
    dropped from the set first: a few hundred of them over the table, dilated,
    made the half-table mask look consistent."""
    ids = np.asarray(object_ids, np.int64)
    keep, scores = [], {}
    for view in views:
        mask = np.asarray(view.silhouette, bool)
        if not mask.any() or ids.size == 0:
            scores[view.view_id] = 0.0
            continue
        _img, alpha = renderer.render_only(view.view_id, ids, edited=False)
        projected = dilate_mask(np.asarray(alpha) > 0.3, scaled_dilation(dilate_px, mask.shape))
        score = float((mask & projected).sum()) / float(mask.sum())
        scores[view.view_id] = score
        if score >= float(min_containment):
            keep.append(view)
    return keep, scores


def hero_surface_ids(means: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray, mask: np.ndarray,
                     depth: np.ndarray | None, alpha: np.ndarray | None, *,
                     depth_tolerance: float = 0.05, min_alpha: float = 0.5) -> np.ndarray:
    """The Gaussians on the surface the hero mask selects: centres inside the
    mask and at the hero render's depth there (within ``depth_tolerance`` of
    it), so not the background behind the object in the mask's frustum.
    Without a depth map every centre inside the mask counts."""
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    points = np.asarray(means, np.float64)
    x, y, front = project_points(points, viewmat, intrinsic, w, h)
    hit = front.copy()
    hit[front] = mask[y[front], x[front]]
    if depth is not None and alpha is not None:
        vm = np.asarray(viewmat, np.float64)
        z = points @ vm[2, :3] + vm[2, 3]
        rendered = np.asarray(depth, np.float64)[y, x]
        opaque = np.asarray(alpha)[y, x] > float(min_alpha)
        hit &= opaque & (rendered > 0) & (np.abs(z - rendered) <= float(depth_tolerance) * rendered)
    return np.flatnonzero(hit)


def mask_instances(mask: np.ndarray) -> np.ndarray:
    """A union mask's connected pieces as instances [N, H, W] (the picker's
    fallback when the segmenter gave only the union)."""
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    return np.stack([labels == i for i in range(1, count + 1)]) if count else np.zeros((0, *mask.shape), bool)


def pick_hero_instances(points: np.ndarray, view: HullView, instances: np.ndarray, *,
                        dilate_px: int = 8, min_fraction: float = 0.1, min_visible: int = 10,
                        occlusion_margin: float | None = None) -> tuple[np.ndarray, dict]:
    """The instances in ``view`` that hold the hero's object: those that
    contain at least ``min_fraction`` of the hero surface ``points`` that land
    in the frame there. A text search can mask a different object of the same
    kind in a camera (counter, 24 Oct: "bread in plastic wrap" found another
    wrapped thing beside the fruit bowl in most cameras, not the bread on the
    cutting board); that instance holds none of the points and is dropped.
    With depth, points that lie more than ``occlusion_margin`` behind the
    rendered surface are hidden by something else and do not count (the
    margin should be about the object's size, so its own back side still
    counts). The bar is low on purpose: an instance that is only a part of
    the object (the stand under a bonsai) must stay, or the hull carves that
    part away; a mask a little too large costs nothing in a hull, and another
    object's instance holds next to none of the points. Returns the chosen
    union mask (empty when no instance holds the object, or too few points
    are in view) and what was decided."""
    from scipy import ndimage

    instances = np.asarray(instances, bool)
    if instances.ndim == 2:
        instances = instances[None]
    shape = np.asarray(view.silhouette).shape
    if instances.shape[1:] != shape:
        raise ValueError(f"instances for {view.view_id} do not match its mask size")
    points = np.asarray(points, np.float64)
    h, w = shape
    x, y, visible = project_points(points, view.viewmat, view.intrinsic, w, h)
    if view.depth is not None and view.alpha is not None and occlusion_margin is not None:
        vm = np.asarray(view.viewmat, np.float64)
        z = points @ vm[2, :3] + vm[2, 3]
        rendered = np.asarray(view.depth, np.float64)[y, x]
        opaque = np.asarray(view.alpha)[y, x] > 0.5
        visible &= ~(opaque & (rendered > 0) & (rendered < z - float(occlusion_margin)))
    count = int(visible.sum())
    info = {"instances": int(len(instances)), "visible_points": count, "fractions": [], "chosen": []}
    chosen = np.zeros(shape, bool)
    if count < int(min_visible):
        info["reason"] = "object not in view"
        return chosen, info
    radius = scaled_dilation(dilate_px, shape)
    vx, vy = x[visible], y[visible]
    for index, instance in enumerate(instances):
        rows, cols = np.nonzero(instance)
        if not len(rows):
            info["fractions"].append(0.0)
            continue
        # Within ``radius`` of the instance (a dilation by it), measured in
        # the instance's padded box only and at the points only.
        y0, x0 = max(int(rows.min()) - radius, 0), max(int(cols.min()) - radius, 0)
        y1, x1 = min(int(rows.max()) + radius + 1, h), min(int(cols.max()) + radius + 1, w)
        inside = (vy >= y0) & (vy < y1) & (vx >= x0) & (vx < x1)
        near = np.zeros(len(vx), bool)
        if inside.any():
            distance = ndimage.distance_transform_edt(~instance[y0:y1, x0:x1])
            near[inside] = distance[vy[inside] - y0, vx[inside] - x0] <= radius
        fraction = float(near.mean())
        info["fractions"].append(round(fraction, 4))
        if fraction >= float(min_fraction):
            chosen |= instance
            info["chosen"].append(index)
    if not info["chosen"]:
        info["reason"] = "no instance holds the object"
    return chosen, info


def object_depth_ranges(means: np.ndarray, object_ids: np.ndarray, views: list[HullView], *,
                        margin_fraction: float = 0.1) -> dict[str, tuple[float, float]]:
    """Per view, the camera-z range of the object (1st-99th percentile of its
    Gaussians, widened by ``margin_fraction`` of its size): what lies behind
    it (the wall) or well in front of it (clutter) is never the object."""
    points = np.asarray(means, np.float64)[np.asarray(object_ids, np.int64)]
    if len(points) == 0:
        return {}
    margin = float(margin_fraction) * float(np.ptp(points, axis=0).max())
    ranges = {}
    for view in views:
        vm = np.asarray(view.viewmat, np.float64)
        z = points @ vm[2, :3] + vm[2, 3]
        low, high = np.percentile(z, [1.0, 99.0])
        ranges[view.view_id] = (float(low) - margin, float(high) + margin)
    return ranges


def peel_round_selection(
    contributions: list[object],
    masks: list[np.ndarray],
    eligible: list[np.ndarray],
    size: int,
    *,
    min_inside_fraction: float,
    min_views: int,
    min_inside_weight: float,
    min_view_weight: float = 0.05,
) -> np.ndarray:
    """One peel round: Gaussians whose rendered weight (in the renders without
    what is already removed) falls mostly inside the object's masks, counting
    inside weight only where the Gaussian is at the object's depth in that
    view (``eligible[view][id]``)."""
    total = np.zeros(size)
    inside = np.zeros(size)
    views_in = np.zeros(size)
    for contribution, mask, ok_depth in zip(contributions, masks, eligible):
        ids = np.asarray(contribution.gaussian_ids)
        weights = np.asarray(contribution.weights, np.float64)
        valid = ids >= 0
        total += np.bincount(ids[valid], weights=weights[valid], minlength=size)[:size]
        hit = valid & np.broadcast_to(np.asarray(mask, bool)[..., None], ids.shape)
        in_view = np.bincount(ids[hit], weights=weights[hit], minlength=size)[:size] * np.asarray(ok_depth, bool)
        inside += in_view
        views_in += in_view >= float(min_view_weight)
    keep = ((inside >= float(min_inside_weight)) & (views_in >= int(min_views))
            & (inside >= float(min_inside_fraction) * np.maximum(total, 1e-12)))
    return np.flatnonzero(keep).astype(np.int64)


def peel_object_ids(
    renderer,
    means: np.ndarray,
    views: list[HullView],
    object_ids: np.ndarray,
    *,
    min_inside_fraction: float,
    min_views: int,
    min_inside_weight: float,
    erode_px: int = 2,
    depth_margin_fraction: float = 0.1,
    max_rounds: int = 12,
    stop_fraction: float = 0.001,
    dilate_px: int = 8,
    view_tolerance: float = 0.1,
) -> tuple[np.ndarray, list[int]]:
    """Peel what still draws the object after ``object_ids`` are removed:
    render every mask view without the removed set, take the Gaussians that
    now draw mostly inside the masks at the object's depth, remove them, and
    repeat until a round adds less than ``stop_fraction`` of the set. Layers
    the hull missed (a pot behind clutter, floaters) come off one by one.

    A peeled Gaussian must also lie inside the silhouettes of the views that
    see it (occlusion-aware, ``view_tolerance``): surfaces the removal reveals
    (a cloth draped under a stand) draw inside a mask once the object is gone,
    but other cameras see them outside the object."""
    from scipy import ndimage

    means = np.asarray(means, np.float64)
    size = len(means)
    removed = set(int(g) for g in np.asarray(object_ids).tolist())
    ranges = object_depth_ranges(means, np.asarray(object_ids, np.int64), views,
                                 margin_fraction=depth_margin_fraction)
    masks = [ndimage.binary_erosion(np.asarray(v.silhouette, bool), iterations=int(erode_px)) for v in views]
    eligible = []
    for view in views:
        vm = np.asarray(view.viewmat, np.float64)
        z = means @ vm[2, :3] + vm[2, 3]
        low, high = ranges[view.view_id]
        eligible.append((z >= low) & (z <= high))
    per_round = []
    for _ in range(int(max_rounds)):
        excluded = np.asarray(sorted(removed), np.int64)
        contributions = []
        for view in views:
            _img, _alpha, meta, _scene = renderer._render_raw(view.view_id, edited=False, excluded_gaussian_ids=excluded)
            contributions.append(renderer._contribution_from_meta(meta, blend=True))
        picked = peel_round_selection(contributions, masks, eligible, size, min_inside_fraction=min_inside_fraction,
                                      min_views=min_views, min_inside_weight=min_inside_weight)
        new = np.asarray([int(g) for g in picked.tolist() if int(g) not in removed], np.int64)
        if new.size:
            inside_all = silhouette_votes(means[new], views, dilate_px=dilate_px, view_tolerance=view_tolerance)[0]
            new = new[inside_all]
        new = new.tolist()
        per_round.append(len(new))
        removed |= set(new)
        if len(new) < float(stop_fraction) * max(len(removed), 1):
            break
    return np.asarray(sorted(removed), np.int64), per_round


def removal_residual(
    renderer,
    views: list[HullView],
    removed_ids: np.ndarray,
    *,
    erode_px: int = 4,
    in_front_depth: float = 0.06,
    same_depth: float = 0.03,
    same_colour: float = 0.06,
    min_visible_share: float = 0.2,
    min_in_frame: float = 0.8,
    world_up: np.ndarray | None = None,
) -> dict[str, object]:
    """G1: after removing ``removed_ids``, the share of each view's (eroded)
    object mask that did not change at all (same colour, same depth): the
    object is still there. What the removal revealed looks different (the
    cloth under a stand), so it does not count. The judged pixels are where
    the removed set was drawn or where the mask showed something at the
    object's depth; pixels where something was in front of the removed
    object (a mask overreaching onto clutter) are left out, and views where
    the object was mostly hidden (less than
    ``min_visible_share`` of its silhouette in front) or mostly out of the
    frame (less than ``min_in_frame`` of its Gaussians) are not judged.
    ``judged`` counts the views that were; with none, ``max`` is None (the
    caller must not pass the gate on missing evidence). ``at_depth`` is the
    share of the mask where the surface that was there is still there (same
    depth, any colour) and, with ``world_up``, above the support level."""
    from scipy import ndimage
    from .view_rank import in_frame_fraction

    excluded = np.asarray(removed_ids, np.int64)
    all_means = np.asarray(renderer._scene(False).means)
    object_means = all_means[excluded] if excluded.size else np.zeros((0, 3))
    ranges = object_depth_ranges(all_means, excluded, views) if excluded.size else {}
    per_view, at_depth, skipped = {}, {}, {}
    for view in views:
        mask = ndimage.binary_erosion(np.asarray(view.silhouette, bool), iterations=int(erode_px))
        if excluded.size == 0:
            skipped[view.view_id] = "nothing removed"
            continue
        if mask.sum() < 50 or view.depth is None:
            skipped[view.view_id] = "mask too small"
            continue
        h, w = mask.shape
        if in_frame_fraction(object_means, view.viewmat, view.intrinsic, w, h) < float(min_in_frame):
            skipped[view.view_id] = "object mostly out of frame"
            continue
        before_depth = np.asarray(view.depth)
        object_depth, object_alpha = renderer.render_only_depth(view.view_id, excluded, edited=False)
        projected = object_alpha > 0.5
        low, high = ranges.get(view.view_id, (0.0, 0.0))
        # Clutter in front of the removed set there (a mask overreaching onto
        # the pasta bag in front of the pot) is not the object.
        in_front = projected & (before_depth < object_depth * (1.0 - float(in_front_depth)))
        if projected.sum() < 50 or 1.0 - in_front.sum() / projected.sum() < float(min_visible_share):
            skipped[view.view_id] = "object mostly hidden"
            continue
        # Judged where the removed set itself was drawn, or where the mask
        # showed something at the object's depth (a part of the object the
        # hull missed entirely: the lower pot behind the clutter), less what
        # stood in front of it. A mask reaching far past the object (SAM's
        # mask of a low side view of the bulldozer covered half the table)
        # is cut to the object's depth range.
        at_object_depth = (before_depth >= low) & (before_depth <= high)
        mask &= (projected | at_object_depth) & ~in_front
        if mask.sum() < 50:
            skipped[view.view_id] = "mask too small"
            continue
        before = np.asarray(renderer.render(view.view_id, edited=False), np.float64) / 255.0
        after, after_alpha, depth_after = renderer.render_rgbd(view.view_id, edited=False, excluded_gaussian_ids=excluded)
        depth_after = np.asarray(depth_after)
        after = np.asarray(after, np.float64) / 255.0
        same = (mask & (np.abs(after - before).mean(axis=-1) < float(same_colour))
                & (np.abs(depth_after - before_depth) < float(same_depth) * np.maximum(before_depth, 1e-6)))
        per_view[view.view_id] = float(same.sum()) / float(mask.sum())
        # What still stands inside the mask after the removal: the surface
        # that was there before is still there (same depth, whatever its
        # colour: a blurred remnant no longer reads as "unchanged") and above
        # the support level (the lowest 5 % of the object's own pixels in this
        # view) by more than 5 % of its visible height. Revealed background
        # is farther; revealed support is at the support level.
        still = (mask & (np.asarray(after_alpha) > 0.5)
                 & (np.abs(depth_after - before_depth) < float(same_depth) * np.maximum(before_depth, 1e-6)))
        if world_up is not None:
            from .geometry_utils import unproject_depth
            up = np.asarray(world_up, np.float64)
            up = up / max(float(np.linalg.norm(up)), 1e-12)
            c2w = np.linalg.inv(np.asarray(view.viewmat, np.float64))
            height_before = unproject_depth(before_depth, view.intrinsic, c2w) @ up
            support = float(np.percentile(height_before[mask], 5.0))
            visible_height = max(float(np.percentile(height_before[mask], 95.0)) - support, 1e-9)
            height_after = unproject_depth(depth_after, view.intrinsic, c2w) @ up
            still &= height_after > support + 0.05 * visible_height
        at_depth[view.view_id] = float(still.sum()) / float(mask.sum())
    if not per_view:
        return {"views": {}, "at_depth": {}, "max": None, "mean": None, "at_depth_max": None,
                "worst_view": None, "judged": 0, "skipped": skipped}
    worst = max(per_view, key=per_view.get)
    worst_depth = max(at_depth, key=at_depth.get)
    return {"views": per_view, "at_depth": at_depth, "max": per_view[worst],
            "mean": float(np.mean(list(per_view.values()))), "worst_view": worst,
            "at_depth_max": at_depth[worst_depth], "at_depth_worst_view": worst_depth,
            "judged": len(per_view), "skipped": skipped}


def multiview_hull_object_ids(
    means: np.ndarray,
    views: list[HullView],
    **hull_kwargs,
) -> tuple[np.ndarray, dict[str, int]]:
    """The visual hull with every silhouette taking a turn as the candidate
    source (the others carve). Parts of the object one camera never saw
    (hidden from the hero) become candidates through the cameras that did."""
    found: set[int] = set()
    per_view: dict[str, int] = {}
    for view in views:
        others = [other for other in views if other.view_id != view.view_id]
        result = visual_hull_object_ids(
            means, hero_viewmat=view.viewmat, hero_intrinsic=view.intrinsic, hero_mask=view.silhouette,
            views=others, **hull_kwargs)
        ids = set(int(g) for g in result.object_ids.tolist())
        per_view[view.view_id] = len(ids - found)
        found |= ids
    return np.asarray(sorted(found), np.int64), per_view


def vote_mask(mask: np.ndarray, band_px: int) -> np.ndarray:
    """The object's mask as vote weights: 1 on it, 1/2 on a ``band_px`` band
    around it (given for a 512 px image), 0 elsewhere."""
    core = np.asarray(mask, bool)
    band = dilate_mask(core, scaled_dilation(band_px, core.shape)) & ~core
    return core.astype(np.float64) + 0.5 * band


def contribution_vote_object_ids(
    maps_and_masks: list[tuple[object, np.ndarray]],
    *,
    min_inside_fraction: float,
    min_views: int,
    min_inside_weight: float,
    min_view_weight: float = 0.05,
) -> np.ndarray:
    """Gaussians whose rendered weight falls mostly inside the object's masks.

    For every view, a Gaussian's weight in the view's contribution map
    (``gaussian_ids``/``weights``, [H,W,K]) is split into inside / outside the
    view's mask. The mask may carry weights in [0, 1] (``vote_mask``: 1 on the
    object, 1/2 on a thin band around it, for mask-edge error, instead of a
    dilation that pulled in the background next to the outline). Floaters
    that draw the object from some cameras but sit elsewhere in 3D fail the
    hull; they pass here, because what they draw is the object. Background
    seen beside the object in other views has most of its weight outside the
    masks and stays."""
    all_ids, all_total, all_inside, all_views = [], [], [], []
    for contribution, mask in maps_and_masks:
        ids = np.asarray(contribution.gaussian_ids)
        weights = np.asarray(contribution.weights, np.float64)
        mask_weight = np.asarray(mask, np.float64)
        valid = ids >= 0
        size = int(ids[valid].max()) + 1 if valid.any() else 0
        if size == 0:
            continue
        total = np.bincount(ids[valid], weights=weights[valid], minlength=size)
        weighted = weights * mask_weight[..., None]
        in_w = np.bincount(ids[valid], weights=weighted[valid], minlength=size)
        present = np.flatnonzero(total > 0)
        all_ids.append(present)
        all_total.append(total[present])
        all_inside.append(in_w[present])
        all_views.append((in_w[present] >= min_view_weight).astype(np.int64))
    if not all_ids:
        return np.zeros(0, np.int64)
    ids = np.concatenate(all_ids)
    size = int(ids.max()) + 1
    total = np.bincount(ids, weights=np.concatenate(all_total), minlength=size)
    inside = np.bincount(ids, weights=np.concatenate(all_inside), minlength=size)
    views = np.bincount(ids, weights=np.concatenate(all_views), minlength=size)
    keep = ((inside >= float(min_inside_weight)) & (views >= int(min_views))
            & (inside >= float(min_inside_fraction) * np.maximum(total, 1e-12)))
    return np.flatnonzero(keep).astype(np.int64)
