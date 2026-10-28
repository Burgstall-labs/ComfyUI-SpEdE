# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def footprint_column_ids(
    means: np.ndarray,
    object_ids: np.ndarray,
    world_up: np.ndarray,
    *,
    radius_fraction: float,
    base_fraction: float,
    min_neighbours: int = 3,
) -> np.ndarray:
    """Gaussians in the removed object's column that the hull missed.

    The column is the object's horizontal footprint (points within
    ``radius_fraction`` x object height of at least ``min_neighbours`` object
    Gaussians, measured in the plane perpendicular to world up), from
    ``base_fraction`` x object height above the object's base up to its top.
    Starting above the base keeps the support the object stood on (a stand's
    top) while the thin pieces of the object the hull rejected (the teapot's
    black shards) are caught. Base and top are the 1st and 99th height
    percentiles and a lone object Gaussian seeds nothing: a hundred strays
    in a 149k hull (one 3 units under the table) had made the column 4x the
    object's height and cut a disc out of the cloth around every stray.
    Returns sorted ids not already in ``object_ids``.
    """
    means = np.asarray(means, np.float64)
    ids = np.unique(np.asarray(object_ids, np.int64))
    if ids.size == 0:
        return ids
    up = np.asarray(world_up, np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    reference = np.eye(3)[int(np.argmin(np.abs(up)))]
    axis_a = np.cross(up, reference)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(up, axis_a)
    height = means @ up
    plane = np.stack([means @ axis_a, means @ axis_b], axis=1)
    low, high = (float(v) for v in np.percentile(height[ids], [1.0, 99.0]))
    extent = high - low
    if extent <= 0:
        return np.zeros(0, np.int64)
    in_band = (height >= low + float(base_fraction) * extent) & (height <= high)
    candidates = np.flatnonzero(in_band)
    candidates = candidates[~np.isin(candidates, ids)]
    if candidates.size == 0:
        return candidates
    # The radius follows the object's size, not only its height: a flat wide
    # object (a book, a rug) is not thin in the plane.
    footprint = float(np.linalg.norm(np.percentile(plane[ids], 99, axis=0) - np.percentile(plane[ids], 1, axis=0)))
    radius = float(radius_fraction) * max(extent, footprint)
    tree = cKDTree(plane[ids])
    k = max(1, int(min_neighbours))
    distance, _ = tree.query(plane[candidates], k=k)
    farthest = distance[:, -1] if k > 1 else distance
    return np.sort(candidates[farthest <= radius])


def stray_object_ids(means: np.ndarray, object_ids: np.ndarray, *, min_neighbours: int = 4,
                     stray_factor: float = 8.0) -> np.ndarray:
    """Members of a removal set far from the rest of it: those whose
    ``min_neighbours``-th nearest member is more than ``stray_factor`` x the
    set's median such distance away. Single Gaussians a vote or a carve let
    through are not the object; removing them cuts holes elsewhere. The
    median makes the test independent of the object's size and density."""
    ids = np.unique(np.asarray(object_ids, np.int64))
    if ids.size <= min_neighbours + 1:
        return np.zeros(0, np.int64)
    pts = np.asarray(means, np.float64)[ids]
    distance, _ = cKDTree(pts).query(pts, k=int(min_neighbours) + 1)
    reach = distance[:, -1]
    return np.sort(ids[reach > float(stray_factor) * max(float(np.median(reach)), 1e-12)])




def separate_cluster_ids(means: np.ndarray, object_ids: np.ndarray, anchor: np.ndarray | None = None,
                         *, link_factor: float = 8.0, near_fraction: float = 0.1) -> np.ndarray:
    """Members of a selection that form separate clusters away from the
    object: the selection split into pieces linked by gaps under
    ``link_factor`` x the median 8th-neighbour spacing, the main piece the
    one with the most ``anchor`` members (a per-Gaussian flag: those under
    the hero mask; the largest piece without one). Pieces nearer the main piece than
    ``near_fraction`` x its extent stay, as detached parts of the object.
    Playroom (24 Oct): the buggy's selection also held a toy truck and
    pictures metres away, each a dense cluster the stray test keeps."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    ids = np.unique(np.asarray(object_ids, np.int64))
    if ids.size < 10:
        return np.zeros(0, np.int64)
    pts = np.asarray(means, np.float64)[ids]
    tree = cKDTree(pts)
    spacing = float(np.median(tree.query(pts, k=9)[0][:, -1]))
    pairs = tree.query_pairs(link_factor * max(spacing, 1e-12), output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    count, label = connected_components(graph, directed=False)
    if count == 1:
        return np.zeros(0, np.int64)
    score = np.bincount(label).astype(np.float64)
    if anchor is not None:
        anchored = np.bincount(label, weights=np.asarray(anchor, bool)[ids].astype(np.float64), minlength=count)
        if anchored.any():
            score = anchored
    main = int(np.argmax(score))
    main_pts = pts[label == main]
    reach = near_fraction * float(np.ptp(main_pts, axis=0).max())
    gap, _ = cKDTree(main_pts).query(pts[label != main], distance_upper_bound=max(reach, 1e-12))
    # A piece stays or goes whole: one member near the object keeps it.
    near = np.union1d([main], label[label != main][np.isfinite(gap)])
    return np.sort(ids[~np.isin(label, near)])
