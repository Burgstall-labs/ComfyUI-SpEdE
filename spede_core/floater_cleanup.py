# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drop small clusters that sit apart from an extracted object.

The multiview selection can hold a piece of something else that happens to
project inside the object's masks from every voting view. Truck (27 Oct):
the extracted truck carried a black railing piece (~6.5k of 435k Gaussians)
well below and beside it, separated from the body by open air.

The selection's Gaussian centres are binned in voxels a fraction of the
object's robust extent; 26-connected occupied voxels form the components.
The largest is the object. A component is dropped only when it is both
small (under a fraction of the largest) *and* clearly apart: its nearest
centre is further than a gap, again a fraction of the extent, from every
large component. Mirrors, wheels and lamp poles touch their body and stay in
its component; a thin part the voxels split off sits within the gap and
stays too.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _voxel_components(points: np.ndarray, cell: float) -> tuple[int, np.ndarray]:
    """26-connected components of the occupied voxels, per point.

    Sparse (unique voxels and their neighbour pairs), so a stray centre far
    out does not blow up a dense grid.
    """
    cells = np.floor((points - points.min(axis=0)) / cell).astype(np.int64)
    voxels, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    # Chebyshev distance 1 between voxel indices = the 26-neighbourhood.
    pairs = cKDTree(voxels).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(voxels), len(voxels))
    )
    count, label = connected_components(graph, directed=False)
    return int(count), label[inverse]


def drop_detached_floaters(
    means: np.ndarray,
    object_ids: np.ndarray,
    *,
    cell_fraction: float = 0.02,
    maximum_component_fraction: float = 0.05,
    minimum_gap_fraction: float = 0.08,
    minimum_gaussians: int = 100,
) -> tuple[np.ndarray, dict[str, object]]:
    """Selected rows to keep, and a report of the components and drops.

    ``means`` are the whole source scene's; ``object_ids`` index it.
    Lengths given as fractions are of the selection's robust diagonal (the
    1st-99th percentile box of its centres). The voxel is ``cell_fraction``
    of it. Components with at least ``maximum_component_fraction`` of the
    largest one's Gaussians are the object; a smaller one is dropped when
    its nearest centre is at least ``minimum_gap_fraction`` of the diagonal
    from theirs. Fewer than ``minimum_gaussians`` selected rows is a no-op.
    """
    means = np.asarray(means, np.float64)
    ids = np.unique(np.asarray(object_ids, np.int64))
    report: dict[str, object] = {"enabled": True, "input_gaussians": int(len(ids)),
                                 "removed_gaussians": 0, "applied": False}
    if len(ids) < int(minimum_gaussians):
        report["reason"] = "too few selected Gaussians"
        return ids, report
    points = means[ids]
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not np.isfinite(diagonal) or diagonal <= 0:
        report["reason"] = "degenerate object extent"
        return ids, report

    cell = float(cell_fraction) * diagonal
    count, label = _voxel_components(points, cell)
    sizes = np.bincount(label, minlength=count)
    largest = int(sizes.max())
    large = sizes >= float(maximum_component_fraction) * largest
    report.update({
        "diagonal": diagonal,
        "cell": cell,
        "components": count,
        "largest_component_gaussians": largest,
        "object_components": int(large.sum()),
    })
    small = np.flatnonzero(~large)
    if small.size == 0:
        report["reason"] = "one connected object"
        return ids, report

    # Gap of each small component to the object: its nearest centre's
    # distance to any centre of a large component.
    gap_limit = float(minimum_gap_fraction) * diagonal
    in_small = ~large[label]
    distance, _ = cKDTree(points[~in_small]).query(points[in_small])
    gap = np.full(count, np.inf)
    np.minimum.at(gap, label[in_small], distance)
    floater = np.zeros(count, bool)
    floater[small] = gap[small] >= gap_limit
    drop = floater[label]
    removed = ids[drop]
    kept = ids[~drop]

    dropped = np.flatnonzero(floater)
    dropped = dropped[np.argsort(sizes[dropped])[::-1]]
    report.update({
        "applied": True,
        "gap_limit": gap_limit,
        "removed_gaussians": int(len(removed)),
        "kept_gaussians": int(len(kept)),
        "removed_fraction": float(len(removed) / max(len(ids), 1)),
        "removed_components": int(len(dropped)),
        "kept_small_components": int(len(small) - len(dropped)),
        # The largest drops, for the report: where they were and how far out.
        "removed_clusters": [
            {
                "gaussians": int(sizes[c]),
                "gap": float(gap[c]),
                "centroid": [float(v) for v in points[label == c].mean(axis=0)],
                "bbox_min": [float(v) for v in points[label == c].min(axis=0)],
                "bbox_max": [float(v) for v in points[label == c].max(axis=0)],
            }
            for c in dropped[:10]
        ],
        "parameters": {
            "cell_fraction": float(cell_fraction),
            "maximum_component_fraction": float(maximum_component_fraction),
            "minimum_gap_fraction": float(minimum_gap_fraction),
            "minimum_gaussians": int(minimum_gaussians),
        },
    })
    return kept, report
