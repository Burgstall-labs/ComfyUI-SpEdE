# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class GaussianROI:
    gaussian_ids: np.ndarray
    confidences: np.ndarray
    halo_ids: np.ndarray

    def validate(self) -> "GaussianROI":
        if self.gaussian_ids.ndim != 1 or self.confidences.shape != self.gaussian_ids.shape:
            raise ValueError("ROI ids/confidences must be one-dimensional and aligned")
        return self


@dataclass(frozen=True)
class ContributionMap:
    gaussian_ids: np.ndarray  # [H,W,K], -1 for empty
    weights: np.ndarray       # [H,W,K]

    def validate(self) -> "ContributionMap":
        if self.gaussian_ids.ndim != 3:
            raise ValueError("gaussian_ids must have shape [H, W, K]")
        if self.weights.shape != self.gaussian_ids.shape:
            raise ValueError("weights must match gaussian_ids shape")
        if np.any(self.weights < 0) or not np.isfinite(self.weights).all():
            raise ValueError("contribution weights must be finite and non-negative")
        return self

    @classmethod
    def load(cls, path: Path) -> "ContributionMap":
        packed = np.load(path)
        return cls(
            gaussian_ids=packed["gaussian_ids"].astype(np.int64),
            weights=packed["weights"].astype(np.float32),
        ).validate()

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            gaussian_ids=self.gaussian_ids.astype(np.int64),
            weights=self.weights.astype(np.float32),
        )


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """The filled convex hull of a mask's pixels (the mask itself if degenerate)."""
    source = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(source)
    if len(xs) < 3:
        return source.copy()
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull, QhullError

    # Pixel corners, so the hull covers whole boundary pixels.
    corners = np.concatenate([
        np.stack([xs + dx, ys + dy], axis=1) for dx in (0, 1) for dy in (0, 1)
    ]).astype(np.float64)
    try:
        hull = ConvexHull(corners)
    except QhullError:
        return source.copy()
    polygon = [tuple(point) for point in corners[hull.vertices]]
    canvas = Image.new("L", (source.shape[1], source.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool) | source


def closed_mask(mask: np.ndarray, radius_fraction: float) -> np.ndarray:
    """Morphological closing with a disk of ``radius_fraction`` of the mask's
    larger extent: gaps and concavities narrower than the disk are filled, the
    outline elsewhere is kept. Between the mask and its convex hull."""
    from scipy import ndimage

    source = np.asarray(mask, dtype=bool)
    if not source.any():
        return source.copy()
    ys, xs = np.nonzero(source)
    extent = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    radius = int(np.ceil(float(radius_fraction) * extent))
    if radius <= 0:
        return source.copy()
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = yy * yy + xx * xx <= radius * radius
    padded = np.pad(source, radius)  # the frame edge must not erode the mask
    closed = ndimage.binary_closing(padded, structure=disk)[radius:-radius, radius:-radius]
    return closed | source


def grow_mask(mask: np.ndarray, margin_fraction: float) -> np.ndarray:
    """Grow a mask by ``margin_fraction`` of its larger extent (Euclidean)."""
    from scipy import ndimage

    source = np.asarray(mask, dtype=bool)
    if not source.any() or margin_fraction <= 0:
        return source.copy()
    ys, xs = np.nonzero(source)
    extent = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    radius = float(margin_fraction) * extent
    return ndimage.distance_transform_edt(~source) <= radius


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return source.copy()
    h, w = source.shape
    result = source.copy()
    for dy in range(-radius, radius + 1):
        y0, y1 = max(0, -dy), min(h, h - dy)
        for dx in range(-radius, radius + 1):
            x0, x1 = max(0, -dx), min(w, w - dx)
            result[y0:y1, x0:x1] |= source[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def _aggregate(
    contribution: ContributionMap,
    mask: np.ndarray,
    *,
    minimum_total_weight: float,
    minimum_mask_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a pixel mask using each Gaussian's in-mask share of visible mass.

    Confidence is not normalized against the strongest Gaussian. It is the
    fraction of that Gaussian's own rendered contribution that fell inside the
    mask in this view. This suppresses large/background Gaussians that only
    graze the mask while preserving small Gaussians genuinely contained by it.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != contribution.gaussian_ids.shape[:2]:
        raise ValueError("mask shape does not match contribution map")

    all_ids = contribution.gaussian_ids.reshape(-1)
    all_weights = contribution.weights.reshape(-1)
    all_valid = (all_ids >= 0) & (all_weights > 0)
    if not np.any(all_valid):
        return np.empty(0, np.int64), np.empty(0, np.float32)

    all_ids = all_ids[all_valid]
    all_weights = all_weights[all_valid]
    unique, inverse = np.unique(all_ids, return_inverse=True)
    visible_mass = np.zeros(len(unique), np.float64)
    np.add.at(visible_mass, inverse, all_weights)

    mask_ids = contribution.gaussian_ids[mask].reshape(-1)
    mask_weights = contribution.weights[mask].reshape(-1)
    mask_valid = (mask_ids >= 0) & (mask_weights > 0)
    in_mask_mass = np.zeros(len(unique), np.float64)
    if np.any(mask_valid):
        positions = np.searchsorted(unique, mask_ids[mask_valid])
        valid_positions = (
            (positions >= 0)
            & (positions < len(unique))
            & (unique[np.clip(positions, 0, len(unique) - 1)] == mask_ids[mask_valid])
        )
        np.add.at(
            in_mask_mass,
            positions[valid_positions],
            mask_weights[mask_valid][valid_positions],
        )

    fraction = in_mask_mass / np.maximum(visible_mass, 1e-12)
    keep = (
        (in_mask_mass >= minimum_total_weight)
        & (fraction >= float(minimum_mask_fraction))
    )
    return (
        unique[keep].astype(np.int64),
        np.clip(fraction[keep], 0.0, 1.0).astype(np.float32),
    )


def lift_roi(
    contribution: ContributionMap,
    mask: np.ndarray,
    *,
    halo_radius: int = 4,
    minimum_total_weight: float = 0.05,
    minimum_mask_fraction: float = 0.35,
) -> GaussianROI:
    contribution.validate()
    mask = np.asarray(mask, dtype=bool)
    ids, confidence = _aggregate(
        contribution,
        mask,
        minimum_total_weight=minimum_total_weight,
        minimum_mask_fraction=minimum_mask_fraction,
    )
    expanded = dilate_mask(mask, halo_radius)
    ring = expanded & ~mask
    halo, _ = _aggregate(
        contribution,
        ring,
        minimum_total_weight=minimum_total_weight,
        minimum_mask_fraction=0.10,
    )
    halo = np.setdiff1d(halo, ids, assume_unique=False)
    return GaussianROI(ids, confidence, halo).validate()


def mask_for_gaussians(
    contribution: ContributionMap,
    gaussian_ids: np.ndarray,
    *,
    minimum_weight: float = 0.01,
    minimum_fraction: float | None = None,
) -> np.ndarray:
    """Project a 3D Gaussian set into a 2D edit mask.

    A pixel qualifies when the set contributes at least ``minimum_weight``
    and, if ``minimum_fraction`` is given, at least that share of the pixel's
    total recorded contribution, i.e. the set dominates the pixel rather than
    just grazing it with a Gaussian's faint tail.
    """
    ids = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.zeros(contribution.gaussian_ids.shape[:2], dtype=bool)
    selected = np.isin(contribution.gaussian_ids, ids)
    weight = np.where(selected, contribution.weights, 0.0).sum(axis=-1)
    mask = weight >= float(minimum_weight)
    if minimum_fraction is not None:
        total = np.where(contribution.gaussian_ids >= 0, contribution.weights, 0.0).sum(axis=-1)
        mask &= weight >= float(minimum_fraction) * np.maximum(total, 1e-12)
    return mask
