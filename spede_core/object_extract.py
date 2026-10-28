# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
"""Extract a selected object from a Gaussian-splat scene without changing it.

The multi-view selector already produces the hard part: source-scene Gaussian
IDs that consistently render inside the object's semantic masks.  Extraction is
therefore an inversion of REMOVE: keep those source rows exactly and discard
everything else.

No optimisation, colour conversion or SH truncation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class ExtractionMetrics:
    view_id: str
    mask_pixels: int
    rendered_pixels: int
    mask_coverage: float
    outside_alpha_fraction: float
    outside_rendered_fraction: float
    occluded_pixels: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "mask_pixels": self.mask_pixels,
            "rendered_pixels": self.rendered_pixels,
            "mask_coverage": self.mask_coverage,
            "outside_alpha_fraction": self.outside_alpha_fraction,
            "outside_rendered_fraction": self.outside_rendered_fraction,
            "occluded_pixels": self.occluded_pixels,
        }


def write_object_subset(source_ply: Path, object_ids: np.ndarray, output_ply: Path) -> int:
    """Write the selected source rows byte-for-property, preserving all fields.

    Graphdeco extras, SH bands and any future per-Gaussian properties survive
    because the structured vertex rows are copied directly instead of rebuilt
    through the normal refinement writer.
    """
    source_ply = Path(source_ply)
    output_ply = Path(output_ply)
    ply = PlyData.read(str(source_ply))
    vertices = ply["vertex"].data
    ids = np.unique(np.asarray(object_ids, np.int64).reshape(-1))
    if ids.size == 0:
        raise ValueError("object_ids is empty")
    if int(ids.min()) < 0 or int(ids.max()) >= len(vertices):
        raise ValueError(
            f"object_ids must be source rows in [0, {len(vertices) - 1}]"
        )
    subset = vertices[ids].copy()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(subset, "vertex")],
        text=False,
        byte_order="<",
        comments=list(ply.comments) + [
            f"SpEdE object extract: kept {len(ids)} of {len(vertices)} source Gaussians"
        ],
    ).write(output_ply)
    return int(len(ids))


def occlusion_aware_alpha(
    object_depth: np.ndarray,
    object_alpha: np.ndarray,
    source_depth: np.ndarray,
    source_alpha: np.ndarray,
    *,
    alpha_threshold: float = 0.05,
    depth_tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide object-only pixels that a nearer source-scene surface occluded."""
    object_depth = np.asarray(object_depth, np.float64)
    object_alpha = np.asarray(object_alpha, np.float32)
    source_depth = np.asarray(source_depth, np.float64)
    source_alpha = np.asarray(source_alpha, np.float32)
    if not (
        object_depth.shape == object_alpha.shape == source_depth.shape == source_alpha.shape
    ):
        raise ValueError("object/source depth and alpha maps must have the same shape")
    hidden = (
        np.isfinite(object_depth)
        & (object_depth > 0)
        & np.isfinite(source_depth)
        & (source_depth > 0)
        & (source_alpha >= float(alpha_threshold))
        & (
            object_depth
            > source_depth * (1.0 + float(depth_tolerance))
        )
    )
    judged = object_alpha.copy()
    judged[hidden] = 0.0
    return judged, hidden


def extraction_view_metrics(
    alpha: np.ndarray,
    mask: np.ndarray,
    *,
    alpha_threshold: float = 0.05,
) -> ExtractionMetrics:
    """Measure semantic completeness and contamination of an object-only render.

    Coverage is intentionally a pixel measure and therefore only a diagnostic
    for thin/open objects such as motorcycles.  Contamination uses alpha mass as
    the primary quantity so a few faint splats do not dominate the score.
    """
    alpha = np.asarray(alpha, np.float64)
    mask = np.asarray(mask, bool)
    if alpha.shape != mask.shape:
        raise ValueError("alpha and mask must have the same shape")
    visible = np.isfinite(alpha) & (alpha >= float(alpha_threshold))
    mask_pixels = int(mask.sum())
    rendered_pixels = int(visible.sum())
    coverage = (
        float((visible & mask).sum() / mask_pixels)
        if mask_pixels
        else 0.0
    )
    total_alpha = float(np.sum(np.clip(alpha, 0.0, 1.0)))
    outside_alpha = float(np.sum(np.clip(alpha[~mask], 0.0, 1.0)))
    outside_alpha_fraction = outside_alpha / max(total_alpha, 1e-12)
    outside_rendered_fraction = (
        float((visible & ~mask).sum() / rendered_pixels)
        if rendered_pixels
        else 0.0
    )
    return ExtractionMetrics(
        view_id="",
        mask_pixels=mask_pixels,
        rendered_pixels=rendered_pixels,
        mask_coverage=coverage,
        outside_alpha_fraction=outside_alpha_fraction,
        outside_rendered_fraction=outside_rendered_fraction,
    )


def summarize_extraction_metrics(
    metrics: list[ExtractionMetrics],
    *,
    minimum_mask_coverage: float,
    maximum_outside_alpha_fraction: float,
) -> dict[str, object]:
    """K1 extraction gate/report over all cameras that supplied object masks."""
    if not metrics:
        return {
            "name": "K1 object extraction",
            "passed": False,
            "reason": "no segmented views were available to judge extraction",
            "judged_views": 0,
        }
    worst_coverage = min(metrics, key=lambda m: m.mask_coverage)
    worst_contamination = max(metrics, key=lambda m: m.outside_alpha_fraction)
    passed = (
        worst_coverage.mask_coverage >= float(minimum_mask_coverage)
        and worst_contamination.outside_alpha_fraction
        <= float(maximum_outside_alpha_fraction)
    )
    reasons = []
    if worst_coverage.mask_coverage < float(minimum_mask_coverage):
        reasons.append(
            f"mask coverage {worst_coverage.mask_coverage:.3f} in "
            f"{worst_coverage.view_id} < {minimum_mask_coverage:.3f}"
        )
    if worst_contamination.outside_alpha_fraction > float(maximum_outside_alpha_fraction):
        reasons.append(
            f"outside alpha {worst_contamination.outside_alpha_fraction:.3f} in "
            f"{worst_contamination.view_id} > {maximum_outside_alpha_fraction:.3f}"
        )
    return {
        "name": "K1 object extraction",
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else "object is covered and contamination is within threshold",
        "judged_views": len(metrics),
        "minimum_mask_coverage": float(worst_coverage.mask_coverage),
        "coverage_worst_view": worst_coverage.view_id,
        "maximum_outside_alpha_fraction": float(worst_contamination.outside_alpha_fraction),
        "contamination_worst_view": worst_contamination.view_id,
        "mean_mask_coverage": float(np.mean([m.mask_coverage for m in metrics])),
        "mean_outside_alpha_fraction": float(np.mean([m.outside_alpha_fraction for m in metrics])),
        "thresholds": {
            "minimum_mask_coverage": float(minimum_mask_coverage),
            "maximum_outside_alpha_fraction": float(maximum_outside_alpha_fraction),
        },
    }


def save_extraction_report(path: Path, report: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def extract_object(
    engine,
    *,
    object_ids_path: Path,
    segment_report: Path,
    hero_mask_path: Path,
    out_dir: Path,
    heldout_targets: Path | None = None,
    alpha_threshold: float = 0.05,
    minimum_mask_coverage: float = 0.55,
    maximum_outside_alpha_fraction: float = 0.08,
    grid_views: int = 6,
    occlusion_depth_tolerance: float = 0.02,
    surface_rescue: bool = False,
    surface_rescue_neighbor_count: int = 32,
    surface_rescue_minimum_neighbors: int = 12,
    surface_rescue_neighbor_radius_fraction: float = 0.08,
    surface_rescue_plane_scale_tolerance: float = 3.0,
    surface_rescue_plane_span_tolerance: float = 0.12,
    surface_rescue_maximum_planarity_ratio: float = 0.20,
    surface_rescue_envelope_dilation_px: int = 3,
    surface_rescue_minimum_support_views: int = 2,
    surface_rescue_maximum_contradiction_views: int = 0,
    surface_rescue_minimum_camera_baseline_deg: float = 5.0,
    surface_rescue_maximum_cameras: int = 32,
    surface_rescue_minimum_visible_mass: float = 0.01,
    depth_clean: bool = False,
    depth_clean_relative_tolerance: float = 0.005,
    depth_clean_scale_tolerance: float = 3.0,
    depth_clean_minimum_alpha: float = 0.20,
    depth_clean_minimum_front_views: int = 2,
    depth_clean_maximum_support_views: int = 0,
    depth_clean_remove_never_visible: bool = False,
    depth_clean_minimum_visible_mass: float = 0.01,
    depth_clean_minimum_camera_baseline_deg: float = 5.0,
    depth_clean_maximum_cameras: int = 48,
    support_trim: bool = True,
    support_trim_band_mads: float = 3.0,
    support_trim_band_scales: float = 3.0,
    support_trim_maximum_band_fraction: float = 0.04,
    support_trim_colour_ratio: float = 1.0,
    floater_cleanup: bool = True,
    floater_cleanup_cell_fraction: float = 0.02,
    floater_cleanup_maximum_component_fraction: float = 0.05,
    floater_cleanup_minimum_gap_fraction: float = 0.08,
) -> dict[str, object]:
    """Export an object-only PLY and judge it against the multiview masks."""
    from PIL import Image
    from .ply import GaussianPly
    from .renderer import CameraSet, GsplatRenderer
    from .visual_hull import HullView, hull_views_from_segment_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(object_ids_path)) as packed:
        if "object_ids" not in packed:
            raise ValueError("object-ids NPZ has no object_ids array")
        ids = np.unique(np.asarray(packed["object_ids"], np.int64))

    source = GaussianPly.load(Path(engine.manifest.source_ply))
    ids = ids[(ids >= 0) & (ids < len(source.vertex))]
    if ids.size == 0:
        raise ValueError("no valid object Gaussians to extract")

    report = json.loads(Path(segment_report).read_text())
    hero_id = report["hero_view_id"]
    voting_ids = [hero_id, *report.get("voting_views", [])]
    voting_ids = list(dict.fromkeys(voting_ids))
    hero_mask = np.asarray(Image.open(hero_mask_path).convert("L")) > 127
    masks_by_view: dict[str, np.ndarray] = {hero_id: hero_mask}
    for view_id in voting_ids:
        if view_id == hero_id:
            continue
        view_record = (report.get("views") or {}).get(view_id) or {}
        mask_path = view_record.get("mask_path")
        if mask_path and Path(mask_path).exists():
            masks_by_view[view_id] = (
                np.asarray(Image.open(mask_path).convert("L")) > 127
            )

    semantic_ids = ids.copy()
    semantic_ply = None
    rescued_ply = None
    rescue_report: dict[str, object] = {"enabled": False}
    rescue_evidence_path = None
    rescued_ids = semantic_ids.copy()

    if surface_rescue:
        from .surface_rescue import collect_surface_rescue_evidence

        semantic_ply = out_dir / "object_semantic.ply"
        write_object_subset(
            Path(engine.manifest.source_ply), semantic_ids, semantic_ply
        )
        rescue_views = [
            engine.cameras.by_id[view_id]
            for view_id in voting_ids
            if view_id in masks_by_view
            and view_id in engine.cameras.by_id
            and not engine.cameras.by_id[view_id].is_virtual
            and not engine.cameras.by_id[view_id].synthetic
        ]
        rescue_evidence, rescue_report = collect_surface_rescue_evidence(
            source,
            semantic_ids,
            rescue_views,
            masks_by_view,
            engine.renderer,
            neighbor_count=surface_rescue_neighbor_count,
            minimum_neighbors=surface_rescue_minimum_neighbors,
            neighbor_radius_fraction=surface_rescue_neighbor_radius_fraction,
            plane_scale_tolerance=surface_rescue_plane_scale_tolerance,
            plane_span_tolerance=surface_rescue_plane_span_tolerance,
            maximum_planarity_ratio=surface_rescue_maximum_planarity_ratio,
            envelope_dilation_px_512=surface_rescue_envelope_dilation_px,
            minimum_visible_mass=surface_rescue_minimum_visible_mass,
            minimum_support_views=surface_rescue_minimum_support_views,
            maximum_contradiction_views=surface_rescue_maximum_contradiction_views,
            minimum_camera_baseline_deg=surface_rescue_minimum_camera_baseline_deg,
            maximum_cameras=surface_rescue_maximum_cameras,
        )
        rescued_ids = np.union1d(semantic_ids, rescue_evidence.rescued_ids)
        rescued_ply = out_dir / "object_rescued.ply"
        write_object_subset(
            Path(engine.manifest.source_ply), rescued_ids, rescued_ply
        )
        rescue_evidence_path = out_dir / "surface_rescue_evidence.npz"
        np.savez_compressed(
            rescue_evidence_path,
            candidate_ids=rescue_evidence.candidate_ids,
            local_plane_distance=rescue_evidence.local_plane_distance,
            local_plane_tolerance=rescue_evidence.local_plane_tolerance,
            planarity_ratio=rescue_evidence.planarity_ratio,
            local_manifold_rank=rescue_evidence.local_manifold_rank,
            envelope_support_views=rescue_evidence.envelope_support_views,
            raw_semantic_support_views=rescue_evidence.raw_semantic_support_views,
            contradiction_views=rescue_evidence.contradiction_views,
            visible_views=rescue_evidence.visible_views,
            rescue=rescue_evidence.rescue,
            rescued_source_ids=rescue_evidence.rescued_ids,
        )

    ids = rescued_ids.copy()
    depth_input_ids = ids.copy()
    depth_report = {"enabled": False}
    depth_evidence_path = None
    if depth_clean:
        from .depth_clean import collect_depth_clean_evidence

        if semantic_ply is None:
            semantic_ply = out_dir / "object_semantic.ply"
            write_object_subset(
                Path(engine.manifest.source_ply), semantic_ids, semantic_ply
            )
        depth_views = [
            engine.cameras.by_id[view_id]
            for view_id in voting_ids
            if view_id in engine.cameras.by_id
            and not engine.cameras.by_id[view_id].is_virtual
            and not engine.cameras.by_id[view_id].synthetic
        ]
        evidence, depth_report = collect_depth_clean_evidence(
            source,
            depth_input_ids,
            depth_views,
            engine.renderer,
            relative_depth_tolerance=depth_clean_relative_tolerance,
            gaussian_scale_tolerance=depth_clean_scale_tolerance,
            minimum_depth_alpha=depth_clean_minimum_alpha,
            minimum_front_views=depth_clean_minimum_front_views,
            maximum_surface_support_views=depth_clean_maximum_support_views,
            remove_never_visible=depth_clean_remove_never_visible,
            minimum_visible_mass=depth_clean_minimum_visible_mass,
            minimum_camera_baseline_deg=depth_clean_minimum_camera_baseline_deg,
            maximum_cameras=depth_clean_maximum_cameras,
        )
        ids = evidence.kept_ids
        if ids.size == 0:
            raise RuntimeError("depth clean rejected every selected Gaussian")
        depth_evidence_path = out_dir / "depth_clean_evidence.npz"
        np.savez_compressed(
            depth_evidence_path,
            source_ids=evidence.source_ids,
            surface_support_views=evidence.surface_support_views,
            front_violation_views=evidence.front_violation_views,
            depth_judged_views=evidence.depth_judged_views,
            visible_views=(
                np.asarray([], np.int16)
                if evidence.visible_views is None
                else evidence.visible_views
            ),
            remove=evidence.remove,
            kept_source_ids=evidence.kept_ids,
            removed_source_ids=evidence.removed_ids,
        )

    # The selection is REMOVE's and takes a little of the table under the
    # object with it (its contact shadow); an extracted object must not.
    trim_input_ids = ids.copy()
    trim_report: dict[str, object] = {"enabled": False}
    trim_evidence_path = None
    if support_trim:
        from .geometry_utils import world_up_from_cameras
        from .support_trim import support_contact_trim

        world_up = world_up_from_cameras([
            np.linalg.inv(np.asarray(view.viewmat, np.float64))
            for view in engine.cameras.views
            if not view.is_virtual
        ])
        ids, trim_report = support_contact_trim(
            source.means,
            source.log_scales,
            source.dc,
            source.opacity_logits,
            trim_input_ids,
            world_up,
            band_mads=support_trim_band_mads,
            band_scales=support_trim_band_scales,
            maximum_band_fraction=support_trim_maximum_band_fraction,
            colour_ratio=support_trim_colour_ratio,
        )
        trim_evidence_path = out_dir / "support_trim_evidence.npz"
        np.savez_compressed(
            trim_evidence_path,
            input_source_ids=trim_input_ids,
            kept_source_ids=ids,
            removed_source_ids=np.setdiff1d(trim_input_ids, ids),
        )

    # The masks can also take a piece of something else that projects inside
    # them from every voting view (truck: a railing below and beside it);
    # small clusters clearly apart from the object go.
    floater_input_ids = ids.copy()
    floater_report: dict[str, object] = {"enabled": False}
    floater_evidence_path = None
    if floater_cleanup:
        from .floater_cleanup import drop_detached_floaters

        ids, floater_report = drop_detached_floaters(
            source.means,
            floater_input_ids,
            cell_fraction=floater_cleanup_cell_fraction,
            maximum_component_fraction=floater_cleanup_maximum_component_fraction,
            minimum_gap_fraction=floater_cleanup_minimum_gap_fraction,
        )
        if ids.size == 0:
            raise RuntimeError("floater cleanup rejected every selected Gaussian")
        floater_evidence_path = out_dir / "floater_cleanup_evidence.npz"
        np.savez_compressed(
            floater_evidence_path,
            input_source_ids=floater_input_ids,
            kept_source_ids=ids,
            removed_source_ids=np.setdiff1d(floater_input_ids, ids),
        )

    output_ply = out_dir / "object_only.ply"
    kept = write_object_subset(Path(engine.manifest.source_ply), ids, output_ply)

    hero_camera = engine.cameras.by_id[hero_id]
    views = hull_views_from_segment_report(
        report, engine.cameras, exclude={hero_id}, renderer=None
    )
    judged = [
        HullView(hero_id, hero_camera.viewmat, hero_camera.intrinsic, hero_mask),
        *views,
    ]

    renderer = GsplatRenderer(
        source_ply=output_ply,
        cameras=engine.cameras,
        device=str(engine.renderer.device),
    )
    semantic_renderer = (
        None
        if semantic_ply is None
        else GsplatRenderer(
            source_ply=semantic_ply,
            cameras=engine.cameras,
            device=str(engine.renderer.device),
        )
    )
    rescued_renderer = (
        None
        if rescued_ply is None or not depth_clean
        else GsplatRenderer(
            source_ply=rescued_ply,
            cameras=engine.cameras,
            device=str(engine.renderer.device),
        )
    )
    metrics: list[ExtractionMetrics] = []
    semantic_metrics: list[ExtractionMetrics] = []
    rescued_metrics: list[ExtractionMetrics] = []
    rendered_by_view: dict[str, np.ndarray] = {}
    semantic_rendered_by_view: dict[str, np.ndarray] = {}
    rescued_rendered_by_view: dict[str, np.ndarray] = {}
    for view in judged:
        rgb, object_alpha, object_depth = renderer.render_rgbd(
            view.view_id, edited=False
        )
        _source_rgb, source_alpha, source_depth = engine.renderer.render_rgbd(
            view.view_id, edited=False
        )
        judged_alpha, hidden = occlusion_aware_alpha(
            object_depth,
            object_alpha,
            source_depth,
            source_alpha,
            alpha_threshold=alpha_threshold,
            depth_tolerance=occlusion_depth_tolerance,
        )
        rendered_by_view[view.view_id] = rgb
        if semantic_renderer is not None:
            semantic_rgb, semantic_alpha, semantic_depth = semantic_renderer.render_rgbd(
                view.view_id, edited=False
            )
            semantic_judged_alpha, semantic_hidden = occlusion_aware_alpha(
                semantic_depth,
                semantic_alpha,
                source_depth,
                source_alpha,
                alpha_threshold=alpha_threshold,
                depth_tolerance=occlusion_depth_tolerance,
            )
            semantic_rendered_by_view[view.view_id] = semantic_rgb
            semantic_item = extraction_view_metrics(
                semantic_judged_alpha,
                view.silhouette,
                alpha_threshold=alpha_threshold,
            )
            semantic_metrics.append(
                ExtractionMetrics(
                    view_id=view.view_id,
                    mask_pixels=semantic_item.mask_pixels,
                    rendered_pixels=semantic_item.rendered_pixels,
                    mask_coverage=semantic_item.mask_coverage,
                    outside_alpha_fraction=semantic_item.outside_alpha_fraction,
                    outside_rendered_fraction=semantic_item.outside_rendered_fraction,
                    occluded_pixels=int(semantic_hidden.sum()),
                )
            )
        if rescued_renderer is not None:
            rescued_rgb, rescued_alpha, rescued_depth = rescued_renderer.render_rgbd(
                view.view_id, edited=False
            )
            rescued_judged_alpha, rescued_hidden = occlusion_aware_alpha(
                rescued_depth,
                rescued_alpha,
                source_depth,
                source_alpha,
                alpha_threshold=alpha_threshold,
                depth_tolerance=occlusion_depth_tolerance,
            )
            rescued_rendered_by_view[view.view_id] = rescued_rgb
            rescued_item = extraction_view_metrics(
                rescued_judged_alpha,
                view.silhouette,
                alpha_threshold=alpha_threshold,
            )
            rescued_metrics.append(
                ExtractionMetrics(
                    view_id=view.view_id,
                    mask_pixels=rescued_item.mask_pixels,
                    rendered_pixels=rescued_item.rendered_pixels,
                    mask_coverage=rescued_item.mask_coverage,
                    outside_alpha_fraction=rescued_item.outside_alpha_fraction,
                    outside_rendered_fraction=rescued_item.outside_rendered_fraction,
                    occluded_pixels=int(rescued_hidden.sum()),
                )
            )
        item = extraction_view_metrics(
            judged_alpha, view.silhouette, alpha_threshold=alpha_threshold
        )
        metrics.append(
            ExtractionMetrics(
                view_id=view.view_id,
                mask_pixels=item.mask_pixels,
                rendered_pixels=item.rendered_pixels,
                mask_coverage=item.mask_coverage,
                outside_alpha_fraction=item.outside_alpha_fraction,
                outside_rendered_fraction=item.outside_rendered_fraction,
                occluded_pixels=int(hidden.sum()),
            )
        )

    k1 = summarize_extraction_metrics(
        metrics,
        minimum_mask_coverage=minimum_mask_coverage,
        maximum_outside_alpha_fraction=maximum_outside_alpha_fraction,
    )
    semantic_k1 = (
        None
        if not semantic_metrics
        else summarize_extraction_metrics(
            semantic_metrics,
            minimum_mask_coverage=minimum_mask_coverage,
            maximum_outside_alpha_fraction=maximum_outside_alpha_fraction,
        )
    )
    rescued_k1 = (
        None
        if not rescued_metrics
        else summarize_extraction_metrics(
            rescued_metrics,
            minimum_mask_coverage=minimum_mask_coverage,
            maximum_outside_alpha_fraction=maximum_outside_alpha_fraction,
        )
    )
    if surface_rescue and semantic_k1 is not None:
        rescue_after = rescued_k1 if rescued_k1 is not None else k1
        rescue_report["k1_before"] = semantic_k1
        rescue_report["k1_after"] = rescue_after
        rescue_report["coverage_delta"] = float(
            rescue_after["minimum_mask_coverage"] - semantic_k1["minimum_mask_coverage"]
        )
        rescue_report["outside_alpha_delta"] = float(
            rescue_after["maximum_outside_alpha_fraction"]
            - semantic_k1["maximum_outside_alpha_fraction"]
        )
    if depth_clean:
        depth_before = rescued_k1 if rescued_k1 is not None else semantic_k1
        if depth_before is not None:
            depth_report["k1_before"] = depth_before
            depth_report["k1_after"] = k1
            depth_report["coverage_delta"] = float(
                k1["minimum_mask_coverage"] - depth_before["minimum_mask_coverage"]
            )
            depth_report["outside_alpha_delta"] = float(
                k1["maximum_outside_alpha_fraction"]
                - depth_before["maximum_outside_alpha_fraction"]
            )

    proof = None
    if judged:
        picks = np.linspace(
            0, len(judged) - 1, min(int(grid_views), len(judged))
        ).round().astype(int).tolist()
        by_id = {view.view_id: index for index, view in enumerate(judged)}
        for key in ("coverage_worst_view", "contamination_worst_view"):
            worst_id = k1.get(key)
            if worst_id in by_id:
                picks.append(by_id[worst_id])
        rows = []
        for index in dict.fromkeys(picks):
            view = judged[index]
            source_rgb = engine.renderer.render(view.view_id, edited=False)
            object_rgb = rendered_by_view[view.view_id]
            overlay = source_rgb.astype(np.float32)
            overlay[view.silhouette] = (
                0.45 * overlay[view.silhouette]
                + 0.55 * np.asarray([215.0, 38.0, 61.0], np.float32)
            )
            panels = [source_rgb]
            if semantic_renderer is not None:
                panels.append(semantic_rendered_by_view[view.view_id])
            if rescued_renderer is not None:
                panels.append(rescued_rendered_by_view[view.view_id])
            panels.extend([object_rgb, overlay.astype(np.uint8)])
            rows.append(np.concatenate(panels, axis=1))
        proof = out_dir / "extract_sheet.png"
        Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(proof)

    heldout_sheet = None
    if heldout_targets is not None:
        cameras = CameraSet.from_camera_pack(Path(heldout_targets))
        heldout_renderer = GsplatRenderer(
            source_ply=output_ply,
            cameras=cameras,
            device=str(engine.renderer.device),
        )
        heldout_ids = [view.view_id for view in cameras.views]
        if heldout_ids:
            picks = np.linspace(
                0, len(heldout_ids) - 1, min(int(grid_views), len(heldout_ids))
            ).round().astype(int)
            rows = [
                heldout_renderer.render(heldout_ids[index], edited=False)
                for index in dict.fromkeys(picks.tolist())
            ]
            heldout_sheet = out_dir / "extract_heldout.png"
            Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(
                heldout_sheet
            )

    extent = np.ptp(source.means[ids], axis=0)
    result = {
        "status": "object_extracted",
        "source_ply": str(Path(engine.manifest.source_ply)),
        "object_ply": str(output_ply),
        "source_gaussians": int(len(source.vertex)),
        "semantic_object_gaussians": int(len(semantic_ids)),
        "surface_rescued_gaussians": int(len(rescued_ids) - len(semantic_ids)),
        "rescued_object_gaussians": int(len(rescued_ids)),
        "object_gaussians": int(kept),
        "removed_gaussians": int(len(source.vertex) - kept),
        "depth_clean_removed_gaussians": int(len(depth_input_ids) - len(trim_input_ids)),
        "support_trim_removed_gaussians": int(len(trim_input_ids) - len(floater_input_ids)),
        "floater_cleanup_removed_gaussians": int(len(floater_input_ids) - kept),
        "kept_fraction": float(kept / max(len(source.vertex), 1)),
        "semantic_ply": None if semantic_ply is None else str(semantic_ply),
        "rescued_ply": None if rescued_ply is None else str(rescued_ply),
        "surface_rescue": rescue_report,
        "surface_rescue_evidence": (
            None if rescue_evidence_path is None else str(rescue_evidence_path)
        ),
        "depth_clean": depth_report,
        "depth_clean_evidence": (
            None if depth_evidence_path is None else str(depth_evidence_path)
        ),
        "support_trim": trim_report,
        "support_trim_evidence": (
            None if trim_evidence_path is None else str(trim_evidence_path)
        ),
        "floater_cleanup": floater_report,
        "floater_cleanup_evidence": (
            None if floater_evidence_path is None else str(floater_evidence_path)
        ),
        "bbox_min": [float(v) for v in source.means[ids].min(axis=0)],
        "bbox_max": [float(v) for v in source.means[ids].max(axis=0)],
        "extent": [float(v) for v in extent],
        "k1": k1,
        "k1_semantic": semantic_k1,
        "k1_after_surface_rescue": (
            rescued_k1 if rescued_k1 is not None else (k1 if surface_rescue else None)
        ),
        "k1_before_depth_clean": (
            rescued_k1 if rescued_k1 is not None else semantic_k1
        ),
        "views": [item.as_dict() for item in metrics],
        "proof": None if proof is None else str(proof),
        "heldout_sheet": None if heldout_sheet is None else str(heldout_sheet),
    }
    save_extraction_report(out_dir / "object_extract_report.json", result)
    return result
