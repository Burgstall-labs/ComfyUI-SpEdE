"""AccGS-free multiview hull+contribution selection from the frozen algorithms."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image

from .ply import GaussianPly
from .geometry_utils import world_up_from_cameras
from .renderer import CameraSet, GsplatRenderer
from .visual_hull import (
    HullView, visual_hull_object_ids, hull_views_from_segment_report,
    consistent_views, multiview_hull_object_ids, contribution_vote_object_ids,
    peel_object_ids, removal_residual, vote_mask, project_points,
    hero_surface_ids, mask_instances, pick_hero_instances,
)
from .removal import stray_object_ids, footprint_column_ids, separate_cluster_ids


def _gate_residual(residual: dict, *, maximum: float = 0.12,
                   mean_maximum: float = 0.03, depth_maximum: float = 0.1):
    if not residual.get("judged") or residual.get("max") is None:
        return {"name": "G1 removal residual", "passed": False, "reason": "no real cameras judged",
                "raw": residual}
    issues = []
    if residual["max"] > maximum:
        issues.append("object unchanged in worst view")
    if residual["mean"] > mean_maximum:
        issues.append("too many unchanged object pixels on average")
    if residual["at_depth_max"] is not None and residual["at_depth_max"] > depth_maximum:
        issues.append("source surface remains at old object depth")
    return {
        "name":"G1 removal residual", "passed": not bool(issues),
        "reason": "; ".join(issues) or "OK", "raw":residual
    }


FAILED_NAME = "selection_failed.json"


def pick_hero_views(report: dict, means: np.ndarray, hero: HullView, views: list[HullView], *,
                    min_surface: int = 20) -> tuple[list[HullView], dict, np.ndarray]:
    """Each non-hero camera's mask reduced to the instances that hold the hero
    selection's surface (the hero's own mask stays as it is); a camera where
    no instance does is dropped like an empty mask. The instances are SAM's
    own (``instances_path``) or, for a union-only report, its connected
    pieces. Returns the kept views, what was decided per camera and the hero
    surface Gaussians."""
    surface = hero_surface_ids(means, hero.viewmat, hero.intrinsic, hero.silhouette,
                               hero.depth, hero.alpha)
    info = {"hero_surface_gaussians": int(len(surface)), "views": {}, "dropped": []}
    if len(surface) < min_surface:
        info["skipped"] = "too few Gaussians on the hero selection's surface"
        return list(views), info, surface
    points = np.asarray(means, np.float64)[surface]
    # A point further behind a camera's surface than the object is deep is
    # hidden by something else there; its own back side still counts.
    low, high = np.percentile(points, [5.0, 95.0], axis=0)
    margin = max(float(np.linalg.norm(high - low)), 1e-6)
    kept = []
    for view in views:
        row = report["views"].get(view.view_id) or {}
        if row.get("instances_path"):
            with np.load(row["instances_path"], allow_pickle=False) as f:
                instances = f["instances"]
            source = "sam"
        else:
            instances = mask_instances(view.silhouette)
            source = "pieces"
        chosen, pick = pick_hero_instances(points, view, instances, occlusion_margin=margin)
        pick["source"] = source
        info["views"][view.view_id] = pick
        if chosen.any():
            kept.append(replace(view, silhouette=chosen))
        else:
            info["dropped"].append(view.view_id)
    return kept, info, surface


def _write_picked_report(report: dict, report_path: Path, views: list[HullView], picks: dict,
                         out_dir: Path) -> Path:
    """The segment report with each camera's chosen instances (what later
    steps read as the selection's masks)."""
    masks = out_dir / "masks"
    masks.mkdir(exist_ok=True)
    hero = str(report["hero_view_id"])
    rows = {hero: report["views"][hero]}
    for index, view in enumerate(views):
        row = dict(report["views"][view.view_id])
        pick = picks["views"].get(view.view_id)
        if pick is not None and len(pick["chosen"]) != row.get("instances", 1):
            path = masks / f"mask_{index:04d}.png"
            Image.fromarray(np.asarray(view.silhouette, np.uint8) * 255, mode="L").save(path)
            row.update(mask_path=str(path), mask_area=int(np.asarray(view.silhouette).sum()),
                       source="hero-consistent instances")
        rows[view.view_id] = row
    picked = {**report, "views": rows, "voting_views": [v.view_id for v in views],
              "empty_views": list(report.get("empty_views") or []) + list(picks["dropped"]),
              "instance_pick": picks, "source_report": str(report_path)}
    path = out_dir / "segment_report.json"
    path.write_text(json.dumps(picked, indent=2) + "\n")
    return path


def collapse_message(name: str, reasons: list[str], dropped: int, cameras: int) -> str:
    """Why a selection that kept almost nothing of the clicked object failed,
    and what to do. No single quotes: ComfyUI's error text quotes with them."""
    named = f'The name "{name}"' if name else "The object name"
    example = f' (e.g. "{name} on the cutting board")' if name else ""
    share = "most" if 2 * dropped >= cameras else "some"
    return (f"Selection collapsed: {'; '.join(reasons)}. {named} finds other objects in {share} cameras "
            f"({dropped} of {cameras} cameras had no mask on the clicked object). Rename it to something "
            f"only the clicked object matches, its colour, material or what it sits on{example}, "
            "and run it again.")


def _fail(out_dir: Path, message: str, **numbers) -> None:
    (out_dir / FAILED_NAME).write_text(json.dumps({"message": message, **numbers}, indent=2) + "\n")
    raise RuntimeError(message)


def run_selection(source_ply: Path, target_npz: Path, segment_report: Path,
                  hero_mask: Path, out_dir: Path, *,
                  device: str = "cuda", footprint_cleanup: bool = False,
                  minimum_carving_views: int = 3,
                  use_contribution_vote: bool = True,
                  peel: bool = True, max_footprint_growth: float = 0.1,
                  min_surface_fraction: float = 0.25) -> dict:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FAILED_NAME).unlink(missing_ok=True)
    report_path = Path(segment_report).resolve(strict=True)
    report = json.loads(report_path.read_text())
    hero = str(report["hero_view_id"])
    cameras = CameraSet.from_camera_pack(Path(target_npz))
    if hero not in cameras.by_id:
        raise ValueError("Hero camera missing from NPZ")
    voting_ids = list(report.get("voting_views") or [])
    if len(voting_ids) != len(set(voting_ids)) or hero in voting_ids:
        raise ValueError("Voting views must be unique and exclude hero")
    if len(voting_ids) < minimum_carving_views:
        raise ValueError("At least three independent non-hero segmented cameras required")
    for name in voting_ids:
        if name not in cameras.by_id or cameras.by_id[name].is_virtual:
            raise ValueError(f"Invalid voting camera {name}")
    source = GaussianPly.load(Path(source_ply))
    renderer = GsplatRenderer(source_ply=source.path, cameras=cameras, device=device)
    with Image.open(hero_mask) as image:
        mask = np.asarray(image.convert("L")) > 127
    camera = cameras.by_id[hero]
    if mask.shape != (camera.height,camera.width) or not mask.any():
        raise ValueError("Hero mask mismatch or empty")
    other = hull_views_from_segment_report(
        report, cameras, exclude={hero}, renderer=renderer,
    )
    world_up = world_up_from_cameras(
        [np.linalg.inv(view.viewmat) for view in cameras.views if not view.is_virtual]
    )
    params = dict(dilate_px=8, view_tolerance=0.1,
                  min_seen_fraction=0.5, relaxed_view_tolerance=0.2, world_up=world_up)
    hero_depth, hero_alpha = renderer.render_depth(hero, edited=False)
    # The name's mask in a camera can be another object of the same kind
    # (counter, 24 Oct: 67 Gaussians of the hero hull survived the carving by
    # those masks and 98 were removed). Each camera keeps the instances that
    # hold the surface the hero mask selects; cameras without one do not vote.
    name = str(report.get("object_text") or "")
    segmented = len(other)
    other, picks, surface = pick_hero_views(
        report, source.means,
        HullView(hero, camera.viewmat, camera.intrinsic, mask, hero_depth, hero_alpha), other,
    )
    report_path = _write_picked_report(report, report_path, other, picks, out_dir)
    if len(other) < minimum_carving_views:
        _fail(out_dir, collapse_message(
            name, [f"only {len(other)} cameras have a mask on the clicked object, "
                   f"{minimum_carving_views} are needed"], segmented - len(other), segmented),
            cameras_kept=len(other), cameras_segmented=segmented)

    def select(other):
        result = visual_hull_object_ids(
            source.means, hero_viewmat=camera.viewmat, hero_intrinsic=camera.intrinsic,
            hero_mask=mask, views=other, **params,
        )
        all_views = [HullView(hero, camera.viewmat, camera.intrinsic, mask,
                              hero_depth, hero_alpha), *other]
        ids = np.asarray(result.object_ids, np.int64)
        multiview_ids, per_view = multiview_hull_object_ids(source.means, all_views, **params)
        ids = np.union1d(ids, multiview_ids)
        diagnostics = {
            "hero_hull":int(len(result.object_ids)),
            "multiview_additional":int(len(multiview_ids) - len(np.intersect1d(multiview_ids, result.object_ids))),
            "multiview_by_camera":per_view,
        }
        if use_contribution_vote:
            maps = [
                (renderer.contribution_map(v.view_id,edited=False,blend=True),
                 vote_mask(v.silhouette,8))
                for v in all_views
            ]
            voted = contribution_vote_object_ids(
                maps,min_inside_fraction=0.8,min_views=2,min_inside_weight=0.3
            )
            diagnostics["contribution_added"] = int(len(np.setdiff1d(voted,ids)))
            ids = np.union1d(ids,voted)
        strays = stray_object_ids(source.means,ids)
        ids = np.setdiff1d(ids,strays)
        diagnostics["strays_removed"]=int(len(strays))
        agreeing,agreement = consistent_views(renderer,all_views,ids)
        return result,all_views,ids,diagnostics,agreeing,agreement

    result,all_views,ids,diagnostics,agreeing,agreement = select(other)
    # A view whose mask is on something else carves the object away in the
    # hull. Playroom (23 Oct): the name "buggy" found the stairs, a desk and
    # a cushion in a third of the 63 views; the hull kept a fragment of the
    # buggy and the removal left the rest. Those views are dropped and the
    # selection redone with the views that agree with it.
    kept = {v.view_id for v in agreeing}
    rejected = [v.view_id for v in other if v.view_id not in kept]
    if rejected and len(other)-len(rejected) >= minimum_carving_views:
        first = {"object_gaussians":int(len(ids)),
                 "mask_agreement":{key:float(val) for key,val in agreement.items()}}
        result,all_views,ids,diagnostics,agreeing,agreement = select(
            [v for v in other if v.view_id in kept])
        diagnostics["rejected_views"] = rejected
        diagnostics["first_pass"] = first
    diagnostics["mask_agreement"]={key:float(val) for key,val in agreement.items()}
    if peel and len(ids) and agreeing:
        peeled,rounds = peel_object_ids(
            renderer, source.means, agreeing, ids, min_inside_fraction=0.7,
            min_views=2, min_inside_weight=0.3,
            dilate_px=8, view_tolerance=0.1,
        )
        diagnostics["peel_rounds"]=rounds
        ids=np.asarray(peeled,np.int64)
    strays2 = stray_object_ids(source.means,ids)
    ids=np.setdiff1d(ids,strays2)
    diagnostics["strays_removed"]+=int(len(strays2))
    if len(ids):
        # Separate objects the name also found (a toy truck, a picture)
        # survive the vote and the stray test as dense clusters of their own.
        x,y,front=project_points(source.means[ids],camera.viewmat,camera.intrinsic,camera.width,camera.height)
        anchor=np.zeros(len(source.means),bool)
        anchor[ids]=front&mask[y,x]
        separate=separate_cluster_ids(source.means,ids,anchor)
        ids=np.setdiff1d(ids,separate)
        diagnostics["separate_clusters_removed"]=int(len(separate))
    if footprint_cleanup and len(ids):
        new_column=footprint_column_ids(source.means,ids,world_up,
                                        radius_fraction=0.12,base_fraction=0.05)
        added=np.setdiff1d(new_column,ids)
        diagnostics["footprint_additional"]=int(len(added))
        # The column catches thin pieces the hull missed: kitchen and bonsai
        # add 0-2 % of the object. Playroom (23 Oct, "buggy" against a wall)
        # added 19x the object, most of the room; then it is not cleanup.
        # With the room's other objects dropped (24 Oct) it added 46 %: the
        # wall behind the buggy, whose large splats blurred the wall above.
        if len(added)>max_footprint_growth*len(ids):
            diagnostics["footprint_skipped"]=True
        else:
            ids=np.union1d(ids,new_column)
    if len(ids)==0:
        raise RuntimeError("No valid object Gaussian IDs selected")
    if ids.max() >= len(source.vertex) or ids.min() < 0:
        raise RuntimeError("Invalid source Gaussian selection")
    residual=removal_residual(renderer,agreeing,ids,world_up=world_up)
    gate=_gate_residual(residual)
    kept_surface=int(len(np.intersect1d(surface,ids)))
    diagnostics["instance_pick"]={"cameras_segmented":segmented,"cameras_dropped":picks["dropped"],
                                  "hero_surface_gaussians":int(len(surface)),
                                  "hero_surface_selected":kept_surface}
    ids_path=out_dir/"object_ids.npz"
    np.savez_compressed(ids_path,object_ids=ids,
                        candidate_ids=result.candidate_ids,
                        views_seen=result.views_seen,
                        views_inside=result.views_inside,
                        column_applied=np.asarray(bool(footprint_cleanup)))
    summary={
        "source_ply":str(source.path), "camera_targets":str(Path(target_npz).resolve()),
        "segment_report":str(report_path), "hero_view_id":hero,
        "object_ids":str(ids_path), "object_gaussians":len(ids),
        "source_gaussians":len(source.vertex),
        "selection":diagnostics, "gates":{"removal_residual":gate},
        "voting_views":[v.view_id for v in other],"footprint_cleanup":bool(footprint_cleanup)
    }
    (out_dir/"object_hull_report.json").write_text(json.dumps(summary,indent=2)+"\n")
    # A selection that lost what the user clicked must not go on to a
    # "done" removal (counter, 24 Oct: 98 Gaussians removed, no camera
    # judged, the bread still there).
    reasons=[]
    if len(surface)>=20 and kept_surface<min_surface_fraction*len(surface):
        reasons.append(f"only {kept_surface} of the {len(surface)} Gaussians on the surface selected in "
                       f"{hero} stayed in the selection ({len(ids)} selected in all)")
    if not residual.get("judged"):
        reasons.append("no camera could judge the removal")
    if reasons:
        _fail(out_dir,collapse_message(name,reasons,len(picks["dropped"])+len(diagnostics.get("rejected_views",[])),
                                       segmented),
              object_gaussians=int(len(ids)),hero_surface_gaussians=int(len(surface)),
              hero_surface_selected=kept_surface,judged=int(residual.get("judged") or 0))
    return summary
