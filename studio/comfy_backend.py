"""SpEdE UI jobs on local ComfyUI through the official Comfy SDK v2.

The hero check, Extract, Remove+Patch, the all-around fill and Replace submit the
pack's API workflows (``workflows/``) to the comfy-api-proxy in front of the
local ComfyUI, wait for them, and lay their results out where the UI's pages read them. The
SpEdE nodes, SAM 3.1, Qwen3-VL and Qwen-Image 2.1 all run inside ComfyUI;
nothing here touches the GPU.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

try:                                    # this repository
    from competition.local_sdk import (MODEL_LICENSE_URL, build_allround_graphs, build_hero_graph,
                                       build_local_graph, build_replace_graph, needs_qwen, validate_local_endpoint)
except ImportError:                     # the exported node pack has local_sdk at its root
    from local_sdk import (MODEL_LICENSE_URL, build_allround_graphs, build_hero_graph, build_local_graph,
                           build_replace_graph, needs_qwen, validate_local_endpoint)
from spede_core.splat_export import ply_to_splat

ACTIVE: dict = {}            # the SDK job being waited on, for Cancel
_active_lock = threading.Lock()
CANCEL = threading.Event()
# The owner's acknowledgement of the Qwen-Image 2.1 Research License
# (config accept_qwen_research_license); set by configure() at server start.
QWEN_ACK = False
QWEN_LICENSE_TEXT = (
    "Qwen-Image 2.1 (and its Qwen3-VL text encoder) is a Research License model. "
    "Read " + MODEL_LICENSE_URL +
    " and set accept_qwen_research_license to true in studio/config.json "
    "(or SPEDE_ACCEPT_QWEN_RESEARCH_LICENSE=1) only for permitted local use, then restart SpEdE."
)


class StepFailed(RuntimeError):
    pass


def _log(log_path: Path, text: str) -> None:
    with open(log_path, "a") as handle:
        handle.write(time.strftime("%H:%M:%S ") + text + "\n")


def check_qwen_license(graph: dict) -> None:
    """The UI's counterpart of ``local_sdk.run_sdk``'s licence gate: a graph
    that loads Qwen-Image 2.1 or its Qwen3-VL text encoder is refused until
    the owner set accept_qwen_research_license. Extract and a hero check
    with a typed name load neither and always pass."""
    if needs_qwen(graph) and not QWEN_ACK:
        raise StepFailed(QWEN_LICENSE_TEXT)


def run_graph(graph: dict, log_path: Path, *, timeout: float = 3600.0):
    """Submit an API-format graph with the Comfy SDK inside the job's GPU
    turn, wait and return the job. Every UI step submits through here, so
    the Qwen licence gate (check_qwen_license) runs before anything is sent.

    Raises StepFailed with ComfyUI's own error text when the job fails."""
    # The worker logs the refusal (repr) once; no extra log line here.
    check_qwen_license(graph)
    from comfy_sdk import Comfy

    import gpu_turn

    endpoint = validate_local_endpoint()
    CANCEL.clear()
    with Comfy(timeout=60.0, client_info="spede-ui") as client:
        workflow = client.workflows.from_json(graph)
        job = client.submit(workflow)
        gpu_turn.attach_prompt(str(job.id))   # the v2 job id is ComfyUI's prompt_id
        _log(log_path, f"submitted Comfy job {job.id} to {endpoint}")
        with _active_lock:
            ACTIVE["job"] = job
        try:
            job.wait(timeout=timeout)
        finally:
            with _active_lock:
                ACTIVE.pop("job", None)
        _log(log_path, f"Comfy job {job.id}: {job.status}")
    if job.status != "succeeded":
        raise StepFailed(f"ComfyUI job {job.id} {job.status}: {json.dumps(job.error, default=str)[:1500]}")
    return job


def cancel_active() -> None:
    CANCEL.set()
    with _active_lock:
        job = ACTIVE.get("job")
    if job is not None:
        try:
            job.cancel()
        except Exception:
            pass


def _job_name(project: str, kind: str) -> str:
    # Each run writes a fresh ComfyUI output folder: spede/local/<job>/...
    stem = project.removeprefix("spede_")[-40:].lstrip("_-")
    return f"{stem}_{kind}_{time.strftime('%H%M%S')}"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_step(*, project: dict, scene: dict, root: Path, comfy_output: Path):
    """A UI step: SAM 3.1 + SpEdE Extract in ComfyUI, copied to root/extract."""
    def run():
        log = root / "comfy_extract.log"
        log.write_text("")
        job_name = _job_name(project["name"], "extract")
        graph = build_local_graph(
            "extract", source_ply=scene["source_ply"], camera_targets=scene["cameras"],
            hero_view=project["view"], object_text=project["label"], job_name=job_name,
            hero_mask=root / "inputs/hero_mask.png",
        )
        out = comfy_output / "spede/local" / job_name
        try:
            run_graph(graph, log)
        except StepFailed as error:
            explained = selection_message(out)
            if explained is None:
                raise
            raise _explain(log, error, explained) from error
        if not (out / "extract/object_only.ply").exists():
            raise StepFailed(f"ComfyUI finished but wrote no {out / 'extract/object_only.ply'}")
        _copy_tree(out / "extract", root / "extract")
        _copy_tree(out / "masks", root / "extract_session/masks")
        _copy_tree(out / "selection", root / "extract_session/selection")
        (root / "extract_session/comfy.json").write_text(json.dumps(
            {"comfy_output": str(out), "job_name": job_name}, indent=2))
        _log(log, "extract copied to " + str(root / "extract"))
    return run


HERO_FILES = (("hero_mask.png", "hero_mask.png"), ("click_mask.png", "click_mask.png"),
              ("name_mask.png", "name_mask.png"), ("preview.png", "selection.png"))


def hero_step(*, name: str, scene: dict, root: Path, comfy_output: Path, points: list, label: str | None,
              load, save, label_source: str | None = None):
    """A UI step: the hero-view check in ComfyUI. SAM 3.1 masks the clicks,
    Qwen3-VL names the selection (unless the user named it), SAM 3.1 finds
    that name in the same view and SpEdE compares the two. The chosen mask
    becomes inputs/hero_mask.png; a disagreement waits for the user."""
    def run():
        project = load(name)
        log = root / "comfy_hero.log"
        log.write_text("")
        job_name = _job_name(name, "hero")
        graph = build_hero_graph(camera_targets=scene["cameras"], hero_view=project["view"], points=points,
                                 object_text=label, job_name=job_name)
        run_graph(graph, log, timeout=900)
        out = comfy_output / "spede/local" / job_name / "hero"
        if not (out / "hero_check.json").exists():
            raise StepFailed(f"ComfyUI finished but wrote no {out / 'hero_check.json'}")
        report = json.loads((out / "hero_check.json").read_text())
        inputs = root / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        for src, dst in HERO_FILES:
            if (out / src).exists():
                shutil.copyfile(out / src, inputs / dst)
            else:
                (inputs / dst).unlink(missing_ok=True)
        (root / "hero_check.json").write_text(json.dumps({**report, "comfy_output": str(out)}, indent=2))
        chosen = report.get("name_pixels") if report.get("chosen") == "name" else report.get("click_pixels")
        project = load(name)
        project.update(stage="decide", label=report.get("name") or "",
                       label_source=label_source or ("user" if label else "qwen-vl"),
                       points=points, selection_pixels=int(chosen or 0), hero_confirmed=bool(report.get("passed")),
                       hero_check={k: report.get(k) for k in ("status", "passed", "chosen", "advice", "iou",
                                                              "name_instances", "click_pixels", "name_pixels")})
        save(project)
        _log(log, f"hero check {report.get('status')}: name {report.get('name')!r}, IoU {report.get('iou')}")
    return run


def _hero_camera(scene: dict, view: str, distance: float = 0.5) -> dict:
    with np.load(scene["cameras"], allow_pickle=True) as data:
        labels = [str(v) for v in data["labels"]]
        viewmats = np.asarray(data["viewmats"], np.float64)
    index = labels.index(view)
    centres = np.stack([np.linalg.inv(m)[:3, 3] for m in viewmats])
    scale = float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) or 1.0
    c2w = np.linalg.inv(viewmats[index])
    return {"position": c2w[:3, 3].tolist(), "lookAt": (c2w[:3, 3] + c2w[:3, 2] * distance * scale).tolist(),
            "up": (-c2w[:3, 1]).tolist()}


SPILL_TEXT = "removed more than the selection"
# The spill gate's numbers (nodes.py SpEdE_LocalAlignMoGeDepth); older nodes give no selection size.
_SPILL_NUMBERS = re.compile(r"changed (\d+) px outside (?:the (\d+)-px selection|it \((\d+)% of the selection)")
SPILL_FRACTION_ANYWAY = 10.0      # the node's maximum: "Remove anyway" when the extra is only shadow


def spill_message(error: str) -> str | None:
    """The UI's explanation of a failed Qwen spill gate: the numbers, the
    cause and what to do; None for any other failure."""
    if SPILL_TEXT not in error:
        return None
    found = _SPILL_NUMBERS.search(error)
    if not found:
        return ("Qwen removed more than the selection (magenta in the picture). If the selection is only part of "
                "the object, use your clicks or add include points; if the extra is only shadow, Remove anyway.")
    pixels = int(found.group(1))
    selected = int(found.group(2)) if found.group(2) else round(100 * pixels / max(int(found.group(3)), 1))
    return (f"Qwen removed more than the selection: {pixels:,} px beyond the {selected:,}-px selection "
            "(magenta in the picture). If the selection is only part of the object, use your clicks or add "
            "include points; if the extra is only shadow, Remove anyway.")


def selection_message(out: Path) -> str | None:
    """The selection's own explanation when it collapsed (the name found other
    objects in most cameras; spede_core.selection writes it); None otherwise."""
    try:
        return str(json.loads((out / "selection/selection_failed.json").read_text())["message"])
    except (OSError, ValueError, KeyError):
        return None


def _explain(log: Path, error: StepFailed, explained: str) -> StepFailed:
    # One line the server's error text puts first (server._error_text).
    with open(log, "a") as handle:
        handle.write(f"{error!r}\nSpEdE: {explained}\n")
    return StepFailed(explained)


def _keep_failed_pictures(out: Path, failed: Path) -> None:
    """A failed Remove keeps Qwen's picture and the spill map where the UI shows them."""
    failed.mkdir(parents=True, exist_ok=True)
    edits = sorted(out.glob("qwen21_edit_*.png"))
    for src, name in ((out / "depth/edit_spill.png", "edit_spill.png"), (edits[-1] if edits else None, "qwen_edit.png")):
        if src is not None and src.exists():
            shutil.copyfile(src, failed / name)


def remove_patch_step(*, project: dict, scene: dict, root: Path, comfy_output: Path, viewer: Path, mode: str = "patch",
                      hero_model: str = "full",
                      seed: int | None = None, ignore_spill: bool = False):
    """Queue the verified planar or experimental depth-aware Comfy workflow.
    ``seed`` varies Qwen-Image's repair (a retry); ``ignore_spill`` lets the
    depth fill pass the spill gate when what Qwen also removed is only shadow."""
    if mode not in {"patch", "depthfill"}:
        raise ValueError(f"Invalid SpEdE patch mode: {mode}")
    def run():
        log = root / "comfy_patch.log"
        log.write_text("")
        failed = root / "comfy_failed"
        if failed.exists():
            shutil.rmtree(failed)
        job_name = _job_name(project["name"], "patch")
        graph = build_local_graph(
            mode, source_ply=scene["source_ply"], camera_targets=scene["cameras"],
            hero_view=project["view"], object_text=project["label"], job_name=job_name,
            hero_mask=root / "inputs/hero_mask.png",
            **({"seed": int(seed)} if seed is not None else {}),
            **({"max_spill_fraction": SPILL_FRACTION_ANYWAY} if ignore_spill and mode == "depthfill" else {}),
            hero_model=hero_model,
        )
        out = comfy_output / "spede/local" / job_name
        if seed is not None or ignore_spill:
            _log(log, f"seed {seed if seed is not None else 'default'}" + (", spill gate relaxed" if ignore_spill else ""))
        try:
            job = run_graph(graph, log)
        except StepFailed as error:
            _keep_failed_pictures(out, failed)
            explained = spill_message(str(error)) or selection_message(out)
            if explained is None:
                raise
            raise _explain(log, error, explained) from error
        artifact_stem = "depth_patch_scene" if mode == "depthfill" else "patched_scene"
        patched = out / f"{artifact_stem}.ply"
        if not patched.exists():
            raise StepFailed(f"ComfyUI finished but wrote no {patched}")
        pictures = root / "comfy"
        if pictures.exists():
            shutil.rmtree(pictures)
        pictures.mkdir(parents=True)
        files = {}
        # The graph saves its pictures into the job's own output folder
        # (local_sdk sets the SaveImage prefixes); read them from disk like
        # the PLY and the reports: the SDK client is closed by now.
        for prefix, name in (("qwen21_edit", "qwen_edit.png"),
                             ("patched_render", "patched_render.png"),
                             ("aligned_moge_depth", "depth_preview.png")):
            saved = sorted(out.glob(f"{prefix}_*.png"))
            if saved:
                shutil.copyfile(saved[-1], pictures / name)
                files[name.removesuffix(".png")] = name
        # The UI serves files from its own run folder: link the patched scene in.
        local_ply = pictures / "patched_scene.ply"
        try:
            os.link(patched, local_ply)
        except OSError:
            shutil.copyfile(patched, local_ply)
        patched = local_ply
        _copy_tree(out / "masks", root / "segment")
        _copy_tree(out / "selection", root / "hull")

        def read(path):
            return json.loads(path.read_text()) if path.exists() else {}
        hull = read(out / "selection/object_hull_report.json")
        patch = read(out / f"{artifact_stem}.patch_report.json")
        g3 = read(out / f"{artifact_stem}.g3_report.json")
        heldout = read(out / f"{artifact_stem}.heldout_report.json")
        removal = (hull.get("gates") or {}).get("removal_residual") or {}
        gates = {
            "removal": removal,
            "plane": (
                {"name": "G2 scene-aligned MoGe depth",
                 "passed": bool((patch.get("g2_depth_alignment") or {}).get("passed")),
                 "support_fraction": (patch.get("g2_depth_alignment") or {}).get("inlier_fraction"),
                 "rmse": (patch.get("g2_depth_alignment") or {}).get("median_relative_error"),
                 "ring": "actual source depth anchors",
                 "depth_model": (patch.get("g2_depth_alignment") or {}).get("depth_model", "affine")}
                if mode == "depthfill" else
                {"name": "G2 support plane",
                 "passed": bool(patch.get("g2_plane_evidence_passed")),
                 "support_fraction": patch.get("plane_support_fraction"),
                 "rmse": patch.get("plane_rmse"),
                 "ring": patch.get("plane_ring")}
            ),
            "patch": {**g3, "name": g3.get("name", "G3 rendered patch agreement")},
            "heldout": heldout,
        }
        (root / "gates.json").write_text(json.dumps(gates, indent=2))
        summary = {
            "backend": "comfy", "comfy_job": str(job.id), "job_name": job_name, "comfy_output": str(out),
            "object_gaussians": hull.get("object_gaussians"), "source_gaussians": hull.get("source_gaussians"),
            "footprint_additional": (hull.get("selection") or {}).get("footprint_additional"),
            "contact_cleared": (patch.get("contact_layer") or {}).get("cleared_gaussians"),
            "births": patch.get("births", patch.get("born_gaussians")),
            "carved": (patch.get("residue_carve") or {}).get("carved_gaussians"),
            "patch_mode": mode, "total_gaussians": patch.get("total_gaussians"),
            "patched_ply": str(patched), "pictures": files, "finished": time.time(),
        }
        (root / "comfy_patch.json").write_text(json.dumps(summary, indent=2))
        generation = int((read(root / "session_manifest.json") or {}).get("generation", 0)) + 1
        (root / "session_manifest.json").write_text(json.dumps(
            {"backend": "comfy", "source_ply": str(scene["source_ply"]), "current_ply": str(patched),
             "generation": generation}, indent=2))
        # The synced original/edited viewer (compare.html).
        target = viewer / project["name"]
        target.mkdir(parents=True, exist_ok=True)
        original = viewer / "_scenes" / project["scene"] / "original.splat"
        if not original.exists():
            ply_to_splat(Path(scene["source_ply"]), original)
        shutil.copyfile(original, target / "original.splat")
        ply_to_splat(patched, target / "edited.splat")
        (target / "camera.json").write_text(json.dumps(_hero_camera(scene, project["view"])))
        _log(log, f"patched scene {patched}: {summary['births']} births, viewer exported")
    return run


def _ply_vertex_count(path: Path) -> int | None:
    """The vertex count from a PLY header (no need to load the splats)."""
    try:
        with open(path, "rb") as handle:
            for _ in range(64):
                line = handle.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header" or not line:
                    break
    except OSError:
        pass
    return None


def progressive_summary(report: dict, **common) -> dict:
    """``comfy_allround.json`` for the progressive fill, from its file report
    (``progressive_scene.progressive_report.json``). ``common`` carries the
    run's bookkeeping (backend, jobs, paths, pictures, finished)."""
    final = report.get("final") or {}
    order = list(report.get("order") or [])
    births = sum(int(((step or {}).get("geometry") or {}).get("born") or 0) for step in report.get("steps") or [])
    return {"method": report.get("method", "progressive"), **common,
            "cameras": order, "cameras_filled": len(order),
            "surface_discs": (report.get("surface") or {}).get("discs"),
            "births": births, "trainable": final.get("trainable"), "pruned": final.get("pruned"),
            "total_gaussians": final.get("total_gaussians", common.get("total_gaussians")),
            "seconds": final.get("seconds"), "after": final.get("after") or {}}


def orbit_summary(report: dict, **common) -> dict:
    """``comfy_allround.json`` for the experimental H3 orbit path, from
    ``allround_scene.allround_report.json``."""
    chain = list(report.get("chain") or [])
    return {"method": "orbit", **common,
            "cameras": chain, "cameras_filled": len(chain), "surface_discs": None,
            "births": report.get("births"), "trainable": report.get("trainable"), "pruned": report.get("pruned"),
            "total_gaussians": report.get("total_gaussians", common.get("total_gaussians")),
            "seconds": report.get("seconds"), "after": report.get("after") or {},
            "before": report.get("before"), "held_out": report.get("held_out"),
            "usable_frames": report.get("usable_frames"), "orbit_frames": report.get("orbit_frames"),
            "pruned_drifted": report.get("pruned_drifted")}


def allround_step(*, project: dict, scene: dict, root: Path, comfy_output: Path, viewer: Path,
                  mode: str = "independent"):
    """The all-around fill on the finished depth-fill result: two Comfy
    prompts in one GPU turn, then the viewer export. Stage 1 plans the fill
    and cleans the hero at 1024 px with Qwen-Image 2.1 (the orbit path's
    stage 1 adds the chain, MoGe births and orbit warp); stage 2, ``mode``
    "independent" (the default): Qwen-Image 2.1 turbo cleans every camera
    that sees the hole on its own photo, the support surface is painted from
    above and one optimisation fits the fill to every cleaned camera (about
    10 min: the kitchen experiment of 23 Oct 2026); "progressive": the hole
    cleaned one camera at a time around the object, births only where the
    render is empty, short optimisations (about 10–25 min and about 26 GB of
    RAM: peak of the 21 Oct 2026 run; the 23 Oct runs ran under a 30 GB
    reservation). ``mode`` "orbit" is the experimental MiniMax H3 orbit video
    path instead."""
    if mode not in {"independent", "progressive", "orbit"}:
        raise ValueError(f"Invalid SpEdE all-around mode: {mode}")

    def run():
        log = root / "comfy_allround.log"
        log.write_text("")
        patch = json.loads((root / "comfy_patch.json").read_text())
        if patch.get("patch_mode") != "depthfill":
            raise StepFailed("the all-around fill builds on the depth fill (patch_mode depthfill)")
        job_name = _job_name(project["name"], "allround")
        one, two = build_allround_graphs(
            run_dir=patch["comfy_output"], source_ply=scene["source_ply"], camera_targets=scene["cameras"],
            object_text=project["label"], job_name=job_name, photo_dir=scene.get("photos"),
            second="orbit" if mode == "orbit" else "progressive",
            method="independent" if mode == "orbit" else mode)
        out = comfy_output / "spede/allround" / job_name
        if mode == "orbit":
            _log(log, f"stage 1: Qwen-Image 2.1 clean hero at 1024 px, chain, MoGe births, orbit warp -> {out}")
        else:
            _log(log, f"stage 1: plan, Qwen-Image 2.1 clean hero at 1024 px -> {out}")
        first = run_graph(one, log, timeout=3 * 3600.0)
        if mode == "orbit":
            _log(log, "stage 2 (orbit, experimental): MiniMax H3 orbit video, optimisation")
        elif mode == "independent":
            _log(log, "stage 2: Qwen-Image 2.1 turbo cleans every camera that sees the hole on its own photo, "
                      "support surface painted from above, one optimisation against every cleaned camera "
                      "(about 10 min)")
        else:
            _log(log, "stage 2: Qwen-Image 2.1 turbo cleans the hole camera by camera, "
                      "births where the render is empty, optimisation against every cleaned camera "
                      "(10–25 min, about 26 GB of RAM: peak of the 21 Oct 2026 run; the 23 Oct runs ran under a 30 GB reservation)")
        job = run_graph(two, log, timeout=3 * 3600.0)
        result = out / ("allround_scene.ply" if mode == "orbit" else "progressive/progressive_scene.ply")
        if not result.exists():
            raise StepFailed(f"ComfyUI finished but wrote no {result}")
        report_path = result.with_suffix(".allround_report.json" if mode == "orbit" else ".progressive_report.json")
        if not report_path.exists():
            raise StepFailed(f"ComfyUI finished but wrote no {report_path}")
        report = json.loads(report_path.read_text())
        pictures = root / "allround"
        if pictures.exists():
            shutil.rmtree(pictures)
        pictures.mkdir(parents=True)
        files = {}
        sources = ([(out / n, n) for n in ("chain_sheet.png", "init_sheet.png", "warp_sheet.png", "result_sheet.png")]
                   if mode == "orbit" else [(result.parent / "progressive_sheet.png", "progressive_sheet.png")])
        for src, name in [*sources, (out / "hero_clean.png", "hero_clean.png")]:
            if src.exists():
                shutil.copyfile(src, pictures / name)
                files[name.removesuffix(".png")] = name
            elif name in {"progressive_sheet.png", "hero_clean.png"}:
                _log(log, f"no {name} at {src}; not copied to the result card")
        local_ply = pictures / "allround_scene.ply"
        try:
            os.link(result, local_ply)
        except OSError:
            shutil.copyfile(result, local_ply)
        common = {"backend": "comfy", "comfy_job": str(job.id), "stage1_job": str(first.id), "job_name": job_name,
                  "comfy_output": str(out), "total_gaussians": _ply_vertex_count(local_ply),
                  "allround_ply": str(local_ply), "pictures": files, "finished": time.time()}
        summary = (orbit_summary if mode == "orbit" else progressive_summary)(report, **common)
        (root / "comfy_allround.json").write_text(json.dumps(summary, indent=2))
        manifest = json.loads((root / "session_manifest.json").read_text())
        first_allround = not manifest.get("allround")
        manifest.update(current_ply=str(local_ply), generation=int(manifest.get("generation", 0)) + 1,
                        allround=True)
        (root / "session_manifest.json").write_text(json.dumps(manifest, indent=2))
        target = viewer / project["name"]
        target.mkdir(parents=True, exist_ok=True)
        hero_only = target / "edited.splat"
        # Only the first all-around run sees the hero-only patch in edited.splat;
        # a rerun would otherwise overwrite the baseline with the last all-around result.
        if first_allround and hero_only.exists():
            shutil.copyfile(hero_only, target / "hero_only.splat")
        ply_to_splat(local_ply, target / "edited.splat")
        for path in target.iterdir():
            path.chmod(0o644)
        if mode == "orbit":
            _log(log, f"all-around scene {local_ply}: {report.get('usable_frames')}/{report.get('orbit_frames')} "
                      "video frames used, viewer exported")
        else:
            _log(log, f"all-around scene {local_ply} ({mode}): {summary['cameras_filled']} cameras filled, "
                      f"{summary['births']} births, viewer exported")
    return run


# A Replace's own files in the project; a new Replace sets them aside (_previous_replace_*), Remove too.
REPLACE_STATE = ("replace", "comfy_replace.json", "comfy_replace.log")
# Replace's pictures in the job's output folder (SaveImage prefixes, local_sdk) -> root/replace.
REPLACE_PICTURES = (("replace_qwen", "qwen_insert.png"), ("replace_render", "replaced_render.png"),
                    ("replace_clean_hero", "clean_hero.png"), ("replace_cutout", "cutout.png"))


def replace_inputs(root: Path) -> dict:
    """What a Replace builds on, from the project folder. Without a Remove,
    Replace is an inverse extract: the source scene without exactly the
    extracted object; with one, the Remove's result as it is.

    - ``object_ply``: the project's Extract (extract/object_only.ply) when it
      is newer than the selection (inputs/hero_mask.png); else None and the
      Replace job extracts first. On a Remove it only measures the old
      object (plane, footprint, size);
    - ``remove_scene_ply``: a Remove's result scene, if the project has one
      (the all-around fill, else the hero patch): the base, the object gone
      and its spot filled there. Always the Remove's, never an earlier
      Replace's, so a retry swaps the new object;
    - ``hero_image``: the Remove's cleaned hero picture (Qwen-Image 2.1's
      repair), which Qwen draws the new object into; else the base is
      rendered in the hero camera."""
    mask = root / "inputs/hero_mask.png"
    extract = root / "extract/object_only.ply"
    fresh = extract.exists() and (not mask.exists() or extract.stat().st_mtime >= mask.stat().st_mtime)
    remove_scene = hero = None
    patch = _read_json(root / "comfy_patch.json")
    if patch:
        allround = _read_json(root / "comfy_allround.json")
        for candidate in (allround.get("allround_ply"), patch.get("patched_ply")):
            if candidate and Path(candidate).exists():
                remove_scene = candidate
                break
        if (root / "comfy/qwen_edit.png").exists():
            hero = str(root / "comfy/qwen_edit.png")
    return {"object_ply": str(extract) if fresh else None, "remove_scene_ply": remove_scene, "hero_image": hero}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def replace_step(*, project: dict, scene: dict, root: Path, comfy_output: Path, viewer: Path, text: str,
                 image: str | None = None, seed: int | None = None, max_shift: float = 0.25, size: str = "match"):
    """Replace: one ComfyUI job, no Remove needed. The new object goes into
    the project's Remove result when there is one (the all-around fill, else
    the hero patch: the object gone and its spot filled there), else into
    the source scene without exactly the extracted object (the project's
    Extract; computed first in the same job when there is none or the
    selection changed since), the bare rest of its footprint filled.
    Qwen-Image 2.1 turbo draws ``text`` (with ``image``, a reference picture
    in ComfyUI's input folder) where the old object stood, SAM 3.1 cuts it
    out, TripoSplat makes Gaussians of it and SpEdE stands them on the old
    object's surface at ``size`` (match, natural or larger; measured on the
    Extract). Every run starts from the Remove's result or the source scene,
    so a retry swaps the object, it never adds a second."""
    def run():
        log = root / "comfy_replace.log"
        log.write_text("")
        inputs = replace_inputs(root)
        job_name = _job_name(project["name"], "replace")
        graph = build_replace_graph(
            source_ply=scene["source_ply"], camera_targets=scene["cameras"], hero_view=project["view"], text=text,
            removed_label=project.get("label") or "", hero_mask=root / "inputs/hero_mask.png",
            reference_image=image, max_shift=max_shift, size=size, job_name=job_name,
            **inputs, **({"seed": int(seed)} if seed is not None else {}))
        out = comfy_output / "spede/local" / job_name
        _log(log, f"replace with {text!r}" + (f" like {image}" if image else "") +
             f", size {size}, seed {graph['35a']['inputs']['noise_seed']}; "
             + (f"extract {inputs['object_ply']}" if inputs["object_ply"] else "extracting first (no current extract)")
             + (f", builds on the Remove result {inputs['remove_scene_ply']}" if inputs["remove_scene_ply"]
                else ", source without the extract, footprint filled")
             + (", Qwen draws into the Remove's cleaned hero" if inputs["hero_image"] else ", Qwen draws into the base render"))
        try:
            job = run_graph(graph, log, timeout=1800)
        except StepFailed as error:
            explained = selection_message(out)
            if explained is None:
                raise
            raise _explain(log, error, explained) from error
        result = out / "replaced_scene.ply"
        report_path = out / "replaced_scene.replace_report.json"
        if not result.exists() or not report_path.exists():
            raise StepFailed(f"ComfyUI finished but wrote no {result} and its report")
        report = json.loads(report_path.read_text())
        if inputs["object_ply"] is None and (out / "extract/object_only.ply").exists():
            # The extract this job made is the project's Extract from now on (as extract_step lays it out).
            _copy_tree(out / "extract", root / "extract")
            for sub, dst in (("masks", "extract_session/masks"), ("selection", "extract_session/selection")):
                if (out / sub).exists():
                    _copy_tree(out / sub, root / dst)
            (root / "extract_session/comfy.json").write_text(json.dumps(
                {"comfy_output": str(out), "job_name": job_name, "by": "replace"}, indent=2))
        # Only now, with the new result in hand: a failed Replace leaves the earlier one as it was.
        if (root / "comfy_replace.json").exists():
            keep = root / f"_previous_replace_{time.strftime('%H%M%S')}"
            n = 1
            while keep.exists():                    # two Replaces within a second
                n += 1
                keep = root / f"_previous_replace_{time.strftime('%H%M%S')}_{n}"
            keep.mkdir()
            for item in REPLACE_STATE[:2]:
                if (root / item).exists():
                    (root / item).rename(keep / item)
        pictures = root / "replace"
        if pictures.exists():
            shutil.rmtree(pictures)
        pictures.mkdir(parents=True)
        files = {}
        for prefix, name in REPLACE_PICTURES:
            saved = sorted(out.glob(f"{prefix}_*.png"))
            if saved:
                shutil.copyfile(saved[-1], pictures / name)
                files[name.removesuffix(".png")] = name
        local_ply = pictures / "replaced_scene.ply"
        try:
            os.link(result, local_ply)
        except OSError:
            shutil.copyfile(result, local_ply)
        placement = report.get("placement") or {}
        fill = report.get("fill") or {}
        summary = {"backend": "comfy", "comfy_job": str(job.id), "job_name": job_name, "comfy_output": str(out),
                   "text": text, "image": image, "size": size, "seed": graph["35a"]["inputs"]["noise_seed"],
                   "source_ply": str(scene["source_ply"]), "object_ply": report.get("object_ply"),
                   "base": "remove" if inputs["remove_scene_ply"] else "source",
                   "remove_scene_ply": inputs["remove_scene_ply"], "qwen_input": "remove hero" if inputs["hero_image"] else "base render",
                   "replaced_ply": str(local_ply), "total_gaussians": report.get("total_gaussians"),
                   "object_gaussians": report.get("object_gaussians"), "removed_gaussians": fill.get("object_gaussians"),
                   "shadow_gaussians": report.get("shadow_gaussians"), "fill_gaussians": report.get("fill_gaussians"),
                   "fill_source": fill.get("source"), "fill_fraction": fill.get("filled_fraction"),
                   "placement": {k: placement.get(k) for k in ("yaw_deg", "size", "height_vs_removed",
                                                               "height_natural_vs_removed", "dominant_vs_removed",
                                                               "shift_fraction", "shift_capped", "silhouette_iou")},
                   "pictures": files, "finished": time.time()}
        (root / "comfy_replace.json").write_text(json.dumps(summary, indent=2))
        manifest = _read_json(root / "session_manifest.json") or {"backend": "comfy", "source_ply": str(scene["source_ply"])}
        manifest.pop("replace_base", None)          # the early beta's key; the base is in comfy_replace.json
        manifest.update(current_ply=str(local_ply), generation=int(manifest.get("generation", 0)) + 1)
        (root / "session_manifest.json").write_text(json.dumps(manifest, indent=2))
        target = viewer / project["name"]
        target.mkdir(parents=True, exist_ok=True)
        original = viewer / "_scenes" / project["scene"] / "original.splat"
        if not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            ply_to_splat(Path(scene["source_ply"]), original)
        if not (target / "original.splat").exists():
            shutil.copyfile(original, target / "original.splat")
        if not (target / "camera.json").exists():
            (target / "camera.json").write_text(json.dumps(_hero_camera(scene, project["view"])))
        ply_to_splat(local_ply, target / "edited.splat")
        _log(log, f"replaced scene {local_ply}: {summary['object_gaussians']} new object Gaussians at "
                  f"{placement.get('height_vs_removed')}x the old height ({size}), "
                  + ("on the Remove result" if inputs["remove_scene_ply"]
                     else f"{summary['fill_gaussians']} fill Gaussians ({summary['fill_source']})")
                  + ", viewer exported")
    return run


def configure(base_url: str, *, accept_qwen: bool = False) -> None:
    """The SDK reads COMFY_BASE_URL when a client is made; the UI makes all
    clients in this one process, so it is set once at start, with the Qwen
    research-licence acknowledgement."""
    global QWEN_ACK
    os.environ["COMFY_BASE_URL"] = base_url
    QWEN_ACK = bool(accept_qwen)
