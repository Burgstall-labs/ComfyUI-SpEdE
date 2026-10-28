"""CPU-safe asset boundary for SpEdE Comfy nodes.

No torch, gsplat or Comfy import: these contracts are independently testable.
The masks must be actual segmentations from Studio/SAM/user (never inferred
from camera views) and the returned paths are local to the execution host.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def inspect_camera_pack(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    with np.load(path, allow_pickle=False) as packed:
        required = {"targets", "labels", "viewmats", "intrinsics", "frame_indices"}
        missing = required.difference(packed.files)
        if missing:
            raise ValueError(f"Camera NPZ missing: {sorted(missing)}")
        images = packed["targets"]
        labels = packed["labels"].astype(str).tolist()
        poses = packed["viewmats"]
        intrinsics = packed["intrinsics"]
        frames = packed["frame_indices"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0:
            raise ValueError("targets must be nonempty [N,H,W,3]")
        count, height, width, _ = images.shape
        if width < 1 or height < 1:
            raise ValueError("camera images must have nonzero dimensions")
        if not (len(labels) == count and poses.shape == (count, 4, 4)
                and intrinsics.shape == (count, 3, 3) and frames.shape == (count,)):
            raise ValueError("Camera pack label, pose, intrinsic or frame dimensions differ")
        if not all(labels) or len(labels) != len(set(labels)):
            raise ValueError("Camera labels must be nonempty and unique")
        if not (np.isfinite(poses).all() and np.isfinite(intrinsics).all()):
            raise ValueError("Camera transforms contain non-finite values")
        if not np.all(intrinsics[:, 0, 0] > 0) or not np.all(intrinsics[:, 1, 1] > 0):
            raise ValueError("Camera focal lengths must be positive")
        if not np.isfinite(images).all():
            raise ValueError("Camera target images contain non-finite values")
        return {
            "camera_targets_npz": str(path),
            "camera_count": int(count),
            "width": int(width),
            "height": int(height),
            "view_ids": labels,
        }


def _as_bool_mask(raw, shape: tuple[int, int], allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.shape != shape:
        raise ValueError(f"Mask shape {arr.shape} does not match camera dimensions {shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Mask contains non-finite pixels")
    if arr.dtype == np.bool_:
        mask = arr
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0:
            raise ValueError("Mask has negative pixels")
        scale = float(arr.max(initial=0))
        if scale <= 1.0 + 1e-6:
            mask = arr >= 0.5
        elif scale <= 255:
            mask = arr > 127
        else:
            raise ValueError("Mask values must lie in [0,1] or [0,255]")
    else:
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError("Integer mask values must be in [0,255]")
        mask = arr > (0 if arr.max() <= 1 else 127)
    if not mask.any() and not allow_empty:
        raise ValueError("Empty semantic masks are not a voting view")
    return mask


def _read_mask(path: Path, shape: tuple[int, int], allow_empty: bool = False) -> np.ndarray:
    with Image.open(allowed_local_path(path)) as image:
        return _as_bool_mask(np.asarray(image.convert("L")), shape, allow_empty)


def write_segment_report(
    cameras: Mapping[str, object],
    masks: Mapping[str, object],
    *,
    hero_view_id: str,
    output_dir: str | Path,
    minimum_voting_views: int = 3,
    instances: Mapping[str, np.ndarray] | None = None,
    object_text: str = "",
) -> tuple[Path, Path, dict]:
    """Persist a source-compatible segment report using *supplied* silhouettes.

    Voting views exclude hero; a hero view must never carve its own frustum.
    No mask-finding, SAM inference, or weak-label "agreement" is implied.
    ``instances`` (per non-hero view, the segmenter's separate instances
    [N, H, W]) are kept beside the masks for the selection to choose from.
    """
    hero_view_id = str(hero_view_id).strip()
    available = set(cameras)
    if hero_view_id not in available or hero_view_id not in masks:
        raise ValueError("Hero view needs a known camera and supplied mask")
    unknown = set(masks) - available
    if unknown:
        raise ValueError(f"Unknown camera mask IDs: {sorted(unknown)}")
    if len(masks) != len(set(masks)):
        raise ValueError("Duplicate camera mask labels")
    if minimum_voting_views < 1:
        raise ValueError("minimum_voting_views must be positive")
    out = allowed_local_path(output_dir, output=True)
    out.mkdir(parents=True, exist_ok=True)
    views = {}
    voting = []
    empty = []
    for index, (view_id, camera) in enumerate(cameras.items()):
        if view_id not in masks:
            continue
        if getattr(camera, "is_virtual", False) or getattr(camera, "synthetic", False):
            raise ValueError(f"Only real capture cameras may vote: {view_id}")
        shape = (int(camera.height), int(camera.width))
        value = masks[view_id]
        # A non-hero view where the segmenter found nothing (a small object
        # out of frame or hidden there) does not vote; the hero must not be
        # empty.
        allow_empty = view_id != hero_view_id
        if isinstance(value, (str, Path)):
            mask = _read_mask(Path(value), shape, allow_empty)
        else:
            mask = _as_bool_mask(value, shape, allow_empty)
        if not mask.any():
            empty.append(view_id)
            continue
        # Controlled index filenames, not user-supplied path fragments.
        output_png = out / f"mask_{index:04d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_png)
        views[view_id] = {
            "mask_path": str(output_png),
            "mask_area": int(mask.sum()),
            "source": "supplied",
        }
        if view_id != hero_view_id:
            voting.append(view_id)
            found = None if instances is None else instances.get(view_id)
            if found is not None:
                found = np.asarray(found) > 0.5
                if found.ndim != 3 or found.shape[1:] != shape:
                    raise ValueError(f"Instance masks for {view_id} do not match its camera size")
                instances_npz = out / f"instances_{index:04d}.npz"
                np.savez_compressed(instances_npz, instances=found)
                views[view_id]["instances_path"] = str(instances_npz)
                views[view_id]["instances"] = int(len(found))
    if len(voting) < minimum_voting_views:
        raise ValueError(
            f"Need {minimum_voting_views} non-hero camera masks; got {len(voting)}"
            + (f" ({len(empty)} views had an empty mask)" if empty else "")
        )
    report = {
        "format": "spede-mask-report-v1",
        "hero_view_id": hero_view_id,
        "voting_views": voting,
        "views": views,
        "empty_views": empty,
        "segmentation_provenance": "provided_by_user_or_external_node",
    }
    if str(object_text).strip():
        report["object_text"] = str(object_text).strip()
    path = out / "segment_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path, Path(views[hero_view_id]["mask_path"]), report


def masks_from_directory(
    cameras: Mapping[str, object], directory: str | Path,
    *,
    suffix: str = ".png",
) -> dict[str, Path]:
    if suffix != ".png":
        raise ValueError("Supported mask filename suffix is .png")
    folder = allowed_local_path(directory).resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    found: dict[str, Path] = {}
    for view_id in cameras:
        # File stem must be safe; use native MASK input instead for arbitrary IDs.
        if view_id in {".", ".."} or "/" in view_id or "\\" in view_id:
            raise ValueError("Mask directory mode requires filename-safe camera IDs")
        candidate = folder / (view_id + suffix)
        if candidate.is_file():
            found[view_id] = candidate
    return found


def masks_from_comfy_batch(masks, view_ids: list[str], cameras: Mapping[str, object]) -> dict[str, np.ndarray]:
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    batch = np.asarray(masks)
    if batch.ndim == 2:
        batch = batch[None]
    if batch.ndim != 3:
        raise ValueError("Comfy MASK input must have shape [B,H,W]")
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("Duplicate view IDs")
    if len(view_ids) != batch.shape[0]:
        raise ValueError("MASK batch length does not match view_ids_json")
    if not all(view_id in cameras for view_id in view_ids):
        raise ValueError("Mask view ID not present in camera pack")
    return {view_id: batch[i] for i, view_id in enumerate(view_ids)}


def instances_from_comfy_batch(masks, boxes, view_ids: list[str],
                               cameras: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Per view, the separate instances [N, H, W] of a SAM3_Detect run with
    ``individual_masks``: its MASK output is every view's instances in one
    batch, and its per-view box lists say how many belong to each view."""
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    batch = np.asarray(masks)
    if batch.ndim == 2:
        batch = batch[None]
    if batch.ndim != 3:
        raise ValueError("Comfy MASK input must have shape [N,H,W]")
    if not isinstance(boxes, list) or len(boxes) != len(view_ids):
        raise ValueError("Instance boxes must be one list per view")
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("Duplicate view IDs")
    if not all(view_id in cameras for view_id in view_ids):
        raise ValueError("Mask view ID not present in camera pack")
    counts = [len(row) if isinstance(row, list) else 1 for row in boxes]
    if sum(counts) != batch.shape[0]:
        raise ValueError(f"MASK batch has {batch.shape[0]} instances; the boxes count {sum(counts)}")
    out, start = {}, 0
    for view_id, count in zip(view_ids, counts):
        camera = cameras[view_id]
        shape = (int(camera.height), int(camera.width))
        if count and batch.shape[1:] != shape:
            raise ValueError(f"Instance masks for {view_id} do not match its camera size")
        out[view_id] = batch[start:start + count] if count else np.zeros((0, *shape), np.float32)
        start += count
    return out


def read_selected_ids(path: str | Path, vertex_count: int) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as npz:
        if "object_ids" not in npz.files:
            raise ValueError("Selection NPZ has no object_ids array")
        raw = np.asarray(npz["object_ids"])
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("object_ids must be a nonempty integer vector")
    ids = np.unique(raw.astype(np.int64))
    if int(ids[0]) < 0 or int(ids[-1]) >= vertex_count:
        raise ValueError("Selection IDs are not valid source Gaussian indices")
    return ids


def write_removed_scene(source_ply: Path, object_ids: np.ndarray, output_ply: Path) -> dict:
    """Property-preserving physical Gaussian deletion; no synthetic patching."""
    source = allowed_local_path(source_ply).resolve(strict=True)
    destination = allowed_local_path(output_ply, output=True)
    if source == destination:
        raise ValueError("Cannot overwrite source PLY")
    ply = PlyData.read(str(source))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("Expected vertex-only 3DGS PLY; extra PLY elements require remapping")
    vertices = ply["vertex"].data
    ids = np.asarray(object_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("object_ids must be an integer vector")
    ids = np.unique(ids.astype(np.int64))
    if ids.size == 0 or int(ids[0]) < 0 or int(ids[-1]) >= len(vertices):
        raise ValueError("Selection IDs out of source range or empty")
    kept_count = int(len(vertices) - len(ids))
    if kept_count == 0:
        raise ValueError("Removal would erase every Gaussian")
    keep = np.ones(len(vertices), dtype=bool)
    keep[ids] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: if a job is interrupted, don't leave a partial .ply.
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".spede-", suffix=".ply", dir=destination.parent,
            delete=False,
        ) as stream:
            temp_file = Path(stream.name)
        PlyData(
            [PlyElement.describe(vertices[keep].copy(), "vertex")],
            text=False, byte_order="<",
            comments=list(ply.comments) + [f"SpEdE removed {len(ids)} source Gaussians"],
            obj_info=list(ply.obj_info),
        ).write(str(temp_file))
        os.replace(temp_file, destination)
    finally:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
    return {
        "status": "object_removed_no_patch",
        "source_ply": str(source),
        "scene_ply": str(destination),
        "source_gaussians": int(len(vertices)),
        "removed_gaussians": int(len(ids)),
        "remaining_gaussians": kept_count,
        "attributes_preserved": list(vertices.dtype.names or ()),
        "patch_applied": False,
    }


def allowed_local_path(path: str | Path, *, output: bool = False) -> Path:
    """Optional execution-host filesystem allowlist for hosted deployments.

    Defaults to unrestricted for local private use. For a hosted endpoint
    the gateway MUST set both environment variables and isolate jobs.
    """
    resolved = Path(path).expanduser().resolve()
    env_name = "SPEDE_ALLOWED_OUTPUT_ROOTS" if output else "SPEDE_ALLOWED_INPUT_ROOTS"
    roots = os.environ.get(env_name, "")
    if not roots.strip():
        return resolved
    allowed = [
        Path(part).expanduser().resolve()
        for part in roots.split(os.pathsep) if part.strip()
    ]
    if not allowed:
        raise PermissionError(f"{env_name} is set but has no allowed roots")
    if not any(resolved == base or base in resolved.parents for base in allowed):
        raise PermissionError(f"SpEdE path outside {env_name}: {resolved}")
    return resolved


def asset_fingerprint(path: str | Path) -> str:
    """File-change signal for Comfy IS_CHANGED, not a content-authenticity hash."""
    p = allowed_local_path(path)
    try:
        stat = p.stat()
        return f"{p}:{stat.st_size}:{stat.st_mtime_ns}"
    except FileNotFoundError:
        return f"{p}:missing"


def mask_directory_fingerprint(directory: str | Path) -> str:
    path = allowed_local_path(directory)
    if not path.is_dir():
        return f"{path}:missing"
    return "|".join(
        asset_fingerprint(child)
        for child in sorted(path.glob("*.png"))
        if child.is_file()
    )
