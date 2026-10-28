# SPDX-License-Identifier: AGPL-3.0-or-later
# Ported by the copyright holder (Burgstall Labs) from Burgstall-labs/Splatedit@cb8ccb665162037da1d991d9a77ab677c70a3612; released under AGPL-3.0-or-later per the owner decision of 2026-10-08 (see NOTICE).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np

try:
    import torch
except ImportError:  # CameraSet / CameraView stay usable on numpy alone
    torch = None

_TORCH_MISSING = "spede_core.renderer needs torch and gsplat for rendering; install them for ComfyUI's Python"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_TORCH_MISSING)

from .ply import GaussianPly
from .roi import ContributionMap
from .view_planner import CandidateView


@dataclass(frozen=True)
class CameraView:
    view_id: str
    viewmat: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int
    frame_index: int = 0
    roi_box: tuple[float, float, float, float] | None = None
    base_view_id: str | None = None
    # Render-only poses (e.g. orbit conditioning) that are not real cameras.
    synthetic: bool = False

    @property
    def is_virtual(self) -> bool:
        return self.roi_box is not None or self.synthetic

    @property
    def primary_view_id(self) -> str:
        return self.base_view_id or base_view_id(self.view_id)


def base_view_id(view_id: str) -> str:
    """Return the base camera view ID, stripping any '@roi' suffix."""
    return view_id.split("@")[0]


class CameraSet:
    def __init__(self, views: Sequence[CameraView]) -> None:
        self.views = list(views)
        self.by_id = {view.view_id: view for view in self.views}
        if len(self.by_id) != len(self.views):
            raise ValueError("camera view ids must be unique")

    def add_view(self, view: CameraView) -> None:
        """Register or update a camera view."""
        if view.view_id in self.by_id:
            idx = next(i for i, v in enumerate(self.views) if v.view_id == view.view_id)
            self.views[idx] = view
        else:
            self.views.append(view)
        self.by_id[view.view_id] = view

    @classmethod
    def from_camera_pack(cls, path: Path) -> "CameraSet":
        packed = np.load(path)
        images = packed["targets"]
        labels = packed["labels"].astype(str).tolist()
        return cls([
            CameraView(
                view_id=label,
                viewmat=packed["viewmats"][i].astype(np.float32),
                intrinsic=packed["intrinsics"][i].astype(np.float32),
                width=int(images.shape[2]),
                height=int(images.shape[1]),
                frame_index=int(packed["frame_indices"][i]),
            )
            for i, label in enumerate(labels)
        ])


def compute_virtual_roi_box(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    padding_frac: float = 0.0,
) -> tuple[float, float, float]:
    """Compute a square ROI bounding box [x0, y0, crop_side] from a mask, clamped to image bounds."""
    mask_array = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_array)
    if not len(xs):
        raise ValueError("cannot build virtual ROI camera from an empty mask")

    min_x, max_x = float(xs.min()), float(xs.max() + 1.0)
    min_y, max_y = float(ys.min()), float(ys.max() + 1.0)
    box_w = max_x - min_x
    box_h = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    raw_side = max(box_w, box_h)
    crop_side = raw_side * (1.0 + 2.0 * float(padding_frac))
    max_side = float(min(width, height))
    crop_side = min(crop_side, max_side)

    x0 = float(np.clip(cx - crop_side / 2.0, 0.0, float(width) - crop_side))
    y0 = float(np.clip(cy - crop_side / 2.0, 0.0, float(height) - crop_side))

    return x0, y0, crop_side


def crop_to_view(array: np.ndarray, camera: CameraView, *, nearest: bool) -> np.ndarray:
    """Resample a base-camera image or mask into a virtual ROI camera.

    Exact because the virtual camera shares the base pose and only crops and
    scales the image plane (see ``make_virtual_roi_camera``).
    """
    from PIL import Image

    if camera.roi_box is None:
        return np.asarray(array)
    x0, y0, x1, y1 = camera.roi_box
    source = np.asarray(array)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if source.ndim == 3 and source.dtype != np.uint8:
        # Multi-channel float data (normals): resample channel by channel.
        return np.stack(
            [crop_to_view(source[..., c], camera, nearest=nearest) for c in range(source.shape[2])],
            axis=-1,
        )
    is_mask = source.dtype == bool
    if is_mask:
        image = Image.fromarray(source.astype(np.uint8) * 255)
    elif source.dtype == np.uint8:
        image = Image.fromarray(source)
    else:
        image = Image.fromarray(source.astype(np.float32), mode="F")
    out = np.asarray(
        image.transform(
            (camera.width, camera.height),
            Image.Transform.EXTENT,
            (x0, y0, x1, y1),
            resample=resample,
        )
    )
    return out > 127 if is_mask else out


def uncrop_into_base(crop: np.ndarray, camera: CameraView, base: np.ndarray) -> np.ndarray:
    """Paste a virtual ROI view back into its base camera's frame."""
    from PIL import Image

    if camera.roi_box is None:
        return np.asarray(crop)
    x0, y0, x1, y1 = (int(round(v)) for v in camera.roi_box)
    out = np.array(base, copy=True)
    patch = np.asarray(crop)
    if patch.dtype == bool:
        resized = np.asarray(
            Image.fromarray(patch.astype(np.uint8) * 255).resize(
                (x1 - x0, y1 - y0), Image.Resampling.NEAREST
            )
        ) > 127
    else:
        resized = np.asarray(
            Image.fromarray(patch).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        )
    out[y0:y1, x0:x1] = resized
    return out


def make_virtual_roi_camera(
    base: CameraView,
    mask: np.ndarray,
    *,
    padding_frac: float = 0.0,
    out_size: int | None = None,
    view_id: str | None = None,
) -> CameraView:
    """Build a virtual ROI camera that natively renders a square crop of the ROI.

    The virtual camera shares the base camera's viewmat (same 3D pose), but uses
    a rescaled and shifted intrinsic:
      fx' = fx * s
      fy' = fy * s
      cx' = (cx - x0) * s
      cy' = (cy - y0) * s
    where s = out_size / crop_side, width = height = out_size.
    """
    resolved_out_size = int(out_size if out_size is not None else base.width)
    x0, y0, crop_side = compute_virtual_roi_box(
        mask,
        base.width,
        base.height,
        padding_frac=padding_frac,
    )
    s = float(resolved_out_size) / float(crop_side)

    fx = float(base.intrinsic[0, 0])
    fy = float(base.intrinsic[1, 1])
    cx = float(base.intrinsic[0, 2])
    cy = float(base.intrinsic[1, 2])

    fx_prime = fx * s
    fy_prime = fy * s
    cx_prime = (cx - x0) * s
    cy_prime = (cy - y0) * s

    intrinsic_prime = np.asarray(
        [
            [fx_prime, 0.0, cx_prime],
            [0.0, fy_prime, cy_prime],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    roi_view_id = view_id or f"{base.view_id}@roi"
    return CameraView(
        view_id=roi_view_id,
        viewmat=np.asarray(base.viewmat, dtype=np.float32).copy(),
        intrinsic=intrinsic_prime,
        width=int(resolved_out_size),
        height=int(resolved_out_size),
        frame_index=base.frame_index,
        roi_box=(x0, y0, x0 + crop_side, y0 + crop_side),
        base_view_id=base.base_view_id or base.view_id,
    )


def contribution_weights(
    g_ids: torch.Tensor,
    p_ids: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    *,
    width: int,
    height: int,
    top_k: int,
    blend: bool,
    scene_ids: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K contributors per pixel from a per-ray intersection listing
    (``g_ids``/``p_ids`` pairs, front to back within each pixel, any order
    across pixels), as the renderer's metadata gives it.

    ``blend=True``: weight = alpha_i * prod_{j<i} (1 - alpha_j) with the Gaussian's
    alpha at the pixel centre, kept in float64 so the transmittance of a late
    pixel does not lose precision to a long cumulative sum (float32 drifted by
    up to a factor e at 1024 px). The K largest weights win, in ray order.
    ``blend=False``: the first K along the ray with weights 1/(rank+1).
    Returns (ids [H, W, K] with -1 for empty, weights [H, W, K])."""
    _require_torch()
    K = int(top_k)
    size = int(width) * int(height)
    empty_ids = np.full((height, width, K), -1, dtype=np.int64)
    empty_w = np.zeros((height, width, K), dtype=np.float32)
    if g_ids.numel() == 0:
        return empty_ids, empty_w
    device = p_ids.device
    g, p = g_ids.long(), p_ids.long()
    order = torch.sort(p, stable=True).indices          # ray order kept within a pixel
    g, p = g[order], p[order]
    if blend:
        px = (p % width).to(torch.float64) + 0.5
        py = torch.div(p, width, rounding_mode="floor").to(torch.float64) + 0.5
        m, c = means2d[g].to(torch.float64), conics[g].to(torch.float64)
        dx, dy = m[:, 0] - px, m[:, 1] - py
        sigma = 0.5 * (c[:, 0] * dx * dx + c[:, 2] * dy * dy) + c[:, 1] * dx * dy
        alpha = torch.clamp(opacities[g].to(torch.float64) * torch.exp(-sigma), max=0.999)
        alpha = torch.where(sigma >= 0, alpha, torch.zeros_like(alpha))
        log_keep = torch.log1p(-alpha)
        cumulative = torch.cumsum(log_keep, 0)
        is_start = torch.ones_like(p, dtype=torch.bool)
        is_start[1:] = p[1:] != p[:-1]
        start = torch.cummax(torch.where(is_start, torch.arange(len(p), device=device), torch.zeros_like(p)), 0).values
        before = cumulative - log_keep - (cumulative[start] - log_keep[start])
        weights = alpha * torch.exp(before)
        # The K largest per pixel: by weight, then (stable) by pixel.
        by_weight = torch.argsort(weights, descending=True, stable=True)
        order = by_weight[torch.sort(p[by_weight], stable=True).indices]
        g, p, weights = g[order], p[order], weights[order]
    is_start = torch.ones_like(p, dtype=torch.bool)
    is_start[1:] = p[1:] != p[:-1]
    start = torch.cummax(torch.where(is_start, torch.arange(len(p), device=device), torch.zeros_like(p)), 0).values
    ranks = torch.arange(len(p), device=device) - start
    keep = ranks < K
    if not blend:
        weights = 1.0 / (ranks.to(torch.float64) + 1.0)
    ids = scene_ids[g].long() if scene_ids is not None else g
    out_ids = torch.full((size * K,), -1, dtype=torch.int64, device=device)
    out_w = torch.zeros((size * K,), dtype=torch.float32, device=device)
    linear = p[keep] * K + ranks[keep]
    out_ids.scatter_(0, linear, ids[keep])
    out_w.scatter_(0, linear, weights[keep].to(torch.float32))
    return out_ids.view(height, width, K).cpu().numpy(), out_w.view(height, width, K).cpu().numpy()


class GsplatRenderer:
    """Render source/current splats and query per-pixel Gaussian contributors.

    Contributor maps require the 2026 gsplat contributor-ID API. Rendering
    itself continues to use the regular public rasterization API.
    """

    def __init__(
        self,
        *,
        source_ply: Path,
        cameras: CameraSet,
        current_ply: Path | None = None,
        device: str = "cuda",
        top_k_contributors: int = 8,
    ) -> None:
        _require_torch()
        self.source_ply = Path(source_ply)
        self.current_ply = Path(current_ply or source_ply)
        self.cameras = cameras
        self.device = torch.device(device)
        self.top_k_contributors = int(top_k_contributors)
        if self.top_k_contributors < 1:
            raise ValueError("top_k_contributors must be positive")
        self._image_cache: dict[tuple[str, str], np.ndarray] = {}
        self._contribution_cache: dict[tuple[str, str], ContributionMap] = {}
        self._depth_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._ply_cache: dict[Path, GaussianPly] = {}
        # Device copies of the scenes (uploading a large splat per render
        # cost more than rendering it); dropped with the PLY cache.
        self._gpu_cache: dict[Path, dict[str, torch.Tensor]] = {}

    def add_view(self, view: CameraView) -> None:
        """Register a virtual or additional camera view."""
        self.cameras.add_view(view)
        for prefix in ("edited", "source"):
            self._image_cache.pop((prefix, view.view_id), None)
            self._contribution_cache.pop((prefix, view.view_id), None)
            self._depth_cache.pop((prefix, view.view_id), None)

    def set_current_ply(self, path: Path) -> None:
        self.current_ply = Path(path)
        # Scene files are rewritten in place across generations.
        self._ply_cache = {
            key: value for key, value in self._ply_cache.items()
            if key == Path(self.source_ply)
        }
        self._gpu_cache = {
            key: value for key, value in self._gpu_cache.items()
            if key == Path(self.source_ply)
        }
        # The source scene never changes: its renders stay.
        for cache in (self._image_cache, self._contribution_cache, self._depth_cache):
            for key in [k for k in cache if k[0] != "source"]:
                del cache[key]

    def _scene(self, edited: bool) -> GaussianPly:
        path = Path(self.current_ply if edited else self.source_ply)
        if path not in self._ply_cache:
            self._ply_cache[path] = GaussianPly.load(path)
        return self._ply_cache[path]

    def _render_raw(
        self,
        view_id: str,
        *,
        edited: bool,
        excluded_gaussian_ids: np.ndarray | None = None,
        render_mode: str = "RGB",
        included_gaussian_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], GaussianPly]:
        """One rasterization. ``excluded_gaussian_ids`` are made transparent;
        ``included_gaussian_ids`` renders only that subset (its tensors are
        sliced, so a small object costs a small render, not a whole-scene one).
        Packed contributor ids in ``meta`` always index the whole scene."""
        from gsplat import rasterization

        camera = self.cameras.by_id[view_id]
        scene = self._scene(edited)
        path = Path(self.current_ply if edited else self.source_ply)
        tensors = self._gpu_cache.get(path)
        if tensors is None:
            tensors = {
                "means": torch.from_numpy(scene.means).to(self.device),
                "quats": torch.from_numpy(scene.quaternions).to(self.device),
                "scales": torch.exp(torch.from_numpy(scene.log_scales).to(self.device)),
                "opacities": torch.sigmoid(torch.from_numpy(scene.opacity_logits).to(self.device)).squeeze(-1),
                "colors": torch.cat((torch.from_numpy(scene.dc).to(self.device),
                                     torch.from_numpy(scene.rest).to(self.device)), dim=1),
            }
            self._gpu_cache[path] = tensors
        means, quats, scales = tensors["means"], tensors["quats"], tensors["scales"]
        opacities, colors = tensors["opacities"], tensors["colors"]
        subset = None
        if included_gaussian_ids is not None:
            subset = torch.from_numpy(np.unique(np.asarray(included_gaussian_ids, dtype=np.int64).reshape(-1))).to(self.device)
            if subset.numel() == 0:
                raise ValueError("render of an empty Gaussian subset")
            if int(subset.min()) < 0 or int(subset.max()) >= len(scene.vertex):
                raise ValueError("included Gaussian ID out of range")
            means, quats, scales = means[subset], quats[subset], scales[subset]
            opacities, colors = opacities[subset], colors[subset]
        if excluded_gaussian_ids is not None:
            # No np.unique: repeated ids are harmless in the assignment, and
            # sorting millions of ids cost more than the render.
            ids = np.asarray(excluded_gaussian_ids, dtype=np.int64).reshape(-1)
            if ids.size:
                if int(ids.min()) < 0 or int(ids.max()) >= len(scene.vertex):
                    raise ValueError("excluded Gaussian ID out of range")
                if subset is not None:
                    raise ValueError("included and excluded ids together are not supported")
                opacities = opacities.clone()
                opacities[torch.from_numpy(ids).to(self.device)] = 0.0
        degree = scene.max_sh_degree
        viewmat = torch.from_numpy(camera.viewmat).to(self.device)[None]
        intrinsic = torch.from_numpy(camera.intrinsic).to(self.device)[None]
        rendered, alpha, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsic,
            width=camera.width,
            height=camera.height,
            sh_degree=degree,
            packed=True,
            render_mode=render_mode,
            rasterize_mode="classic",
        )
        image = (
            rendered[0, ..., :3]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        alpha_np = alpha[0, ..., 0].detach().to(torch.float32).cpu().numpy()
        meta["alpha"] = alpha
        if subset is not None and isinstance(meta.get("gaussian_ids"), torch.Tensor):
            meta["gaussian_ids"] = subset[meta["gaussian_ids"].long()]
        if render_mode == "RGB+ED":
            depth_np = rendered[0, ..., 3].detach().to(torch.float32).cpu().numpy()
            meta["depth"] = depth_np
        return image, alpha_np, meta, scene

    def render(self, view_id: str, *, edited: bool) -> np.ndarray:
        key = ("edited" if edited else "source", view_id)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached.copy()
        image, _alpha, _meta, _scene = self._render_raw(view_id, edited=edited)
        self._image_cache[key] = image
        return image.copy()

    def render_depth(
        self,
        view_id: str,
        *,
        edited: bool = True,
        excluded_gaussian_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render expected depth and alpha for a camera view."""
        if excluded_gaussian_ids is not None and np.asarray(excluded_gaussian_ids).size == 0:
            excluded_gaussian_ids = None            # nothing excluded: the cached render
        key = ("edited" if edited else "source", view_id)
        cached = self._depth_cache.get(key) if excluded_gaussian_ids is None else None
        if cached is not None:
            depth, alpha = cached
            return depth.copy(), alpha.copy()
        _image, alpha, meta, _scene = self._render_raw(
            view_id,
            edited=edited,
            excluded_gaussian_ids=excluded_gaussian_ids,
            render_mode="RGB+ED",
        )
        if "depth" in meta and isinstance(meta["depth"], np.ndarray):
            depth = meta["depth"]
        elif "depth" in meta and isinstance(meta["depth"], torch.Tensor):
            depth = meta["depth"].detach().to(torch.float32).cpu().numpy()
        else:
            raise RuntimeError(
                f"render_depth failed to produce expected depth for view {view_id!r}"
            )
        depth = depth.astype(np.float32)
        alpha = alpha.astype(np.float32)
        if excluded_gaussian_ids is None:
            self._depth_cache[key] = (depth, alpha)
        return depth.copy(), alpha.copy()

    def render_rgbd(
        self,
        view_id: str,
        *,
        edited: bool = True,
        excluded_gaussian_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image, alpha and expected depth from one pass (callers that need
        both had rendered twice). Cached, and filling both caches, when
        nothing is excluded."""
        if excluded_gaussian_ids is not None and np.asarray(excluded_gaussian_ids).size == 0:
            excluded_gaussian_ids = None
        key = ("edited" if edited else "source", view_id)
        if excluded_gaussian_ids is None and key in self._image_cache and key in self._depth_cache:
            depth, alpha = self._depth_cache[key]
            return self._image_cache[key].copy(), alpha.copy(), depth.copy()
        image, alpha, meta, _scene = self._render_raw(view_id, edited=edited, excluded_gaussian_ids=excluded_gaussian_ids,
                                                      render_mode="RGB+ED")
        depth = np.asarray(meta["depth"], np.float32)
        alpha = np.asarray(alpha, np.float32)
        if excluded_gaussian_ids is None:
            self._image_cache[key] = image
            self._depth_cache[key] = (depth, alpha)
        return image.copy(), alpha.copy(), depth.copy()

    def render_raw(
        self,
        view_id: str,
        *,
        edited: bool,
        excluded_gaussian_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], GaussianPly]:
        return self._render_raw(
            view_id,
            edited=edited,
            excluded_gaussian_ids=excluded_gaussian_ids,
        )

    def contribution_map(self, view_id: str, *, edited: bool = True, blend: bool = False) -> ContributionMap:
        """Top-K contributors per pixel. ``blend=True``: true alpha-blending
        weights (alpha * transmittance); otherwise the legacy rank weights."""
        key = ("edited" if edited else "source", view_id) if not blend else ("edited" if edited else "source", view_id, "blend")
        cached = self._contribution_cache.get(key)
        if cached is not None:
            return cached
        image, _alpha, meta, _scene = self._render_raw(view_id, edited=edited)
        contribution = self._contribution_from_meta(meta, blend=blend)
        self._image_cache.setdefault(key, image)
        self._contribution_cache[key] = contribution
        return contribution


    def render_excluding(
        self,
        view_id: str,
        gaussian_ids: np.ndarray,
        *,
        edited: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, ContributionMap]:
        """Render the current scene with selected Gaussians made transparent.

        Used by REMOVE/REPLACE to ask whether the scene already contains a
        trustworthy background behind an object before any hallucination.
        """
        image, alpha, meta, _scene = self._render_raw(
            view_id,
            edited=edited,
            excluded_gaussian_ids=gaussian_ids,
        )
        return image, alpha, self._contribution_from_meta(meta)

    def render_only(
        self,
        view_id: str,
        gaussian_ids: np.ndarray,
        *,
        edited: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render only a selected Gaussian set as an isolated QA witness."""
        keep = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
        if keep.size == 0:
            raise ValueError("render_only requires at least one Gaussian")
        image, alpha, _meta, _scene = self._render_raw(view_id, edited=edited, included_gaussian_ids=keep)
        return image, alpha

    def render_only_depth(
        self,
        view_id: str,
        gaussian_ids: np.ndarray,
        *,
        edited: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render expected depth/alpha from only a selected Gaussian set.

        This is the depth equivalent of :meth:`render_only` and is primarily
        used by replacement transport: immutable source-object geometry acts as
        a correspondence scaffold without being rendered into the canonical
        edited scene.
        """
        keep = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
        if keep.size == 0:
            raise ValueError("render_only_depth requires at least one Gaussian")
        _image, alpha, meta, _scene = self._render_raw(view_id, edited=edited, included_gaussian_ids=keep,
                                                       render_mode="RGB+ED")
        depth = meta.get("depth")
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().to(torch.float32).cpu().numpy()
        if not isinstance(depth, np.ndarray):
            raise RuntimeError("render_only_depth did not produce depth")
        return depth.astype(np.float32), alpha.astype(np.float32)

    def _contribution_from_meta(self, meta: dict[str, object], blend: bool = False) -> ContributionMap:
        """Top-K contributors per pixel from the rasterizer's metadata.

        ``blend=True``: true alpha-blending weights alpha_i * prod_{j<i}(1 - alpha_j)
        (what the Gaussian draws). ``blend=False``: the first K Gaussians along
        the ray with rank weights 1/(rank+1) (the legacy v4 maps, whose consumers'
        thresholds were set on them). Both come from one intersection listing."""
        try:
            from gsplat import rasterize_to_indices_in_range
        except ImportError as error:
            raise RuntimeError("SpEdE contribution maps need gsplat.rasterize_to_indices_in_range") from error
        W, H, K = int(meta["width"]), int(meta["height"]), self.top_k_contributors
        isect_offsets = meta["isect_offsets"]
        if isinstance(isect_offsets, torch.Tensor) and isect_offsets.ndim == 3:
            isect_offsets = isect_offsets[0]
        means2d, conics, opacities = meta["means2d"], meta["conics"], meta["opacities"]
        if means2d.ndim == 3:            # [C, N, 2] (not packed): one camera
            means2d, conics, opacities = means2d[0], conics[0], opacities[0]
        # Every Gaussian along each pixel's ray, front to back, starting at
        # full transmittance.
        g_ids, p_ids, _img_ids = rasterize_to_indices_in_range(
            range_start=0, range_end=1000000,
            transmittances=torch.ones((H, W), dtype=torch.float32, device=means2d.device),
            means2d=meta["means2d"], conics=meta["conics"], opacities=meta["opacities"],
            image_width=W, image_height=H, tile_size=int(meta["tile_size"]),
            isect_offsets=isect_offsets, flatten_ids=meta["flatten_ids"],
        )
        packed_scene_ids = meta.get("gaussian_ids")
        scene_ids = packed_scene_ids if isinstance(packed_scene_ids, torch.Tensor) else None
        ids, weights = contribution_weights(g_ids, p_ids, means2d, conics, opacities, width=W, height=H,
                                            top_k=K, blend=blend, scene_ids=scene_ids)
        return ContributionMap(gaussian_ids=ids, weights=weights).validate()

    def candidate_views(self, gaussian_ids: np.ndarray) -> list[CandidateView]:
        target_ids = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
        if target_ids.size == 0:
            return []
        scene = self._scene(True)
        roi_center = scene.means[target_ids].mean(axis=0)
        target_set = set(map(int, target_ids.tolist()))
        candidates: list[CandidateView] = []
        for camera in self.cameras.views:
            if camera.is_virtual or camera.roi_box is not None:
                continue
            contribution = self.contribution_map(camera.view_id, edited=True)
            ids = contribution.gaussian_ids
            weights = contribution.weights
            target_mask = np.isin(ids, target_ids)
            per_pixel_weight = np.where(target_mask, weights, 0.0).sum(axis=-1)
            visible = np.unique(ids[target_mask & (weights > 0)])
            visible = visible[visible >= 0].astype(np.int64)
            projected_area = float(np.mean(per_pixel_weight > 0.01))
            if bool((per_pixel_weight > 0.01).any()):
                mean_support = float(
                    np.clip(per_pixel_weight[per_pixel_weight > 0.01].mean(), 0.0, 1.0)
                )
                occlusion_fraction = 1.0 - mean_support
            else:
                occlusion_fraction = 1.0
            camera_to_world = np.linalg.inv(camera.viewmat)
            camera_center = camera_to_world[:3, 3]
            direction = roi_center - camera_center
            norm = float(np.linalg.norm(direction))
            angular = direction / max(norm, 1e-8)
            candidates.append(
                CandidateView(
                    view_id=camera.view_id,
                    camera_to_world=camera_to_world.astype(np.float32),
                    intrinsics=camera.intrinsic.astype(np.float32),
                    visible_gaussian_ids=visible,
                    projected_area=projected_area,
                    occlusion_fraction=occlusion_fraction,
                    angular_signature=angular.astype(np.float32),
                )
            )
        return candidates
