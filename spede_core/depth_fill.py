"""Edited RGB -> raw monocular depth -> scene-aligned depth -> 3D Gaussian births.

The recovered depth is a *proposal*, never ground truth. Known source-scene
depth is authoritative outside the removal mask; a verified support plane
clamps hallucinated bumps inside flat surfaces. Missing/contradictory evidence
fails closed. No AccGS, model weights, or extra custom node dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .geometry_utils import plane_depth_map, unproject_depth
from .ply import SH_C0, GaussianPly, append_planar_vertices, write_vertices


@dataclass(frozen=True)
class SceneAlignedDepth:
    depth: np.ndarray                   # camera-Z, original camera resolution
    valid: np.ndarray
    anchor_mask: np.ndarray             # real source observations OUTSIDE selection
    selection_mask: np.ndarray
    report: dict[str, Any]


def _depth_array(depth: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    a=np.asarray(depth, np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} differs from camera {shape}")
    return a


def _robust_affine(x: np.ndarray, y: np.ndarray, *,
                   ransac_steps: int = 128, seed: int = 17) -> tuple[float, float, np.ndarray]:
    """Estimate scale and shift with a robust, deterministic inlier test.

    Positive slope only: reversing depth order to improve an affine error
    would be a geometry inversion. Near-constant predictions use scale-only
    instead of fitting an ill-conditioned intercept.
    """
    x=np.asarray(x,np.float64).ravel()
    y=np.asarray(y,np.float64).ravel()
    if x.size<24 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Not enough valid depth anchor samples")
    epsilon=1e-9
    tolerance=np.maximum(0.065*y,0.025*float(np.median(y)))
    rng=np.random.default_rng(seed)
    best_a=float(np.median(y/x))
    best_b=0.0
    best_mask=np.abs(best_a*x-y)<=tolerance
    best_error=float(np.median(np.abs(best_a*x-y)))
    xs=np.percentile(x,[10,90])
    if xs[1]-xs[0] >= max(1e-6,0.025*float(np.median(x))):
        for _ in range(int(ransac_steps)):
            i,j=rng.choice(x.size,2,replace=False)
            if abs(x[i]-x[j])<=max(1e-6,0.01*np.median(x)):
                continue
            a=(y[i]-y[j])/(x[i]-x[j])
            if not np.isfinite(a) or not (0.01<a<100.0):
                continue
            b=y[i]-a*x[i]
            if abs(b)>6*float(np.median(y)):
                continue
            err=np.abs(a*x+b-y)
            inliers=err<=tolerance
            median_error=float(np.median(err[inliers])) if np.any(inliers) else np.inf
            if (inliers.sum()>best_mask.sum() or
                (inliers.sum()==best_mask.sum() and median_error<best_error)):
                best_a,best_b,best_mask,best_error=float(a),float(b),inliers,median_error
    if best_mask.sum()>=24:
        # IRLS on the inlier population, with positive-slope constraint.
        xx=x[best_mask]; yy=y[best_mask]
        for _ in range(5):
            residual=yy-(best_a*xx+best_b)
            t=max(1e-9,1.4826*np.median(np.abs(residual-np.median(residual)))*2.0)
            weight=np.clip(t/np.maximum(np.abs(residual),1e-12),0.01,1.0)
            matrix=np.column_stack([xx,np.ones(len(xx))])
            try:
                candidate=np.linalg.lstsq(matrix*weight[:,None],yy*weight,rcond=None)[0]
                if candidate[0]>0 and np.isfinite(candidate).all() and candidate[0]<100:
                    best_a,best_b=map(float,candidate)
            except np.linalg.LinAlgError:
                break
    if not (np.isfinite(best_a) and np.isfinite(best_b) and best_a>0):
        raise RuntimeError("No positive monotonic depth alignment")
    final=np.abs(best_a*x+best_b-y)<=tolerance
    return best_a,best_b,final


def monotone_depth_map(x: np.ndarray, y: np.ndarray, *, bins: int=24,
                       min_per_bin: int=30) -> tuple[np.ndarray, np.ndarray]:
    """Knots of a nondecreasing map from predicted to scene depth.

    Monocular depth compresses the jump between depth layers (a tabletop in
    front of a far wall), so one scale+shift cannot fit both. Per quantile
    bin of the prediction, the median scene depth is a robust knot; adjacent
    violators are pooled (PAV) so depth order is never reversed.
    """
    x=np.asarray(x,np.float64).ravel(); y=np.asarray(y,np.float64).ravel()
    edges=np.unique(np.quantile(x,np.linspace(0,1,int(bins)+1)))
    kx=[];ky=[];kw=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        k=(x>=lo)&((x<hi) if i<len(edges)-2 else (x<=hi))
        if k.sum()>=min_per_bin:
            kx.append(float(np.median(x[k]))); ky.append(float(np.median(y[k]))); kw.append(float(k.sum()))
    if len(kx)<2:
        raise RuntimeError("Too few depth anchor bins for a monotone depth map")
    blocks=[[v,w,1] for v,w in zip(ky,kw)]
    i=0
    while i<len(blocks)-1:
        if blocks[i][0]>blocks[i+1][0]:
            a,b=blocks[i],blocks[i+1]
            blocks[i]=[(a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]),a[1]+b[1],a[2]+b[2]]
            del blocks[i+1]
            i=max(i-1,0)
        else:
            i+=1
    fitted=np.concatenate([np.full(n,v) for v,_w,n in blocks])
    return np.asarray(kx),fitted


def apply_monotone_depth_map(pred: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Interpolate between knots; outside them, keep the end knot's ratio."""
    p=np.asarray(pred,np.float64)
    out=np.interp(p,kx,ky)
    out=np.where(p<kx[0],p*(ky[0]/kx[0]),out)
    return np.where(p>kx[-1],p*(ky[-1]/kx[-1]),out)


def align_depth_to_scene(
    predicted_depth: np.ndarray,
    source_depth: np.ndarray,
    source_alpha: np.ndarray,
    object_mask: np.ndarray,
    *, predicted_valid: np.ndarray | None=None,
    support_plane_depth: np.ndarray | None=None,
    ring_px: int=32,
    min_anchors: int=80,
    min_inlier_fraction: float=0.52,
    max_median_relative_error: float=0.055,
    plane_clamp_fraction: float=0.045,
    max_plane_disagreement_fraction: float=0.35,
    depth_model: str="auto",
) -> SceneAlignedDepth:
    mask=np.asarray(object_mask,bool)
    if mask.ndim!=2 or not mask.any():
        raise ValueError("Object selection mask must be a nonempty 2D array")
    shape=mask.shape
    pred=_depth_array(predicted_depth,shape,"predicted depth")
    source=_depth_array(source_depth,shape,"source depth")
    alpha=_depth_array(source_alpha,shape,"source alpha")
    if not (4<=ring_px<=256):
        raise ValueError("Depth alignment ring must be 4..256 pixels")
    valid=np.isfinite(pred)&(pred>1e-6)
    if predicted_valid is not None:
        support=np.asarray(predicted_valid,bool)
        if support.shape!=shape:
            raise ValueError("MoGe reliability mask size mismatch")
        valid &= support
    known=np.isfinite(source)&(source>1e-6)&np.isfinite(alpha)&(alpha>=0.65)
    ring=ndimage.binary_dilation(mask,iterations=int(ring_px)) & ~ndimage.binary_dilation(mask,iterations=2)
    anchors=known&valid&ring
    if int(anchors.sum())<int(min_anchors):
        raise RuntimeError(f"Too few depth anchors outside removed region: {int(anchors.sum())} / {min_anchors}")
    xx=pred[anchors]; yy=source[anchors]
    if len(xx)>10000:
        sample=np.linspace(0,len(xx)-1,10000,dtype=np.int64)
        xx=xx[sample];yy=yy[sample]
    if depth_model not in {"auto","affine","monotone"}:
        raise ValueError("depth_model must be auto, affine or monotone")
    def score(mapped):
        error=np.abs(mapped-yy)/np.maximum(yy,1e-6)
        return float((error<=0.075).mean()),float(np.median(error))
    a,b,_=_robust_affine(xx,yy)
    inlier_fraction,median_error=score(a*xx+b)
    model="affine"
    mapping=lambda p: a*p+b
    failed=inlier_fraction<min_inlier_fraction or median_error>max_median_relative_error
    tried={"affine":{"inlier_fraction":inlier_fraction,"median_relative_error":median_error}}
    knots=None
    try:
        kx,ky=monotone_depth_map(xx,yy) if depth_model!="affine" else (None,None)
    except RuntimeError:
        if depth_model=="monotone":
            raise
        kx=None                   # near-constant prediction: affine only
    if kx is not None:
        m_inliers,m_error=score(apply_monotone_depth_map(xx,kx,ky))
        tried["monotone"]={"inlier_fraction":m_inliers,"median_relative_error":m_error}
        # Auto keeps the simpler affine map unless it fails or the monotone
        # one explains clearly more of the ring: an affine fit can pass the
        # gate while missing a whole depth layer around the hole.
        if depth_model=="monotone" or failed or m_inliers>=inlier_fraction+0.05:
            model,inlier_fraction,median_error="monotone",m_inliers,m_error
            mapping=lambda p: apply_monotone_depth_map(p,kx,ky)
            knots=[kx.tolist(),ky.tolist()]
    if inlier_fraction<min_inlier_fraction or median_error>max_median_relative_error:
        raise RuntimeError(
            f"Edited depth does not register to actual scene ({model}): "
            f"{inlier_fraction:.1%} inliers, {median_error:.1%} median relative error"
        )
    aligned=mapping(pred)
    aligned=np.where(valid&(aligned>1e-6)&np.isfinite(aligned),aligned,np.nan)
    if int(np.count_nonzero(mask&np.isfinite(aligned)))<max(16,int(mask.sum()*0.65)):
        raise RuntimeError("Depth model left too many removed pixels invalid")
    plane_used=False
    plane_median_disagreement=None
    plane_correction=None
    if support_plane_depth is not None:
        plane=_depth_array(support_plane_depth,shape,"support plane depth")
        # The plane is fitted to Gaussian centres, which sit behind the
        # surface a camera renders (splats have thickness). Births on the
        # raw plane then show up behind the visible support, displaced in
        # parallax. Scale it along the rays onto the rendered ring surface:
        # for a plane, a constant depth factor is again a parallel plane.
        on_plane=known&ring&np.isfinite(plane)&(plane>0)
        rel=(source[on_plane]-plane[on_plane])/plane[on_plane]
        rel=rel[np.abs(rel)<0.15]
        plane_correction=None
        if rel.size>=int(min_anchors):
            plane_correction=float(np.median(rel))
            plane=plane*(1.0+plane_correction)
        suitable=mask&np.isfinite(plane)&(plane>0)&np.isfinite(aligned)
        if suitable.sum()<16:
            raise RuntimeError("Support plane has insufficient ray intersections")
        diff=np.abs(aligned[suitable]-plane[suitable])/plane[suitable]
        plane_median_disagreement=float(np.median(diff))
        if plane_median_disagreement>max_plane_disagreement_fraction:
            raise RuntimeError(
                "Edited-image depth disagrees with known planar scene geometry "
                f"({plane_median_disagreement:.1%} median)"
            )
        # One coherent floor/table/wall: the source plane is the primary
        # depth authority. Monocular detail is kept only within a small band.
        deviation=np.clip(aligned-plane,
                          -plane_clamp_fraction*plane,
                          plane_clamp_fraction*plane)
        aligned=np.where(suitable,plane+0.2*deviation,aligned)
        plane_used=True
    # Absolute source depth is never revised outside the old object mask.
    aligned=np.where(~mask&known,source,aligned)
    valid=np.isfinite(aligned)&(aligned>0)
    report={
        "name":"SpEdE edited RGB depth alignment",
        "passed":True,
        "input_semantics":"MoGe camera-z proposal, mapped to source-rendered depth",
        "depth_model":model,"models_tried":tried,"monotone_knots":knots,
        "scale":a,"shift":b,"anchor_pixels":int(anchors.sum()),
        "inlier_fraction":inlier_fraction,
        "median_relative_error":median_error,
        "source_scene_authoritative_outside_mask":True,
        "plane_prior_used":plane_used,
        "plane_median_disagreement":plane_median_disagreement,
        "plane_visible_surface_correction":plane_correction,
        "valid_selected_fraction":float(np.mean(valid[mask])),
        "model_used_for_geometry":True,
    }
    return SceneAlignedDepth(aligned.astype(np.float32),valid,anchors,mask,report)


def _frames_from_depth(z: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray):
    """World-space per-pixel X and Y tangent vectors, normals and footprints."""
    world=unproject_depth(z,intrinsic,np.linalg.inv(np.asarray(viewmat,np.float64)))
    # Axis=1: pixel x; axis=0: pixel y. 3D splat scale follows these metrics.
    dx=np.gradient(world,axis=1)
    dy=np.gradient(world,axis=0)
    lengths_x=np.linalg.norm(dx,axis=-1)
    tangent=dx/np.maximum(lengths_x[...,None],1e-12)
    normal=np.cross(dx,dy)
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
    bitangent=np.cross(normal,tangent)
    return world,dx,dy,normal,tangent,bitangent


def max_observed_weight(renderer, view_ids, count: int) -> np.ndarray:
    """Largest alpha-blending weight each source Gaussian draws in any of the
    capture cameras, rendered from the ORIGINAL scene (object in place).

    Near zero means no camera ever saw the Gaussian: the reconstruction
    invented its colour (support under an object, its shadowed contact).
    """
    seen=np.zeros(int(count),np.float32)
    for view_id in view_ids:
        contribution=renderer.contribution_map(view_id,edited=False,blend=True)
        ids=np.asarray(contribution.gaussian_ids).reshape(-1)
        weights=np.asarray(contribution.weights,np.float32).reshape(-1)
        keep=(ids>=0)&(ids<len(seen))
        np.maximum.at(seen,ids[keep],weights[keep])
    return seen


def carving_exposed(alpha_before: np.ndarray, depth_before: np.ndarray,
                    alpha_after: np.ndarray, depth_after: np.ndarray,
                    *, alpha_drop: float=0.1, depth_step: float=0.03) -> np.ndarray:
    """Pixels whose render carving changed: coverage fell or the visible
    surface moved back. Births behind the clean surface are allowed there."""
    a0=np.asarray(alpha_before,np.float64); a1=np.asarray(alpha_after,np.float64)
    z0=np.asarray(depth_before,np.float64); z1=np.asarray(depth_after,np.float64)
    with np.errstate(invalid="ignore"):
        moved=np.isfinite(z0)&np.isfinite(z1)&(z0>0)&(z1>z0*(1+float(depth_step)))
    return (a1<a0-float(alpha_drop))|moved


def _unit_rgb(rgb: np.ndarray) -> np.ndarray:
    a=np.asarray(rgb,np.float32)[...,:3]
    return a/255.0 if a.max()>1.5 else a


def _mkl_transfer(source_cov: np.ndarray, target_cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """The linear map taking a colour distribution with ``source_cov`` to
    one with ``target_cov`` with the least change (Monge-Kantorovich, as in
    the common ColorMatch "mkl" method)."""
    ea, ua = np.linalg.eigh(source_cov)
    da = np.sqrt(np.maximum(ea, eps))
    c = (da[:, None] * ua.T) @ target_cov @ (ua * da[None, :])
    ec, uc = np.linalg.eigh(c)
    dc = np.sqrt(np.maximum(ec, 0.0))
    inv = ua / da[None, :]
    return inv @ (uc * dc[None, :]) @ uc.T @ inv.T


def match_edit_tone(edited_rgb: np.ndarray, scene_rgb: np.ndarray, selection: np.ndarray, *,
                    margin_px: int = 24, limits: tuple[float, float] = (0.7, 1.4),
                    method: str = "mkl") -> tuple[np.ndarray, dict]:
    """The cleaned image in the scene's colours, matched on the pixels well
    away from the selection, which the edit was not asked to change (the
    object itself would pull the match towards its own colours). ``mkl``
    matches the colour mean and covariance (exposure, contrast, cast);
    ``gain`` one per-channel gain. Qwen lifts the whole image; playroom
    (24 Oct) came out 14 % lighter and its births showed as light patches
    next to the kept wall. A map stretching any colour axis outside
    ``limits`` (a flat, near-grey far region has no covariance to match)
    falls back to the gain."""
    img = _unit_rgb(edited_rgb).astype(np.float64)
    ref = _unit_rgb(scene_rgb).astype(np.float64)
    if ref.shape != img.shape:
        raise ValueError("Scene render must match the edited image")
    far = ~ndimage.binary_dilation(np.asarray(selection, bool), iterations=int(margin_px))
    far &= (img.min(-1) > 0.03) & (img.max(-1) < 0.97) & (ref.min(-1) > 0.03)
    report = {"far_pixels": int(far.sum())}
    if far.sum() < 500:
        return img.astype(np.float32), {**report, "applied": False}
    gain = np.clip(np.median(ref[far], 0) / np.maximum(np.median(img[far], 0), 1e-3), *limits)
    report["gain"] = np.round(gain, 4).tolist()
    if method == "mkl":
        src, dst = img[far], ref[far]
        transfer = _mkl_transfer(np.cov(src.T) + 1e-5 * np.eye(3), np.cov(dst.T) + 1e-5 * np.eye(3))
        stretch = np.linalg.svd(transfer, compute_uv=False)
        report["mkl_stretch"] = np.round(stretch, 4).tolist()
        if stretch.min() >= limits[0] and stretch.max() <= limits[1]:
            out = (img - src.mean(0)) @ transfer.T + dst.mean(0)
            return np.clip(out, 0, 1).astype(np.float32), {**report, "applied": True, "method": "mkl"}
    return np.clip(img * gain, 0, 1).astype(np.float32), {**report, "applied": True, "method": "gain"}


def shadow_halo(selection: np.ndarray, remaining_rgb: np.ndarray, edited_rgb: np.ndarray,
                *, halo_px: int=24, colour_tolerance: float=0.15, lift: float=0.06) -> np.ndarray:
    """Pixels just outside the selection that the cleaned image brightens.

    SAM selects the object, not its shadow; Qwen removes both. Left alone the
    shadow stays in the splat as a dark seam around the patch (kitchen: the
    bulldozer's shadow, 4-24 px outside the mask at 512 px). A pixel is in
    the halo when it lies within ``halo_px`` of the selection, the render
    disagrees with the cleaned image, the cleaned image is brighter there,
    and its region touches the selection.
    """
    sel=np.asarray(selection,bool)
    rem=_unit_rgb(remaining_rgb); img=_unit_rgb(edited_rgb)
    if rem.shape!=img.shape or rem.shape[:2]!=sel.shape:
        raise ValueError("Halo render, cleaned image and selection sizes differ")
    if halo_px<=0 or not sel.any():
        return np.zeros(sel.shape,bool)
    off=ndimage.median_filter(np.abs(rem-img).mean(axis=-1),size=5)
    brighter=ndimage.median_filter(img.mean(axis=-1)-rem.mean(axis=-1),size=5)
    near=~sel&(ndimage.distance_transform_edt(~sel)<=float(halo_px))
    candidate=near&(off>float(colour_tolerance))&(brighter>float(lift))
    labels,_n=ndimage.label(candidate|sel)
    touching=np.unique(labels[sel]); touching=touching[touching>0]
    return candidate&np.isin(labels,touching)


def edit_spill(selection: np.ndarray, render_rgb: np.ndarray, edited_rgb: np.ndarray,
               *, halo_px: int=24, colour_tolerance: float=0.2) -> np.ndarray:
    """Pixels the cleaned image changed beyond the selection and its shadow
    halo, in regions touching it: Qwen removed more than was selected.

    The name drives Qwen, the mask drives the 3D removal. When the name
    covers more (bonsai "tree in a pot" with only the tree selected; kitchen
    "bucket" where Qwen removed most of the bulldozer), the scene keeps what
    the hero picture lost: the depth fit fails or the patch disagrees with
    every other camera. Measured on the 18 Oct UI runs (5 px median colour
    difference > 0.2): good runs 0-0.6 % of the selection, these 23-190 %.
    """
    sel=np.asarray(selection,bool)
    rem=_unit_rgb(render_rgb); img=_unit_rgb(edited_rgb)
    if rem.shape!=img.shape or rem.shape[:2]!=sel.shape:
        raise ValueError("Render, cleaned image and selection sizes differ")
    if not sel.any():
        return np.zeros(sel.shape,bool)
    off=ndimage.median_filter(np.abs(rem-img).mean(axis=-1),size=5)
    near=ndimage.binary_dilation(sel,iterations=max(int(halo_px),1))
    changed=(off>float(colour_tolerance))&~near
    labels,_n=ndimage.label(changed|near)
    touching=np.unique(labels[near]); touching=touching[touching>0]
    return changed&np.isin(labels,touching)


def carve_residue_ids(
    scene: GaussianPly, object_means: np.ndarray, camera, object_mask: np.ndarray,
    clean_depth: np.ndarray, *, observed_weight: np.ndarray | None=None,
    front_tolerance: float=0.04, visible_layer: float | None=None,
    unseen_below: float=0.02, bbox_margin: float=0.10, mask_dilate_px: int=2,
    spill_px: int=24, halo: np.ndarray | None=None,
    halo_depth: np.ndarray | None=None, halo_band: float=0.06,
) -> tuple[np.ndarray, dict]:
    """Leftover Gaussians that the fill replaces instead of keeping.

    Removing a selection leaves residue: unselected underside and interior
    pieces, and the support surface under the object, whose colour no camera
    saw. Seen from the hero camera it is many semi-dark layers stacked under
    the clean surface plus large splats spreading in from the side, so no
    per-Gaussian colour test separates it from good surface. With the
    scene-aligned depth of the cleaned hero image as the clean surface, a
    remaining Gaussian is carved when its centre projects inside the
    (slightly dilated) selection and

    * ``in_front``: it is more than ``front_tolerance`` in front of the clean
      surface and inside the old object's 3D bounding box grown by
      ``bbox_margin`` (real foreground away from the object stays), or
    * ``visible_layer`` (off by default): it is at most ``visible_layer``
      behind the clean surface, so the whole visible layer of the hole is
      replaced by births from the cleaned image. That also removes surface
      other cameras saw correctly; on kitchen and bonsai it degraded the
      side views, so it is only an option, or
    * ``unseen``: its largest weight in any capture camera of the original
      scene is below ``unseen_below`` (no capture view changes when it goes,
      at any depth),

    or, ``spill``: its centre lies within ``spill_px`` outside the selection,
    in the visible layer, and its projected 2-sigma radius reaches more than
    twice its distance into the selection (mostly drawn inside the hole),
    or, ``halo``: its centre lies in the shadow halo (see ``shadow_halo``) no
    more than ``halo_band`` behind the rendered surface ``halo_depth`` there:
    the shadow's own Gaussians, so it goes from every camera, not only the
    hero.
    """
    obj=np.asarray(object_means,np.float64)
    if obj.ndim!=2 or obj.shape[1]!=3 or len(obj)<3:
        raise ValueError("Object means must be Nx3 with at least 3 rows")
    if not (0<front_tolerance<0.5) or not (visible_layer is None or 0<visible_layer<1) or not (0<=bbox_margin<=1):
        raise ValueError("Invalid carve tolerance, visible layer or bounding-box margin")
    from .geometry_utils import project_to_pixels
    selection=np.asarray(object_mask,bool)
    clean=np.asarray(clean_depth,np.float64)
    if clean.shape!=selection.shape:
        raise ValueError("Clean depth and selection mask sizes differ")
    mask=ndimage.binary_dilation(selection,iterations=int(mask_dilate_px)) if mask_dilate_px>0 else selection
    means=np.asarray(scene.means,np.float64)
    if observed_weight is not None and len(observed_weight)!=len(means):
        raise ValueError("Observed weights do not match the scene's Gaussians")
    uv,z=project_to_pixels(means,camera.viewmat,camera.intrinsic)
    h,w=mask.shape
    pix=np.isfinite(uv).all(axis=1)&(z>0)
    x=np.full(len(means),-1,np.int64); y=np.full(len(means),-1,np.int64)
    x[pix]=np.floor(uv[pix,0]).astype(np.int64); y[pix]=np.floor(uv[pix,1]).astype(np.int64)
    pix&=(x>=0)&(y>=0)&(x<w)&(y<h)
    # Distance (px) to the selection; the clean depth there is the nearest
    # selected pixel's, so spill splats are judged against the hole surface.
    distance,(ny,nx)=ndimage.distance_transform_edt(~selection,return_indices=True)
    surface=np.full(len(means),np.nan)
    surface[pix]=clean[ny[y[pix],x[pix]],nx[y[pix],x[pix]]]
    pix&=np.isfinite(surface)&(surface>0)
    ok=pix.copy(); ok[pix]=mask[y[pix],x[pix]]
    lo,hi=np.percentile(obj,0.5,axis=0),np.percentile(obj,99.5,axis=0)
    grow=np.maximum((hi-lo)*float(bbox_margin),1e-6)
    in_box=np.all((means>=lo-grow)&(means<=hi+grow),axis=1)
    in_front=ok&in_box&(z<surface*(1-float(front_tolerance)))
    layer=np.zeros(len(means),bool); spill=np.zeros(len(means),bool)
    if visible_layer is not None:
        near=z<surface*(1+float(visible_layer))
        layer=ok&near
        if spill_px>0:
            radius=2.0*np.exp(np.asarray(scene.log_scales,np.float64).max(axis=1))*float(camera.intrinsic[0][0])/np.maximum(z,1e-9)
            d=np.full(len(means),np.inf); d[pix]=distance[y[pix],x[pix]]
            spill=pix&~ok&near&(d<=float(spill_px))&(radius>2.0*d)
    shadow=np.zeros(len(means),bool)
    if halo is not None:
        hmask=np.asarray(halo,bool)&~mask
        hz=np.asarray(halo_depth,np.float64)
        if hmask.shape!=mask.shape or hz.shape!=mask.shape:
            raise ValueError("Halo and its depth must match the selection")
        inside=pix.copy(); inside[pix]=hmask[y[pix],x[pix]]
        ref=np.full(len(means),np.nan); ref[inside]=hz[y[inside],x[inside]]
        with np.errstate(invalid="ignore"):
            shadow=inside&np.isfinite(ref)&(z<ref*(1+float(halo_band)))
    unseen=np.zeros(len(means),bool)
    if observed_weight is not None:
        unseen=ok&(np.asarray(observed_weight)<float(unseen_below))
    ids=np.flatnonzero(in_front|layer|unseen|spill|shadow)
    report={"carved_gaussians":int(len(ids)),"in_selection":int(ok.sum()),
            "in_front_of_clean_surface":int(in_front.sum()),
            "visible_layer":int(layer.sum()),"spill":int(spill.sum()),"unseen":int(unseen.sum()),
            "shadow_halo":int(shadow.sum()),
            "visibility_used":observed_weight is not None,
            "front_tolerance":float(front_tolerance),"visible_layer_depth":visible_layer,
            "unseen_below":float(unseen_below),"bbox_margin":float(bbox_margin),"spill_px":int(spill_px)}
    return ids,report


def create_depth_patch(
    removed_scene_ply: Path, camera, edited_rgb: np.ndarray,
    aligned: SceneAlignedDepth, source_depth: np.ndarray,
    source_alpha: np.ndarray, output_ply: Path,
    *, pixel_stride:int=4, max_births:int=30000,
    alpha_missing_below:float=0.82,
    behind_tolerance:float|None=0.03,
    exposed: np.ndarray | None=None,
    remaining_rgb: np.ndarray | None=None,
    colour_tolerance: float=0.15,
    colour_front_band: float=0.06,
    depth_edge_step: float | None=0.03,
    max_footprint_ratio: float=2.5,
    max_grazing_stretch: float=3.0,
    forbidden: np.ndarray | None=None,
    halo: np.ndarray | None=None,
    halo_depth: np.ndarray | None=None,
    max_relative_neighbor_step:float=0.09,
    min_births:int=12,
    birth_rgb: np.ndarray | None=None,
) -> dict:
    """Add only genuinely unseen RGB pixels as depth-grounded 3D Gaussians."""
    from .patch import _quaternions_from_frames
    scene=GaussianPly.load(removed_scene_ply)
    z=np.asarray(aligned.depth,np.float64)
    mask=np.asarray(aligned.selection_mask,bool)
    if z.shape!=(int(camera.height),int(camera.width)):
        raise ValueError("Aligned depth does not match selected camera")
    image=np.asarray(edited_rgb,np.float32)
    if image.shape!=(*z.shape,3):
        raise ValueError("Edited RGB must be HxWx3 at source camera resolution")
    if not np.isfinite(image).all() or image.min()<0:
        raise ValueError("Invalid edited RGB pixels")
    if image.max()>1:
        if image.max()>255:
            raise ValueError("Edited RGB out of range")
        image=image/255.
    source_z=_depth_array(source_depth,z.shape,"rendered depth")
    alpha=_depth_array(source_alpha,z.shape,"rendered alpha")
    if not (1<=pixel_stride<=64):
        raise ValueError("Pixel stride must be 1..64")
    if not (0<alpha_missing_below<1):
        raise ValueError("alpha_missing_below must be in (0,1)")
    if max_births<min_births or max_births>200000:
        raise ValueError("Invalid max_births")
    if np.count_nonzero(mask)<min_births:
        raise RuntimeError("Selected region too small for depth patch")
    valid=np.asarray(aligned.valid,bool)
    region=mask
    if halo is not None:
        # The shadow halo outside the selection is born on the aligned clean
        # depth (fitted on exactly this ring; smooth), coloured from the clean
        # image. The rendered depth is only a fallback: a splat render's depth
        # is noisy, its normals tilted halo discs edge-on (a pale smooth band).
        halo=np.asarray(halo,bool)&~mask
        hz=_depth_array(halo_depth,z.shape,"halo depth")
        fallback=halo&~valid&np.isfinite(hz)&(hz>0)
        z=np.where(fallback,hz,z); valid=valid|fallback
        halo&=valid; region=mask|halo
    # Filter geometry disagreements before converting to a Gaussian.
    v=valid&region
    finite=np.isfinite(source_z)&(source_z>0)&np.isfinite(alpha)
    # Already observed background pixels remain untouched even if Qwen
    # paints a different texture in that area, unless what remains there
    # lies clearly behind the clean surface (carved residue exposed a lower
    # layer): then the surface itself is missing.
    covered=finite&(alpha>=alpha_missing_below)
    if behind_tolerance is not None:
        behind=source_z>z*(1+float(behind_tolerance))
        if exposed is not None:
            # Only where carving uncovered something: monocular depth alone
            # is too noisy (far walls, gardens) to overrule a kept surface.
            behind &= np.asarray(exposed,bool)
        covered &= ~behind
    if remaining_rgb is not None:
        # What remains is drawn wrong (residue: dark, smeared) where its
        # render disagrees with the cleaned image; a kept surface Qwen left
        # alone agrees and keeps its seen Gaussians (side views stay right).
        # Only where the remaining surface is not in front of the clean one,
        # so births never float in front of real geometry.
        rem=np.asarray(remaining_rgb,np.float32)
        if rem.shape!=image.shape:
            raise ValueError("Remaining render must match the edited image")
        if rem.max()>1.5:
            rem=rem/255.0
        off=ndimage.median_filter(np.abs(rem-image).mean(axis=-1),size=5)
        wrong=(off>float(colour_tolerance))&(source_z>=z*(1-float(colour_front_band)))
        covered &= ~wrong
    if halo is not None:
        covered &= ~halo
    v &= ~covered
    if forbidden is not None:
        # Pixels another camera's births already draw: a later view must not
        # stack its own guess of the same surface on top of them.
        v &= ~np.asarray(forbidden,bool)
    # Never grow through foreground Gaussians at smaller camera-z.
    v &= ~(finite&(alpha>=0.4)&(source_z<z*0.98))
    smooth=ndimage.median_filter(np.where(valid,z,0),size=3)
    v &= np.abs(smooth-z)/np.maximum(z,1e-6)<max_relative_neighbor_step
    # Invalid pixels take the nearest valid depth: a constant fill made the
    # gradients, normals and footprints of every border birth jump.
    good=valid&np.isfinite(z)&(z>0)
    if not good.any():
        raise RuntimeError("No valid aligned depth")
    _d,(fy_,fx_)=ndimage.distance_transform_edt(~good,return_indices=True)
    world,dx,dy,n,t1,t2=_frames_from_depth(
        z[fy_,fx_].astype(np.float64),
        camera.viewmat,camera.intrinsic
    )
    # Normals/radii become unstable at depth discontinuities.
    grad=np.maximum(np.abs(np.gradient(z,axis=0)),
                    np.abs(np.gradient(z,axis=1)))
    v &= np.isfinite(grad)&(grad/np.maximum(z,1e-6)<max_relative_neighbor_step)
    # Pixels next to a depth jump of the clean surface (a table edge in front
    # of a far wall) carry the near colour at the far depth or vice versa:
    # births there float as discs in the air seen from other cameras.
    if depth_edge_step is not None:
        edge=np.isfinite(grad)&(grad/np.maximum(z,1e-6)>=float(depth_edge_step))
        # The selection's border is where the clean depth meets the source
        # depth: a step there is the seam the births must cover, not an edge.
        band=lambda m: ndimage.binary_dilation(m,iterations=3)&~ndimage.binary_erosion(m,iterations=3)
        border=band(mask)|band(region)
        edge&=~border
        v &= ~ndimage.binary_dilation(edge,iterations=int(pixel_stride))
    h,w=mask.shape
    ys,xs=np.mgrid[0:h:pixel_stride,0:w:pixel_stride]
    ys=ys.ravel();xs=xs.ravel()
    eligible=v[ys,xs]
    ys=ys[eligible];xs=xs[eligible]
    if len(xs)>max_births:
        picks=np.linspace(0,len(xs)-1,max_births,dtype=np.int64)
        ys=ys[picks];xs=xs[picks]
    if len(xs)<min_births:
        raise RuntimeError(
            f"No sufficiently reliable missing-depth pixels for 3D birth ({len(xs)} < {min_births})"
        )
    points=world[ys,xs]
    normal=n[ys,xs]
    tangent=t1[ys,xs]
    bitangent=t2[ys,xs]
    valid=np.isfinite(points).all(axis=1)&np.isfinite(normal).all(axis=1)
    valid &= np.linalg.norm(normal,axis=1)>0.85
    ys=ys[valid];xs=xs[valid];points=points[valid]
    normal=normal[valid];tangent=tangent[valid];bitangent=bitangent[valid]
    if len(points)<min_births:
        raise RuntimeError("Insufficient valid depth surface normals")
    # Each birth covers its pixel's footprint on the surface: along the
    # tangent (pixel x) and the bitangent (in-plane, across). On a grazing
    # tabletop the across extent is 3-5x the x extent; a round disc of the
    # larger one smeared the kitchen mat's ribs into a flat film.
    ex=np.linalg.norm(dx,axis=-1)
    ey=np.maximum(np.abs(np.einsum("hwi,hwi->hw",dy,t2)),ex)
    # A surface's footprint changes smoothly; single spikes (seams, depth
    # noise) are capped at 1.5x the median of the birth region around them.
    extents=[]
    for e in (ex,ey):
        windows=np.lib.stride_tricks.sliding_window_view(
            np.pad(np.where(v,e,np.nan),3,constant_values=np.nan),(7,7))[ys,xs]
        local=np.nanmedian(windows.reshape(len(ys),-1),axis=1)
        extents.append(np.minimum(e[ys,xs],1.5*local))
    radii=np.c_[extents[0],extents[1]]*float(pixel_stride)*0.68
    radii=np.clip(radii,1e-5,np.percentile(radii,95,axis=0)*1.5)
    # Absolute cap: a few pixel footprints at the birth's depth; the across
    # axis may stretch further on grazing surfaces (to ~5 degrees).
    # Depth-gradient spikes gave 13-17 footprints (kitchen, bonsai).
    footprint=float(pixel_stride)*z[ys,xs]/float(np.asarray(camera.intrinsic)[0][0])
    radii[:,0]=np.minimum(radii[:,0],float(max_footprint_ratio)*footprint)
    radii[:,1]=np.minimum(radii[:,1],float(max_footprint_ratio)*float(max_grazing_stretch)*footprint)
    quaternions=_quaternions_from_frames(tangent,bitangent,normal)
    output=append_planar_vertices(
        scene.path,output_ply,points,(image if birth_rgb is None else _unit_rgb(birth_rgb))[ys,xs],radii,
        opacity=0.93,thickness_fraction=0.14,quaternions=quaternions
    )
    output.update({
        "method":"moge3_edited_image_scene_aligned_depth",
        "not_general_volume_completion":True,
        "source_depth_authoritative":True,
        "birth_pixels_only":True,
        "g2_depth_alignment":dict(aligned.report),
        "reliable_birth_pixels":int(np.count_nonzero(v)),
        "shadow_halo_pixels":int(np.count_nonzero(halo)) if halo is not None else 0,
        "born_gaussians":int(len(points)),
        "g3_render_required":True,
    })
    return output


def feather_seam_colours(
    means: np.ndarray, dc: np.ndarray, born: np.ndarray, camera,
    selection: np.ndarray, target_rgb: np.ndarray, rendered_rgb: np.ndarray,
    hero_weight: np.ndarray, *, band_px: int, birth_px: int=1,
    smooth_px: float | None=None, full_weight: float=0.25, min_weight: float=0.02,
) -> tuple[np.ndarray, dict]:
    """Kept Gaussians' DC colours near the patch, pulled towards the cleaned
    image so the seam between births and the kept scene fades out.

    Every hero pixel is drawn either by births (the cleaned image's colours)
    or by kept Gaussians (the scene's), so a local tone difference shows as a
    hard edge: kitchen's wall is darker near the removed object (its shadow)
    and the fill sat in it as a lighter V on the orange mat; playroom's kept
    blurry surface inside the selection made a patchwork on wall and floor.
    The inner region is the selection plus the pixels the births draw (their
    centres in the hero view, closed over ``birth_px``, the birth stride).
    Each kept Gaussian whose centre lies in it or within ``band_px`` outside
    moves by the hero render's residual (cleaned minus rendered, smoothed
    over ``smooth_px`` on the kept pixels only so the births' zero residual
    does not dilute the seam), times a ramp 1 -> 0 across the band, times
    how much it draws in the hero view (``hero_weight``: its largest blend
    weight there, full from ``full_weight``, none below ``min_weight``).
    A shift by the residual rather than a lerp to the pixel colour leaves a
    kept Gaussian alone where the render already agrees (its texture stays),
    and the residual includes the view-dependent SH, so ``rest`` is kept:
    scaling it would change the other cameras' highlights for no gain here.
    Only DC colours change, never geometry or opacity. ``dc``: N x 3.
    Returns the new DC and a report."""
    xyz=np.asarray(means,np.float64)
    dc=np.asarray(dc,np.float32).reshape(len(xyz),3)
    born=np.asarray(born,bool).reshape(-1)
    seen=np.asarray(hero_weight,np.float64).reshape(-1)
    sel=np.asarray(selection,bool)
    h,w=sel.shape
    if born.shape!=(len(xyz),) or seen.shape!=(len(xyz),):
        raise ValueError("Birth flags and hero weights must have one value per Gaussian")
    if (int(camera.height),int(camera.width))!=(h,w):
        raise ValueError("Selection does not match the hero camera")
    target=_unit_rgb(target_rgb).astype(np.float64)
    render=_unit_rgb(rendered_rgb).astype(np.float64)
    if target.shape!=(h,w,3) or render.shape!=(h,w,3):
        raise ValueError("Cleaned image and render must match the selection")
    report={"band_px":int(band_px),"birth_px":int(birth_px)}
    if band_px<=0:
        return dc.copy(),{**report,"applied":False}
    from .visual_hull import project_points
    x,y,front=project_points(xyz,camera.viewmat,camera.intrinsic,w,h)
    births=np.zeros((h,w),bool)
    births[y[born&front],x[born&front]]=True
    if birth_px>1:
        births=ndimage.binary_closing(births,iterations=int(birth_px))
        births=ndimage.binary_dilation(births,iterations=max(1,int(birth_px)//2))
    inner=sel|births
    if not inner.any():
        return dc.copy(),{**report,"applied":False,"reason":"no selection or births in view"}
    ramp=np.clip(1-ndimage.distance_transform_edt(~inner)/float(band_px),0,1)
    sigma=float(smooth_px) if smooth_px is not None else max(1.0,band_px/8)
    kept=~births
    den=ndimage.gaussian_filter(kept.astype(np.float64),sigma)
    residual=target-render
    for c in range(3):
        num=ndimage.gaussian_filter(np.where(kept,residual[...,c],0.0),sigma)
        residual[...,c]=np.where(den>0.05,num/np.maximum(den,1e-6),residual[...,c])
    visible=np.clip(seen/float(full_weight),0,1)*(seen>=float(min_weight))
    gain=np.where(~born&front,ramp[y,x]*visible,0.0)
    moved=gain>0
    old=0.5+SH_C0*dc[moved].astype(np.float64)
    new=old+gain[moved,None]*residual[y[moved],x[moved]]
    # Never push a colour past the displayable range it started in.
    new=np.clip(new,np.minimum(old,0),np.maximum(old,1))
    out=dc.copy()
    out[moved]=((new-0.5)/SH_C0).astype(np.float32)
    report.update({
        "applied":True,"smooth_px":round(sigma,2),
        "inner_pixels":int(inner.sum()),"band_pixels":int(((ramp>0)&~inner).sum()),
        "recoloured_gaussians":int(moved.sum()),
        "recoloured_inside_selection":int((moved&sel[y,x]).sum()),
        "mean_colour_shift":round(float(np.abs(new-old).mean()),4) if moved.any() else 0.0,
    })
    return out,report


def write_dc(scene: GaussianPly, dc: np.ndarray, output: Path, message: str) -> Path:
    """``scene`` with its DC colours replaced, everything else as it was."""
    rows=scene.vertex.copy()
    dc=np.asarray(dc,np.float32).reshape(len(rows),3)
    for i in range(3):
        rows[f"f_dc_{i}"]=dc[:,i]
    write_vertices(scene,output,rows,message)
    return Path(output).expanduser().resolve()
