"""Replace-v1 CPU geometry and image asset preparation for local Comfy.

Qwen does scene-aware insertion; SAM supplies NEW object segmentation.
TripoSplat inference is a SEPARATE, officially supplied Comfy workflow.
These helpers prepare the image, placement prior, and optional Graphdeco PLY
merge; they do not claim to run TripoSplat or place shadows in 3D.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy import ndimage

from .ply import GaussianPly, write_vertices
from .geometry_utils import project_to_pixels


def image_object_assets(
    edited_rgb: np.ndarray, clean_background_rgb: np.ndarray,
    new_object_mask: np.ndarray, output_dir: Path, *,
    crop_size: int=512, padding_fraction: float=0.15,
    shadow_radius_px: int=32,
) -> dict:
    """Save alpha cutout and *candidate* shadow mask separately.

    Shadow is not a verified radiometric layer or a completed 3D decal.
    Restrict differences outside the object to its near-contact region.
    """
    new=np.asarray(edited_rgb,np.float32)
    old=np.asarray(clean_background_rgb,np.float32)
    m=np.asarray(new_object_mask,bool)
    if new.shape!=old.shape or new.ndim!=3 or new.shape[-1]!=3:
        raise ValueError("Replacement and background RGB must match H,W,3")
    if new.shape[:2]!=m.shape or not m.any():
        raise ValueError("Replacement SAM segmentation mask missing or dimension mismatch")
    if not np.isfinite(new).all() or not np.isfinite(old).all():
        raise ValueError("Nonfinite RGB replacement pixels")
    if min(float(new.min()),float(old.min()))<0 or max(float(new.max()),float(old.max()))>255:
        raise ValueError("RGB value outside [0,255]")
    if max(float(new.max()),float(old.max()))<=1.0:
        new=new*255.;old=old*255.
    h,w=m.shape
    yy,xx=np.nonzero(m)
    x0,x1=int(xx.min()),int(xx.max()+1)
    y0,y1=int(yy.min()),int(yy.max()+1)
    cx=(x0+x1)/2;cy=(y0+y1)/2
    side=max(x1-x0,y1-y0)*(1+2*float(padding_fraction))
    side=max(8,int(np.ceil(side)))
    ix0=int(np.floor(cx-side/2));iy0=int(np.floor(cy-side/2))
    rgba=np.zeros((side,side,4),dtype=np.uint8)
    for ya in range(side):
        srcy=iy0+ya
        if 0<=srcy<h:
            sx0=max(0,ix0);sx1=min(w,ix0+side)
            if sx1>sx0:
                rgba[ya,sx0-ix0:sx1-ix0,:3]=np.rint(new[srcy,sx0:sx1]).astype(np.uint8)
                rgba[ya,sx0-ix0:sx1-ix0,3]=m[srcy,sx0:sx1].astype(np.uint8)*255
    # No shadow pixels from inside the object cutout. A conservative,
    # positive luminance loss beyond its silhouette is only a candidate.
    luminance=np.array([0.2126,0.7152,0.0722])
    dark=np.clip((old-new)@luminance/255.,0,1)
    close=ndimage.binary_dilation(m,iterations=int(shadow_radius_px))&~m
    score=ndimage.gaussian_filter(np.where(close,dark,0),sigma=2)
    shadow=np.where(close,np.clip(score,0,1),0).astype(np.float32)
    out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    crop=out/"replacement_object_rgba.png"
    alpha=out/"replacement_mask.png"
    candidate=out/"contact_shadow_candidate.png"
    Image.fromarray(rgba,"RGBA").resize(
        (crop_size,crop_size),Image.Resampling.LANCZOS
    ).save(crop)
    Image.fromarray(m.astype(np.uint8)*255,"L").save(alpha)
    Image.fromarray(np.rint(shadow*255).astype(np.uint8),"L").save(candidate)
    meta={
        "status":"replacement_asset_prepared",
        "image_object_rgba":str(crop),
        "mask_png":str(alpha),
        "shadow_candidate_png":str(candidate),
        "shadow_is_only_candidate":True,
        "bbox_xyxy":[x0,y0,x1,y1],
        "pixel_center":[float(xx.mean()),float(yy.mean())],
        "crop_size":int(crop_size),
        "tripo_inference_completed":False,
    }
    (out/"replacement_assets.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta


def replacement_prior(
    old_scene_ply: Path,
    source_gaussian_ids: np.ndarray,
    camera,
    new_object_mask: np.ndarray,
    *,
    minimum_points: int=4,
    scale_multiplier: float=1.0,
) -> dict:
    """Use actual removed object's 3D extent/position as default.

    Camera projection only provides an offset and bounded apparent-size
    adjustment. It cannot independently recover arbitrary object size.
    """
    scene=GaussianPly.load(old_scene_ply)
    ids=np.unique(np.asarray(source_gaussian_ids,np.int64).reshape(-1))
    if len(ids)<minimum_points or ids[0]<0 or ids[-1]>=len(scene.vertex):
        raise ValueError("Need valid original Gaussian object IDs")
    mask=np.asarray(new_object_mask,bool)
    if mask.ndim!=2 or not mask.any():
        raise ValueError("New object segmentation is empty")
    if not (0.25<=scale_multiplier<=4):
        raise ValueError("scale_multiplier must be 0.25..4")
    old=scene.means[ids].astype(np.float64)
    bounds=np.percentile(old,[1,99],axis=0)
    target_extent=np.maximum(bounds[1]-bounds[0],1e-6)
    center=np.median(old,axis=0)
    uv,view_z=project_to_pixels(old,camera.viewmat,camera.intrinsic)
    good=np.isfinite(uv).all(axis=1)&(view_z>0)
    if int(good.sum())<minimum_points:
        raise RuntimeError("Old object not visible in chosen hero camera")
    old_screen=np.percentile(uv[good],[2,98],axis=0)
    height=max(float(old_screen[1,1]-old_screen[0,1]),1e-6)
    yy,xx=np.nonzero(mask)
    new_height=float(yy.max()-yy.min()+1)
    apparent_ratio=float(np.clip(new_height/height,0.6,1.6))
    scale=float(scale_multiplier)*apparent_ratio
    projected_center=np.median(uv[good],axis=0)
    k=np.asarray(camera.intrinsic,np.float64)
    z=float(np.median(view_z[good]))
    camera_delta=np.array([
        (float(xx.mean())-projected_center[0])*z/k[0,0],
        (float(yy.mean())-projected_center[1])*z/k[1,1],
        0.0
    ])
    new_center=center+camera_delta@np.linalg.inv(camera.viewmat)[:3,:3].T
    return {
        "method":"old_object_bbox_and_camera_projection",
        "source_object_gaussians":int(len(ids)),
        "old_world_center":center.tolist(),
        "new_world_center":new_center.tolist(),
        "old_world_extent":target_extent.tolist(),
        "target_uniform_extent":float(np.max(target_extent)*scale),
        "apparent_size_ratio":apparent_ratio,
        "scale_multiplier":float(scale_multiplier),
        "estimated_object_center_depth":z,
        "camera_id":str(camera.view_id),
        "source_geometry_authoritative":True,
        "registration_has_been_applied":False,
    }


def merge_tripo_ply(
    background_ply:Path, generated_ply:Path,
    placement:dict, output_ply:Path,
    *, generated_up_axis:int=1,
) -> dict:
    """Register a *Graphdeco-formatted* generated PLY with a conservative
    uniform transform. Unsupported formats require an explicit converter.
    Does not infer contact physics, yaw or shadows.
    """
    source=GaussianPly.load(background_ply)
    obj=GaussianPly.load(generated_ply)
    if not (0<=generated_up_axis<=2):
        raise ValueError("generated_up_axis must be 0,1,2")
    means=obj.means.astype(np.float64)
    center=np.median(means,axis=0)
    ref_ext=np.ptp(means,axis=0)
    if not np.isfinite(ref_ext).all() or np.max(ref_ext)<=1e-9:
        raise ValueError("Generated object has no valid scale")
    target=float(placement["target_uniform_extent"])
    anchor=np.asarray(placement["new_world_center"],np.float64)
    if not (np.isfinite(anchor).all() and anchor.shape==(3,) and 0<target<1e8):
        raise ValueError("Invalid placement prior")
    scale=target/float(np.max(ref_ext))
    if not (1e-5<scale<1e5):
        raise ValueError("Generated-to-scene scaling out of bounds")
    # Leave axis-alignment explicit for now: TripoSplat's native orientation
    # needs a verified mapping to source cameras before auto-yaw/grounding.
    born=np.zeros(len(obj.vertex),dtype=source.vertex.dtype)
    for name in born.dtype.names:
        if name in obj.vertex.dtype.names:
            born[name]=obj.vertex[name]
    transformed=(means-center)*scale+anchor
    for i,name in enumerate(("x","y","z")):
        born[name]=transformed[:,i]
    for name in ("scale_0","scale_1","scale_2"):
        born[name]=obj.vertex[name]+np.log(scale)
    # All shared SH fields transfer; missing high-order terms are zero.
    combined=np.concatenate([source.vertex,born])
    write_vertices(source,output_ply,combined,
                   f"SpEdE registered {len(born)} separately generated Gaussians")
    return {
        "method":"old_object_bbox_uniform_registration_v1",
        "object_gaussians":len(born),
        "scene_gaussians":len(source.vertex),
        "result_gaussians":len(combined),
        "uniform_scale":scale,
        "new_world_center":anchor.tolist(),
        "scene_ply":str(Path(output_ply).resolve()),
        "automatic_axis_yaw_alignment":False,
        "contact_shadow_applied":False,
    }
