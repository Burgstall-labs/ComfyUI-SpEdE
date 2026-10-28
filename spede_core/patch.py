"""Conservative 3D plane-patch births from Qwen's edited camera image.

The image model supplies RGB only. Geometry is derived from the *remaining
source Gaussians* around the object's silhouette, never from a fabricated
depth map. If a stable plane or unoccluded geometry cannot be validated, do
not create a patch. This is a limited floor/wall support-surface method, NOT
arbitrary 3D reconstruction.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, replace
import numpy as np
from scipy import ndimage
from .ply import GaussianPly, append_planar_vertices
from .geometry_utils import project_to_pixels


@dataclass(frozen=True)
class PlaneEvidence:
    normal: np.ndarray
    offset: float
    inlier_count: int
    support_count: int
    inlier_fraction: float
    rmse: float
    center: np.ndarray
    ring: str = "full"           # which part of the silhouette ring it was fitted on
    tolerance: float = 0.0       # RANSAC distance tolerance; 0 when unknown


def _quaternions_from_frames(t1: np.ndarray, t2: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Rotations [w,x,y,z] whose local X,Y,Z map to t1,t2,normal (Nx3 each)."""
    R=np.stack([t1,t2,normal],axis=2)
    m00,m11,m22=R[:,0,0],R[:,1,1],R[:,2,2]
    w=np.sqrt(np.maximum(0,1+m00+m11+m22))/2
    x=np.sqrt(np.maximum(0,1+m00-m11-m22))/2
    y=np.sqrt(np.maximum(0,1-m00+m11-m22))/2
    z=np.sqrt(np.maximum(0,1-m00-m11+m22))/2
    x=np.copysign(x,R[:,2,1]-R[:,1,2])
    y=np.copysign(y,R[:,0,2]-R[:,2,0])
    z=np.copysign(z,R[:,1,0]-R[:,0,1])
    q=np.stack([w,x,y,z],axis=1)
    return (q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)


def _quaternion_from_normal(normal: np.ndarray) -> np.ndarray:
    """Rotation [w,x,y,z] whose local +Z maps to a fitted plane normal."""
    normal = np.asarray(normal, np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < -1 + 1e-8:
        return np.array([0, 1, 0, 0], np.float32)
    cross = np.cross(np.array([0, 0, 1], np.float64), normal)
    quat = np.r_[1 + normal[2], cross]
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat.astype(np.float32)


def fit_support_plane(points: np.ndarray, *,
                      distance_tolerance: float,
                      minimum_inlier_fraction: float = 0.65,
                      minimum_inliers: int = 48,
                      seed: int = 17,
                      attempts: int = 96) -> PlaneEvidence:
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Support cloud must be Nx3")
    if not np.isfinite(points).all() or len(points) < minimum_inliers:
        raise RuntimeError("Insufficient valid support Gaussians")
    if distance_tolerance <= 0:
        raise ValueError("Distance tolerance must be positive")
    rng = np.random.default_rng(seed)
    best = np.empty(0,np.int64)
    best_error = np.inf
    # Limit support samples for predictable CPU runtime.
    if len(points) > 12000:
        points = points[rng.choice(len(points),12000,replace=False)]
    for _ in range(int(attempts)):
        picked = points[rng.choice(len(points),3,replace=False)]
        n = np.cross(picked[1]-picked[0],picked[2]-picked[0])
        length=float(np.linalg.norm(n))
        if length < 1e-10:
            continue
        n /= length
        residual=np.abs(points@n - picked[0]@n)
        idx=np.flatnonzero(residual < distance_tolerance)
        error=float(np.mean(residual[idx])) if len(idx) else np.inf
        if len(idx)>len(best) or (len(idx)==len(best) and error<best_error):
            best=idx
            best_error=error
    if len(best) < minimum_inliers or len(best)/len(points) < minimum_inlier_fraction:
        raise RuntimeError(
            f"No trustworthy support plane: {len(best)}/{len(points)} "
            "candidates satisfy plane RANSAC"
        )
    subset=points[best]
    center=np.median(subset,axis=0)
    centered=subset-center
    _u,singular,vh=np.linalg.svd(centered,full_matrices=False)
    normal=vh[-1]
    normal /= np.linalg.norm(normal)
    residual=np.abs((subset-center)@normal)
    rmse=float(np.sqrt(np.mean(residual**2)))
    if rmse > distance_tolerance:
        raise RuntimeError("Support-plane residual too large")
    return PlaneEvidence(
        normal=normal.astype(np.float64),
        offset=float(center@normal),
        inlier_count=int(len(best)),
        support_count=int(len(points)),
        inlier_fraction=float(len(best)/len(points)),
        rmse=rmse,
        center=center.astype(np.float64),
    )


def _support_samples(
    scene: GaussianPly, mask: np.ndarray,
    viewmat: np.ndarray, intrinsic: np.ndarray,
    *, ring_px: int, minimum_opacity: float = 0.55, ring: str = "full",
    base_fraction: float = 0.3, front_tile_px: int = 4, front_margin: float = 0.1,
):
    if ring_px < 3:
        raise ValueError("Support ring must be >= 3 px")
    # Select world-space kept scene Gaussians whose projections occur just
    # beyond the object's silhouette. Unlike selecting an arbitrary depth map,
    # these are actual real reconstructed Gaussian positions.
    outside=ndimage.binary_dilation(mask,iterations=ring_px)&~ndimage.binary_dilation(mask,iterations=2)
    if ring=="base":
        # Only around the object's lowest part (image down): a tall object's
        # ring otherwise runs over the wall and whatever stands behind it.
        rows=np.flatnonzero(mask.any(axis=1))
        top=rows.max()-base_fraction*(rows.max()-rows.min()+1)
        outside[:int(np.floor(top))]=False
    elif ring!="full":
        raise ValueError("ring must be 'full' or 'base'")
    uv, z=project_to_pixels(scene.means, viewmat, intrinsic)
    x=np.floor(uv[:,0]).astype(np.int64)
    y=np.floor(uv[:,1]).astype(np.int64)
    h,w=mask.shape
    valid=np.isfinite(uv).all(axis=1)&(z>0)&(x>=0)&(y>=0)&(x<w)&(y<h)
    xs=x[valid]
    ys=y[valid]
    src_ids=np.flatnonzero(valid)
    # Use opacity per source splat, never just geometric existence.
    alpha=1/(1+np.exp(-np.clip(scene.opacity_logits[:,0],-30,30)))
    keep=outside[ys,xs]&(alpha[src_ids]>=minimum_opacity)
    selected=src_ids[keep]
    if front_tile_px>0 and len(selected):
        # Only the surface the camera sees in the ring: Gaussians that
        # project there from behind it (the floor and wall behind a table
        # seen through its top) are not what the object stands on.
        depth=z[valid][keep]
        tile=(ys[keep]//front_tile_px)*(w//front_tile_px+1)+xs[keep]//front_tile_px
        _,group=np.unique(tile,return_inverse=True)
        near=np.full(group.max()+1,np.inf)
        np.minimum.at(near,group,depth)
        selected=selected[depth<=near[group]*(1+front_margin)]
    return scene.means[selected].astype(np.float64)


def _unproject_to_plane(y: np.ndarray, x: np.ndarray,
                        viewmat: np.ndarray, intrinsic: np.ndarray,
                        plane: PlaneEvidence) -> tuple[np.ndarray,np.ndarray]:
    K=np.asarray(intrinsic,np.float64)
    if abs(float(K[0,0]))<1e-9 or abs(float(K[1,1]))<1e-9:
        raise ValueError("Invalid camera focal length")
    ray_camera=np.column_stack([
        (x+0.5-K[0,2])/K[0,0],
        (y+0.5-K[1,2])/K[1,1],
        np.ones(len(x)),
    ])
    camera_to_world=np.linalg.inv(np.asarray(viewmat,np.float64))
    origin=camera_to_world[:3,3]
    directions=ray_camera@camera_to_world[:3,:3].T
    denominator=directions@plane.normal
    valid=np.abs(denominator)>1e-7
    t=np.zeros(len(x),np.float64)
    t[valid]=(plane.offset-float(origin@plane.normal))/denominator[valid]
    valid &= (t>0)
    return origin[None,:]+t[:,None]*directions, valid


def fit_view_support_plane(
    scene: GaussianPly, camera_view, object_mask: np.ndarray,
    *, ring_px: int = 24, min_support_inlier_fraction: float = 0.65,
    minimum_births: int = 12,
) -> tuple[PlaneEvidence, float]:
    """Support plane from real Gaussians just outside the hero silhouette,
    and the distance tolerance it was fitted with."""
    mask=np.asarray(object_mask,bool)
    failures=[]
    # The ring around the object's base first: that is where it stands, and
    # a tall object's full ring also crosses the wall or whatever stands
    # behind it (which can be a plane too). The full ring when the base
    # alone has too little support, e.g. a flat object seen from above.
    for ring in ("base","full"):
        support=_support_samples(scene,mask,camera_view.viewmat,camera_view.intrinsic,ring_px=ring_px,ring=ring)
        if len(support)<minimum_births*4:
            failures.append(f"{ring} ring: too few surrounding scene Gaussians")
            continue
        lo,hi=np.percentile(support,[2,98],axis=0)
        diag=max(float(np.linalg.norm(hi-lo)),1e-5)
        tolerance=max(0.008*diag,1e-5)
        try:
            plane=fit_support_plane(
                support,distance_tolerance=tolerance,
                minimum_inlier_fraction=min_support_inlier_fraction,
                minimum_inliers=max(48,minimum_births*4),
            )
        except RuntimeError as error:
            failures.append(f"{ring} ring: {str(error).strip().removeprefix('No trustworthy support plane: ')}")
            continue
        return replace(plane,ring=ring,tolerance=tolerance),tolerance
    if all("too few" in f for f in failures):
        raise RuntimeError("Too few surrounding scene Gaussians for a 3D support plane")
    raise RuntimeError("No trustworthy support plane: "+"; ".join(failures))


def unobserved_contact_ids(
    scene: GaussianPly, object_means: np.ndarray, plane: PlaneEvidence,
    camera_view, object_mask: np.ndarray, tolerance: float,
    *, radius_fraction: float = 0.03, below_tolerances: float = 2.0,
    mask_dilate_px: int = 2,
) -> np.ndarray:
    """Remaining Gaussians in the removed object's contact layer.

    The support surface directly under an object was never seen by any
    capture camera, so the Gaussians a reconstruction left there (and the
    low pieces the selection's footprint column starts above) have arbitrary
    dark colours, yet they are opaque and sit on the plane. They are
    selected only when all of these hold: their centre projects inside the
    (slightly dilated) hero mask, it lies within ``radius_fraction`` x the
    object's footprint of at least three object Gaussians measured in the
    plane, and its height above the plane on the camera side is between
    ``-below_tolerances`` x the plane tolerance and the object's top. The
    visible support around the object and anything under the support stay.
    """
    obj=np.asarray(object_means,np.float64)
    if obj.ndim!=2 or obj.shape[1]!=3 or len(obj)<3:
        raise ValueError("Object means must be Nx3 with at least 3 rows")
    from scipy.spatial import cKDTree
    normal=np.asarray(plane.normal,np.float64)
    camera_center=np.linalg.inv(np.asarray(camera_view.viewmat,np.float64))[:3,3]
    side=1.0 if float((camera_center-plane.center)@normal)>=0 else -1.0
    up=normal*side
    reference=np.eye(3)[int(np.argmin(np.abs(up)))]
    axis_a=np.cross(up,reference); axis_a/=np.linalg.norm(axis_a)
    axis_b=np.cross(up,axis_a)
    def frame(points):
        rel=np.asarray(points,np.float64)-plane.center
        return rel@up, np.stack([rel@axis_a,rel@axis_b],axis=1)
    obj_height,obj_flat=frame(obj)
    top=float(np.percentile(obj_height,99))
    footprint=float(np.linalg.norm(np.percentile(obj_flat,99,axis=0)-np.percentile(obj_flat,1,axis=0)))
    tree=cKDTree(obj_flat)
    # Sparse objects: never below twice the object's own in-plane spacing.
    spacing=float(np.median(tree.query(obj_flat,k=4)[0][:,-1]))
    radius=max(float(radius_fraction)*footprint,2.0*spacing,tolerance)
    means=np.asarray(scene.means,np.float64)
    height,flat=frame(means)
    band=(height>=-float(below_tolerances)*tolerance)&(height<=max(top,0.0)+tolerance)
    mask=np.asarray(object_mask,bool)
    if mask_dilate_px>0:
        mask=ndimage.binary_dilation(mask,iterations=int(mask_dilate_px))
    uv,z=project_to_pixels(means,camera_view.viewmat,camera_view.intrinsic)
    h,w=mask.shape
    finite=np.isfinite(uv).all(axis=1)&(z>0)
    x=np.full(len(means),-1,np.int64); y=np.full(len(means),-1,np.int64)
    x[finite]=np.floor(uv[finite,0]).astype(np.int64)
    y[finite]=np.floor(uv[finite,1]).astype(np.int64)
    inside=finite&(x>=0)&(y>=0)&(x<w)&(y<h)
    inside[inside]=mask[y[inside],x[inside]]
    candidates=np.flatnonzero(band&inside)
    if not len(candidates):
        return candidates
    distance,_=tree.query(flat[candidates],k=3)
    return np.sort(candidates[distance[:,-1]<=radius])


def patch_support_plane(
    removed_scene_ply: Path, camera_view, object_mask: np.ndarray,
    edited_rgb: np.ndarray, output_ply: Path,
    *,
    remaining_alpha: np.ndarray | None = None,
    remaining_depth: np.ndarray | None = None,
    min_support_inlier_fraction: float = 0.65,
    ring_px: int = 24,
    pixel_stride: int = 5,
    cover_alpha_below: float = 0.85,
    minimum_births: int = 12,
    max_births: int = 40000,
    plane: PlaneEvidence | None = None,
    behind_plane_relative: float = 0.03,
    birth_dilate_px: int = 4,
) -> dict:
    source=GaussianPly.load(Path(removed_scene_ply))
    mask=np.asarray(object_mask,bool)
    image=np.asarray(edited_rgb,np.float32)
    h,w=mask.shape
    if (h,w)!=(int(camera_view.height),int(camera_view.width)):
        raise ValueError("Hero mask shape differs from camera")
    if image.shape!=(h,w,3):
        raise ValueError("Qwen edit image must match hero camera H,W,3")
    if not (np.isfinite(image).all() and float(image.min())>=0):
        raise ValueError("Edited image contains invalid pixels")
    if float(image.max(initial=0))>1+1e-6:
        if float(image.max())>255:
            raise ValueError("Edited image out of bounds")
        image=image/255.0
    if not mask.any():
        raise ValueError("Cannot patch empty mask")
    if birth_dilate_px<0:
        raise ValueError("birth_dilate_px must be >= 0")
    if not (2<=pixel_stride<=64):
        raise ValueError("pixel_stride must be 2..64")
    if not (0<=cover_alpha_below<=1):
        raise ValueError("cover_alpha_below must be in [0,1]")
    if plane is None:
        plane,_tolerance=fit_view_support_plane(
            source,camera_view,mask,ring_px=ring_px,
            min_support_inlier_fraction=min_support_inlier_fraction,
            minimum_births=minimum_births,
        )
    # Births stay inside the real support ring around the silhouette. Stray
    # mask pixels whose rays meet the plane far beyond it would otherwise
    # become huge Gaussians floating over the background. Only the ring
    # Gaussians on the plane count: the rest of the ring (the wall or a
    # bicycle beside a tabletop) would stretch the plane out into the air.
    ring=_support_samples(source,mask,camera_view.viewmat,camera_view.intrinsic,ring_px=ring_px,ring=plane.ring)
    on_plane=np.abs(ring@plane.normal-plane.offset)<=(plane.tolerance or 3*plane.rmse)
    if on_plane.sum()>=minimum_births:
        ring=ring[on_plane]
    ring_radius=float(np.percentile(np.linalg.norm(ring-plane.center,axis=1),98))
    # Avoid hallucinated surfaces across steep/edge-on support planes.
    center_to_camera=np.linalg.inv(camera_view.viewmat)[:3,3]-plane.center
    cosine=abs(float(plane.normal@center_to_camera))/max(float(np.linalg.norm(center_to_camera)),1e-12)
    if cosine<0.12:
        raise RuntimeError("Support plane almost edge-on to the hero camera")
    # Smooth sample grid; skip already-opaque genuine background.
    ys,xs=np.mgrid[0:h:pixel_stride,0:w:pixel_stride]
    ys=ys.reshape(-1);xs=xs.reshape(-1)
    # A few pixels past the silhouette: the unseen contact hole is a little
    # wider than the object's outline. Genuine plane content there is
    # opaque at plane depth and is skipped below.
    birth_mask=ndimage.binary_dilation(mask,iterations=int(birth_dilate_px)) if birth_dilate_px>0 else mask
    selected=birth_mask[ys,xs]
    ys=ys[selected];xs=xs[selected]
    xyz,valid=_unproject_to_plane(ys,xs,camera_view.viewmat,camera_view.intrinsic,plane)
    # Sample spacing on the plane itself: at grazing angles one pixel step
    # covers much more surface along depth than across, and radii from the
    # fronto-parallel footprint leave striped gaps.
    step=float(pixel_stride)
    right,valid_r=_unproject_to_plane(ys,xs+step,camera_view.viewmat,camera_view.intrinsic,plane)
    down,valid_d=_unproject_to_plane(ys+step,xs,camera_view.viewmat,camera_view.intrinsic,plane)
    # Anisotropic footprint: one in-plane axis along the image's x step, the
    # other across it, each sized by its own on-plane step length.
    normal=np.asarray(plane.normal,np.float64)
    t1=right-xyz
    t1-=np.outer(t1@normal,normal)
    step_x=np.linalg.norm(t1,axis=1)
    t1/=np.maximum(step_x,1e-12)[:,None]
    t2=np.cross(normal[None,:],t1)
    step_y=np.abs(np.einsum("ij,ij->i",down-xyz,t2))
    spacing=np.stack([step_x,step_y],axis=1)
    valid &= valid_r&valid_d&np.isfinite(spacing).all(axis=1)&(spacing>0).all(axis=1)
    valid &= np.linalg.norm(xyz-plane.center,axis=1)<=ring_radius
    vm=np.asarray(camera_view.viewmat,np.float64)
    patch_z=(xyz@vm[2,:3])+vm[2,3]
    depth=None
    if remaining_depth is not None:
        depth=np.asarray(remaining_depth,np.float32)
        if depth.shape!=(h,w):
            raise ValueError("Scene depth must match mask")
        depth=depth[ys,xs]
    if remaining_alpha is not None:
        alpha=np.asarray(remaining_alpha,np.float32)
        if alpha.shape!=(h,w) or not np.isfinite(alpha).all():
            raise ValueError("Scene alpha must match mask")
        covered=alpha[ys,xs]>=cover_alpha_below
        if depth is not None:
            # Opaque content far behind the plane is not the support surface:
            # the plane is missing there (an unseen hole under the object),
            # so it still needs births.
            covered &= (depth>0)&(depth<=patch_z*(1+float(behind_plane_relative)))
        valid &= ~covered
    if depth is not None:
        # Do not paste a new layer through known opaque foreground geometry.
        # Transparent holes already passed alpha filtering above.
        occluded=(depth>0)&(depth<patch_z*0.99)
        if remaining_alpha is not None:
            occluded &= np.asarray(remaining_alpha)[ys,xs]>0.35
        valid &= ~occluded
    ys=ys[valid];xs=xs[valid];xyz=xyz[valid];spacing=spacing[valid]
    t1=t1[valid];t2=t2[valid]
    if len(xs)>max_births:
        # Fixed spatially spread downsample for bounded memory.
        rows=np.linspace(0,len(xs)-1,max_births).round().astype(int)
        ys=ys[rows];xs=xs[rows];xyz=xyz[rows];spacing=spacing[rows]
        t1=t1[rows];t2=t2[rows]
    if len(xyz)<minimum_births:
        raise RuntimeError("No sufficiently exposed Gaussian births; scene may already contain the background")
    scale=np.maximum((0.75*spacing).astype(np.float32),1e-6)
    quats=_quaternions_from_frames(t1,t2,np.repeat(normal[None,:],len(xyz),axis=0))
    output=append_planar_vertices(
        source.path,output_ply,xyz,image[ys,xs],
        scale,opacity=0.94,thickness_fraction=0.15,quaternions=quats
    )
    output.update({
        "method":"local_supported_plane",
        "geometry_is_planar":True,
        "model_used_for_geometry":False,
        "support_points":plane.support_count,
        "plane_inliers":plane.inlier_count,
        "plane_support_fraction":plane.inlier_fraction,
        "plane_ring":plane.ring,
        "plane_rmse":plane.rmse,
        "plane_normal":plane.normal.tolist(),
        "plane_offset":float(plane.offset),
        "camera_view_id":str(camera_view.view_id),
        "mask_pixel_count":int(mask.sum()),
        "births":int(len(xyz)),
        "g2_plane_evidence_passed":True,
        "g3_image_agreement":None,
        "g3_requires_render":True,
    })
    return output


def patch_image_metrics(target_rgb: np.ndarray, rendered_rgb: np.ndarray,
                        mask: np.ndarray, rendered_alpha: np.ndarray,
                        *, minimum_coverage:float=0.80,
                        maximum_mae:float=0.20) -> dict:
    """G3 photometric agreement is a diagnostic on *actual rendered* pixels."""
    ref=np.asarray(target_rgb,np.float32)
    got=np.asarray(rendered_rgb,np.float32)
    m=np.asarray(mask,bool)
    a=np.asarray(rendered_alpha,np.float32)
    if ref.shape!=got.shape or ref.shape[:2]!=m.shape or m.shape!=a.shape or not m.any():
        raise ValueError("Patch QA images/mask/alpha shape mismatch")
    if ref.max()>1.0: ref=ref/255.0
    if got.max()>1.0: got=got/255.0
    if not (np.isfinite(ref).all() and np.isfinite(got).all() and np.isfinite(a).all()):
        raise ValueError("Patch QA contains nonfinite values")
    coverage=float(np.mean(a[m]>=0.5))
    mae=float(np.mean(np.abs(ref[m]-got[m])))
    return {
        "name":"G3 rendered patch agreement",
        "passed":bool(coverage>=minimum_coverage and mae<=maximum_mae),
        "coverage":coverage,"rgb_mae":mae,
        "minimum_coverage":float(minimum_coverage),
        "maximum_mae":float(maximum_mae),
        "judged_pixels":int(m.sum()),
    }
