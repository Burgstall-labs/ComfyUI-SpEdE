"""AccGS-free edited-RGB -> MoGe depth -> real depth-constrained Gaussian patch tests.

Pure CPU tests. GPU and visual quality remain separate acceptance gates.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from types import SimpleNamespace

from spede_core.depth_fill import (
    SceneAlignedDepth, align_depth_to_scene, create_depth_patch,
)
from spede_core.ply import GaussianPly
try:                                    # this repository
    from competition.nodes import NODE_CLASS_MAPPINGS, LocalCameraFOV
    from competition.local_sdk import build_local_graph, check_local_weights
except ImportError:                     # the exported node pack has these at its root
    from nodes import NODE_CLASS_MAPPINGS, LocalCameraFOV
    from local_sdk import build_local_graph, check_local_weights


def cam(size=96):
    return SimpleNamespace(
        view_id="camera_0",
        width=size,height=size,
        intrinsic=np.array([[90.,0.,size/2],[0.,90.,size/2],[0.,0.,1.]],np.float32),
        viewmat=np.eye(4,dtype=np.float32),
    )


def mask_for(size=96):
    m=np.zeros((size,size),bool)
    m[size//3:size*2//3,size//3:size*2//3]=True
    return m


def sample_rigid_depth(size=96):
    yy,xx=np.indices((size,size))
    return 3 + .0005*(xx-size/2) + .0003*(yy-size/2)


def make_scene(path,rows=200):
    dtype=[(name,"<f4") for name in (
        "x","y","z","f_dc_0","f_dc_1","f_dc_2",
        "opacity","scale_0","scale_1","scale_2",
        "rot_0","rot_1","rot_2","rot_3"
    )]
    dtype.append(("trace_id","<i4"))
    a=np.zeros(rows,dtype=dtype)
    a["x"]=np.linspace(-1,1,rows)
    a["y"]=np.linspace(-.5,.5,rows)
    a["z"]=3
    a["rot_0"]=1
    a["opacity"]=2.8
    for key in ("scale_0","scale_1","scale_2"):
        a[key]=np.log(0.02)
    a["trace_id"]=np.arange(rows)
    PlyData([PlyElement.describe(a,"vertex")],text=False).write(str(path))
    return a


def test_depth_affine_anchored_outside_selection():
    true=sample_rigid_depth()
    object_mask=mask_for()
    model=(true-0.40)/0.75
    model[object_mask]+=0.07  # predicted hole has model uncertainty
    a=align_depth_to_scene(model,true,np.ones_like(true),object_mask,
                            min_anchors=100)
    assert a.report["passed"] is True
    assert a.report["inlier_fraction"]>.9
    np.testing.assert_allclose(a.depth[~object_mask],true[~object_mask],atol=1e-6)
    assert abs(float(np.median(a.depth[object_mask]-true[object_mask])))<0.2
    assert a.report["plane_prior_used"] is False


def test_depth_prior_holds_known_flat_surface_against_moge_bulge():
    true=np.full((96,96),3.0)
    mask=mask_for()
    raw=np.full_like(true,2.5)
    raw[mask]=2.68
    aligned=align_depth_to_scene(
        raw,true,np.ones_like(raw),mask,support_plane_depth=true,
        min_anchors=100
    )
    assert aligned.report["plane_prior_used"] is True
    assert np.max(np.abs(aligned.depth[mask]-3))<0.03
    assert aligned.report["source_scene_authoritative_outside_mask"] is True


def test_depth_refuses_missing_anchors_and_invalid_moge():
    true=np.full((96,96),3.)
    mask=mask_for()
    raw=np.full_like(true,2.)
    with pytest.raises(RuntimeError,match="Too few depth anchors"):
        align_depth_to_scene(raw,true,np.zeros_like(raw),mask)
    with pytest.raises(RuntimeError,match="too many removed pixels invalid"):
        align_depth_to_scene(raw,true,np.ones_like(raw),mask,
                             predicted_valid=~mask)


def test_depth_refuses_wrong_dimensions_and_nonplanar_conflict():
    true=np.full((96,96),3.)
    mask=mask_for()
    with pytest.raises(ValueError,match="shape"):
        align_depth_to_scene(np.ones((16,16)),true,np.ones_like(true),mask)
    raw=np.full_like(true,2.)
    raw[mask]=8.
    with pytest.raises(RuntimeError,match="disagrees with known planar"):
        align_depth_to_scene(raw,true,np.ones_like(true),mask,
                             support_plane_depth=true,
                             max_plane_disagreement_fraction=.35)


def test_depth_patch_creates_actual_3d_xyz_and_preserves_old_vertices(tmp_path):
    c=cam()
    m=mask_for()
    original=tmp_path/"source.ply"
    rows=make_scene(original)
    z=sample_rigid_depth().astype(np.float32)
    region=np.ones_like(m,bool)
    proposal=SceneAlignedDepth(
        depth=z, valid=region,anchor_mask=~m,selection_mask=m,
        report={"passed":True,"anchor_pixels":3000}
    )
    rgb=np.zeros((*m.shape,3),np.float32)
    rgb[:]=[.7,.4,.2]
    alpha=np.ones_like(z)*.98
    alpha[m]=.02
    scene_z=np.full_like(z,3.0)
    output=tmp_path/"patched.ply"
    result=create_depth_patch(
        original,c,rgb,proposal,scene_z,alpha,output,pixel_stride=4
    )
    assert result["born_gaussians"]>=12
    assert result["method"]=="moge3_edited_image_scene_aligned_depth"
    patched=PlyData.read(str(output))["vertex"].data
    np.testing.assert_array_equal(patched[:len(rows)],rows)
    assert 2.9<float(np.median(patched["z"][len(rows):]))<3.1
    assert result["g3_render_required"] is True


def test_depth_patch_never_repaints_already_observed_geometry(tmp_path):
    c=cam()
    m=mask_for()
    src=tmp_path/"scene.ply"
    make_scene(src)
    z=np.full(m.shape,3.,np.float32)
    proposal=SceneAlignedDepth(z,np.ones_like(m),~m,m,{"passed":True})
    with pytest.raises(RuntimeError,match="reliable missing-depth pixels"):
        create_depth_patch(
            src,c,np.ones((*m.shape,3),np.float32),proposal,
            z,np.ones_like(z),tmp_path/"out.ply"
        )


def test_native_moge_node_schema_and_camera_fov():
    assert {"SpEdE_LocalCameraFOV","SpEdE_LocalAlignMoGeDepth",
            "SpEdE_LocalPatchFromDepth"}.issubset(NODE_CLASS_MAPPINGS)
    c=cam()
    pack=SimpleNamespace(cameras=SimpleNamespace(by_id={"camera_0":c}))
    (fov,)=LocalCameraFOV().fov(pack,"camera_0")
    assert fov==pytest.approx(56.14497,abs=.1)
    assert NODE_CLASS_MAPPINGS["SpEdE_LocalAlignMoGeDepth"].INPUT_TYPES()["required"]["moge_geometry"][0]=="MOGE_GEOMETRY"


def test_depthfill_graph_runs_moge_raw_geometry_node(tmp_path):
    scene=tmp_path/"source.ply"
    make_scene(scene)
    npz=tmp_path/"cameras.npz"
    labels=[f"camera_{i}" for i in range(4)]
    np.savez_compressed(npz,targets=np.zeros((4,96,96,3),np.uint8),
                        labels=np.array(labels),viewmats=np.tile(np.eye(4)[None],(4,1,1)),
                        intrinsics=np.tile(cam().intrinsic[None],(4,1,1)),
                        frame_indices=np.arange(4))
    graph=build_local_graph(
        "depthfill",source_ply=scene,camera_targets=npz,
        hero_view="camera_0",object_text="chair",job_name="moge"
    )
    assert graph["61"]["class_type"]=="LoadMoGeModel"
    assert graph["62"]["class_type"]=="MoGeInference"
    assert graph["62"]["inputs"]["image"]==["36",0]
    assert graph["62"]["inputs"]["fov_x_degrees"]==["60",0]
    assert graph["63"]["inputs"]["moge_geometry"]==["62",0]
    assert graph["64"]["inputs"]["aligned_depth"]==["63",0]
    assert graph["41"]["inputs"]["patched_scene"]==["64",0]
    assert graph["45"]["inputs"]["patched_scene"]==["64",0]
    assert graph["65"]["inputs"]["filename_prefix"].startswith("spede/local/moge/")
    assert "Remove the chair from <image1>" in graph["34"]["inputs"]["prompt"]
    assert "segmented foreground object" not in graph["34"]["inputs"]["prompt"]
    with pytest.raises(FileNotFoundError,match="moge"):
        check_local_weights(tmp_path,graph,"depthfill")


def layered_depth(size=96):
    """Four depth layers from a far wall (top) to a tabletop (bottom)."""
    yy,_xx=np.indices((size,size))
    band=np.minimum(yy*4//size,3)
    return np.array([15.0,10.0,6.0,3.5])[band]+0.002*yy


def test_monotone_map_registers_compressed_depth_layers():
    true=layered_depth()
    mask=mask_for()
    # Monocular depth compresses the far layers more (as MoGe 3 did on
    # bonsai): no single scale+shift fits more than two of the layers.
    raw=np.interp(true,[3.5,6.0,10.0,15.0],[1.0,1.5,1.85,2.0])
    a=align_depth_to_scene(raw,true,np.ones_like(true),mask)
    assert a.report["depth_model"]=="monotone"
    tried=a.report["models_tried"]
    assert tried["monotone"]["inlier_fraction"]>=tried["affine"]["inlier_fraction"]+0.05
    assert float(np.median(np.abs(a.depth[mask]-true[mask])/true[mask]))<0.03


def test_monotone_map_never_reverses_depth_order():
    from spede_core.depth_fill import monotone_depth_map, apply_monotone_depth_map
    rng=np.random.default_rng(3)
    x=rng.uniform(1,3,4000); y=4*x+rng.normal(0,1.5,4000)
    kx,ky=monotone_depth_map(x,y)
    assert np.all(np.diff(ky)>=0)
    q=np.linspace(0.5,4,200)
    assert np.all(np.diff(apply_monotone_depth_map(q,kx,ky))>=-1e-12)


def test_plane_prior_is_moved_onto_rendered_surface():
    true=np.full((96,96),3.0)
    mask=mask_for()
    # Plane fitted to Gaussian centres lies 4% behind the rendered surface.
    plane=true*1.04
    a=align_depth_to_scene(true*0.5,true,np.ones_like(true),mask,
                            support_plane_depth=plane)
    assert abs(a.report["plane_visible_surface_correction"]+0.04/1.04)<1e-3
    np.testing.assert_allclose(a.depth[mask],3.0,rtol=2e-3)


def residue_scene(path):
    """Clean tabletop at z=3 plus residue in the hole: a dark layer just
    under the surface, an unseen layer deep under it, a floater in front
    and a far wall Gaussian; each row's role is in trace_id order."""
    a=make_scene(path,rows=6)
    xyz=[(0,0,3.0),(0,0,3.1),(0,0,3.8),(0,0,2.5),(0,0,14.0),(0.6,0,3.0)]
    for i,(x,y,z) in enumerate(xyz):
        a[i]["x"],a[i]["y"],a[i]["z"]=x,y,z
    PlyData([PlyElement.describe(a,"vertex")],text=False).write(str(path))
    return GaussianPly.load(path)


def test_carve_takes_unseen_and_front_residue_and_optionally_visible_layer(tmp_path):
    from spede_core.depth_fill import carve_residue_ids
    scene=residue_scene(tmp_path/"r.ply")
    object_means=np.array([[-.1,-.1,2.4],[.1,.1,3.0],[0,.05,2.7]])
    clean=np.full((96,96),3.0)
    seen=np.array([1,1,0,1,1,1],np.float32)    # row 2 (deep) no camera saw
    # 2 unseen deep layer; 3 floater in front inside the object's box.
    # 0 seen surface, 1 seen sub-surface, 4 far wall, 5 tabletop outside stay.
    ids,report=carve_residue_ids(scene,object_means,cam(),mask_for(),clean,
                                 observed_weight=seen)
    assert ids.tolist()==[2,3]
    assert report["in_front_of_clean_surface"]==1 and report["unseen"]==1
    # The optional visible-layer replacement also takes 0 and 1.
    ids,_=carve_residue_ids(scene,object_means,cam(),mask_for(),clean,
                            observed_weight=seen,visible_layer=0.12)
    assert ids.tolist()==[0,1,2,3]


def test_depth_patch_births_where_remaining_scene_lies_behind_clean_surface(tmp_path):
    src=tmp_path/"s.ply"; make_scene(src)
    true=np.full((96,96),3.0)
    mask=mask_for()
    aligned=align_depth_to_scene(true*0.5,true,np.ones_like(true),mask)
    behind=true.copy(); behind[mask]=3.6          # exposed lower layer
    image=np.full((96,96,3),0.5,np.float32)
    covered=np.ones_like(true)
    with pytest.raises(RuntimeError):
        create_depth_patch(src,cam(),image,aligned,behind,covered,tmp_path/"a.ply",
                           behind_tolerance=None)
    out=create_depth_patch(src,cam(),image,aligned,behind,covered,tmp_path/"b.ply")
    assert out["born_gaussians"]>12


def test_births_behind_clean_surface_only_where_carving_exposed(tmp_path):
    from spede_core.depth_fill import carving_exposed
    src=tmp_path/"s.ply"; make_scene(src)
    true=np.full((96,96),3.0)
    mask=mask_for()
    aligned=align_depth_to_scene(true*0.5,true,np.ones_like(true),mask)
    after=true.copy(); after[mask]=3.6
    before=true.copy()
    before[mask]=3.6; before[30:48,30:64]=3.0  # carving moved only this band
    covered=np.ones_like(true)
    exposed=carving_exposed(covered,before,covered,after)
    assert exposed[mask].sum()==(mask&(np.indices(mask.shape)[0]<48)).sum()
    image=np.full((96,96,3),0.5,np.float32)
    out=create_depth_patch(src,cam(),image,aligned,after,covered,tmp_path/"b.ply",exposed=exposed)
    full=create_depth_patch(src,cam(),image,aligned,after,covered,tmp_path/"c.ply")
    assert 12<out["reliable_birth_pixels"]<full["reliable_birth_pixels"]


def test_shadow_halo_takes_brightened_border_touching_selection():
    from spede_core.depth_fill import shadow_halo
    mask=mask_for()                               # rows/cols 32..63
    clean=np.full((96,96,3),0.6,np.float32)
    render=clean.copy()
    render[64:72,32:64]=0.2                       # shadow under the object
    render[5:12,5:12]=0.2                         # dark spot far away
    render[72:80,32:64]=0.9                       # brighter than clean: not a shadow
    halo=shadow_halo(mask,render,clean,halo_px=24)
    assert halo[66:70,36:60].all()
    assert not halo[5:12,5:12].any() and not halo[74:80,36:60].any() and not halo[mask].any()
    assert not shadow_halo(mask,render,clean,halo_px=0).any()


def test_halo_births_cover_shadow_outside_selection(tmp_path):
    from spede_core.depth_fill import shadow_halo
    src=tmp_path/"s.ply"; make_scene(src)
    true=np.full((96,96),3.0)
    mask=mask_for()
    aligned=align_depth_to_scene(true*0.5,true,np.ones_like(true),mask)
    behind=true.copy(); behind[mask]=3.6
    image=np.full((96,96,3),0.5,np.float32)
    render=image.copy(); render[64:72,32:64]=0.1
    halo=shadow_halo(mask,render,image)
    plain=create_depth_patch(src,cam(),image,aligned,behind,np.ones_like(true),tmp_path/"a.ply",pixel_stride=1)
    out=create_depth_patch(src,cam(),image,aligned,behind,np.ones_like(true),tmp_path/"b.ply",pixel_stride=1,
                           halo=halo,halo_depth=true)
    assert out["shadow_halo_pixels"]==halo.sum()>0
    assert out["born_gaussians"]>=plain["born_gaussians"]+0.8*halo.sum()


def test_births_on_grazing_surface_are_stretched_only_across(tmp_path):
    src=tmp_path/"s.ply"; make_scene(src)
    yy,_xx=np.indices((96,96))
    table=1.0/(0.05+0.006*yy)                  # a tabletop receding towards the top
    mask=mask_for()
    aligned=align_depth_to_scene(table*0.5,table,np.ones_like(table),mask)
    behind=table*1.3
    image=np.full((96,96,3),0.5,np.float32)
    create_depth_patch(src,cam(),image,aligned,behind,np.ones_like(table),tmp_path/"b.ply",pixel_stride=1)
    v=GaussianPly.load(tmp_path/"b.ply").vertex
    born=v[len(GaussianPly.load(src).vertex):]
    across=np.exp(born["scale_1"])/np.exp(born["scale_0"])
    assert np.median(across)>1.5             # round discs (the old writer) give 1


def test_edit_spill_finds_what_qwen_removed_beyond_the_selection():
    from spede_core.depth_fill import edit_spill
    mask=mask_for()                               # rows/cols 32..63
    render=np.full((96,96,3),0.5,np.float32)
    clean=render.copy()
    clean[64:90,30:66]=0.9                        # the pot under the selected tree, gone
    clean[2:8,2:8]=0.9                            # unrelated change far away
    clean[mask]=0.1                               # inside the selection: anything goes
    spill=edit_spill(mask,render,clean,halo_px=4)
    assert spill[75:88,35:60].all()
    assert not spill[2:8,2:8].any() and not spill[mask].any()
    assert not edit_spill(mask,render,np.where(mask[...,None],0.1,render)).any()


def test_match_edit_tone_undoes_a_colour_shift_measured_outside_the_selection():
    from spede_core.depth_fill import match_edit_tone

    rng = np.random.default_rng(1)
    scene = np.clip(rng.normal([0.5, 0.4, 0.3], 0.08, (128, 128, 3)), 0, 1)
    selection = np.zeros((128, 128), bool)
    selection[40:90, 40:90] = True
    scene[selection] = [0.9, 0.1, 0.1]                  # the object: must not steer the match
    lift = np.array([[1.15, 0.05, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.05]])
    edited = np.clip(scene @ lift.T + 0.03, 0, 1)
    edited[selection] = [0.5, 0.4, 0.3]                 # the cleaned surface
    for method in ("gain", "mkl"):
        out, report = match_edit_tone(edited, scene, selection, margin_px=4, method=method)
        far = ~selection
        assert report["method"] == method
        assert np.abs(out[far].mean(0) - scene[far].mean(0)).max() < 0.01
    assert np.abs(out[far] - scene[far]).mean() < np.abs(edited[far] - scene[far]).mean() / 3


def test_seam_feather_eases_kept_colours_near_births_only():
    from spede_core.depth_fill import feather_seam_colours
    from scipy import ndimage
    from spede_core.ply import SH_C0
    size=96; c=cam(size); z=2.0
    ys,xs=np.mgrid[0:size,0:size]
    pix=np.c_[xs.ravel()+0.5,ys.ravel()+0.5]
    plane=np.c_[(pix-size/2)/90.*z,np.full(len(pix),z)]
    births=mask_for(size)
    # Kept surface everywhere, a hidden copy behind it, births over the hole.
    means=np.r_[plane,plane*1.5,plane[births.ravel()]]
    n_kept=len(plane)
    born=np.r_[np.zeros(2*n_kept,bool),np.ones(int(births.sum()),bool)]
    rgb=np.r_[np.full((2*n_kept,3),0.4),np.full((int(births.sum()),3),0.8)]
    dc=(rgb-0.5)/SH_C0
    weight=np.r_[np.full(n_kept,0.9),np.zeros(n_kept),np.full(int(births.sum()),0.9)]
    target=np.full((size,size,3),0.8,np.float32)
    rendered=np.repeat(np.where(births,0.8,0.4)[...,None],3,axis=-1).astype(np.float32)
    out,report=feather_seam_colours(
        means,dc,born,c,np.zeros_like(births),target,rendered,weight,band_px=12)
    new=0.5+SH_C0*out[:,0]
    dist=ndimage.distance_transform_edt(~births).ravel()
    front=new[:n_kept]
    near=(dist>0)&(dist<=2)
    assert report["applied"] and report["recoloured_gaussians"]>0
    assert (front[near]>0.65).all()                  # next to the births: close to the target
    np.testing.assert_allclose(front[dist>=13],0.4,atol=1e-6)   # beyond the band: unchanged
    mid=(dist>5)&(dist<7)
    assert (0.45<front[mid]).all() and (front[mid]<front[near].min()).all()   # ramps down
    np.testing.assert_allclose(out[n_kept:],dc[n_kept:])                    # hidden and births kept
    off,_r=feather_seam_colours(means,dc,born,c,np.zeros_like(births),target,rendered,weight,band_px=0)
    np.testing.assert_allclose(off,dc)
    # A kept surface inside the selection takes the cleaned image's tone fully.
    sel_out,sel_report=feather_seam_colours(
        plane,dc[:n_kept],np.zeros(n_kept,bool),c,births,target,
        np.full((size,size,3),0.4,np.float32),np.full(n_kept,0.9),band_px=12)
    inside=0.5+SH_C0*sel_out[births.ravel(),0]
    np.testing.assert_allclose(inside,0.8,atol=1e-5)
    assert sel_report["recoloured_inside_selection"]==int(births.sum())
