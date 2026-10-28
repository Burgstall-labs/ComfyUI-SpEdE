"""CPU-only tests for AccGS-free local SpEdE competition path."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from spede_core.ply import GaussianPly, remove_rows, append_planar_vertices, SH_C0
from spede_core.patch import (
    fit_support_plane, patch_support_plane, patch_image_metrics,
    _quaternion_from_normal,
)
from spede_core.visual_hull import HullView, visual_hull_object_ids
try:                                    # this repository
    from competition.nodes import NODE_CLASS_MAPPINGS
    from competition.local_sdk import build_local_graph, validate_local_endpoint
except ImportError:                     # the exported node pack has these at its root
    from nodes import NODE_CLASS_MAPPINGS
    from local_sdk import build_local_graph, validate_local_endpoint


def make_scene(path: Path, xyz):
    xyz=np.asarray(xyz,dtype=np.float32)
    props=[
        "x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity",
        "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"
    ]
    rows=np.zeros(len(xyz),dtype=np.dtype([(name,"<f4") for name in props]+[("custom_id","<i4")]))
    for i,key in enumerate(("x","y","z")):
        rows[key]=xyz[:,i]
    rows["opacity"]=4.2
    for key in ("scale_0","scale_1","scale_2"):
        rows[key]=np.log(0.03)
    rows["rot_0"]=1
    rows["custom_id"]=np.arange(len(rows))
    PlyData([PlyElement.describe(rows,"vertex")],text=False).write(str(path))
    return rows


def camera():
    return SimpleNamespace(
        view_id="camera_0",
        viewmat=np.eye(4,dtype=np.float32),
        intrinsic=np.array([[60,0,32],[0,60,32],[0,0,1]],np.float32),
        width=64,height=64
    )


def plane_fixture(tmp_path):
    cam=camera()
    y,x=np.mgrid[0:64,0:64]
    mask=np.zeros((64,64),bool)
    mask[20:44,20:44]=True
    # Source scene has only real points outside the hole.
    keep=~mask
    xyz=np.column_stack([
        (x[keep]+0.5-32)/60*3,
        (y[keep]+0.5-32)/60*3,
        np.full(int(keep.sum()),3.0)
    ])
    path=tmp_path/"source.ply"
    rows=make_scene(path,xyz)
    image=np.zeros((64,64,3),np.float32)
    image[:]=np.array([.4,.6,.3])
    image[mask]=np.array([.8,.25,.17])
    return path,cam,mask,image,rows


def test_standalone_ply_can_preserve_custom_source_properties(tmp_path):
    source=tmp_path/"source.ply"
    rows=make_scene(source,[[0,0,3],[1,0,3],[2,0,3]])
    ply=GaussianPly.load(source)
    assert ply.max_sh_degree==0
    assert len(ply.vertex)==3
    result=remove_rows(source,np.array([1],np.int64),tmp_path/"removed.ply")
    assert result["remaining_gaussians"]==2
    np.testing.assert_array_equal(
        PlyData.read(tmp_path/"removed.ply")["vertex"].data,
        rows[[0,2]]
    )
    with pytest.raises(ValueError,match="overwrit"):
        remove_rows(source,np.array([0]),source)
    with pytest.raises(ValueError,match="all Gaussians"):
        remove_rows(source,np.array([0,1,2]),tmp_path/"empty.ply")


def test_gaussian_births_keep_original_order_dtype_and_SH(tmp_path):
    path=tmp_path/"source.ply"
    rows=make_scene(path,[[0,0,3],[1,0,3]])
    result=append_planar_vertices(
        path,tmp_path/"out.ply",
        np.array([[0.5,0.4,3.0]],np.float32),
        np.array([[0.8,0.1,0.2]],np.float32),
        np.array([0.05],np.float32),
        quaternions=np.array([[1,0,0,0]],np.float32)
    )
    new=PlyData.read(tmp_path/"out.ply")["vertex"].data
    np.testing.assert_array_equal(new[:2],rows)
    assert result["added_gaussians"]==1
    assert new["custom_id"][2]==0
    assert float(new["f_dc_0"][2])==pytest.approx((.8-.5)/SH_C0,abs=1e-5)
    assert new["scale_2"][2]<new["scale_0"][2]


def test_ransac_resists_outliers_and_rejects_unjustified_plane():
    rng=np.random.default_rng(3)
    support=np.column_stack([rng.normal(size=(150,2)),np.full(150,3.0)])
    support+=rng.normal(0,0.001,support.shape)
    wrong=rng.normal(size=(25,3))*5
    evidence=fit_support_plane(np.concatenate([support,wrong]),
                                distance_tolerance=0.02)
    assert evidence.inlier_fraction>0.7
    assert evidence.rmse<0.02
    with pytest.raises(RuntimeError,match="Insufficient"):
        fit_support_plane(rng.normal(size=(10,3)),distance_tolerance=0.01)
    with pytest.raises(RuntimeError,match="No trustworthy"):
        fit_support_plane(rng.normal(size=(100,3)),distance_tolerance=0.00001)


def test_quaternion_maps_local_z_to_plane_normal():
    for normal in ([0,0,1],[0,0,-1],[0,1,0]):
        q=_quaternion_from_normal(np.array(normal,float))
        w,x,y,z=q
        rot=np.array([
            [1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
            [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]
        ])
        np.testing.assert_allclose(rot@np.array([0,0,1]),normal,atol=1e-5)


def test_planar_patch_grows_real_3d_gaussians_and_preserves_source(tmp_path):
    src,cam,mask,image,rows=plane_fixture(tmp_path)
    out=tmp_path/"patched.ply"
    result=patch_support_plane(
        src,cam,mask,image,out,
        remaining_alpha=np.zeros(mask.shape,np.float32),
        remaining_depth=np.zeros(mask.shape,np.float32),
        ring_px=12,pixel_stride=4
    )
    assert result["method"]=="local_supported_plane"
    assert result["geometry_is_planar"] is True
    assert result["g2_plane_evidence_passed"] is True
    assert result["g3_image_agreement"] is None
    assert result["added_gaussians"]>=12
    patched=PlyData.read(out)["vertex"].data
    np.testing.assert_array_equal(patched[:len(rows)],rows)
    assert np.median(patched["z"][len(rows):])==pytest.approx(3.0,abs=0.025)
    assert out.is_file()
    assert result["source_gaussians"]==len(rows)


def test_planar_patch_refuses_missing_support_and_wrong_shape(tmp_path):
    src,cam,mask,image,_=plane_fixture(tmp_path)
    with pytest.raises(ValueError,match="match hero"):
        patch_support_plane(src,cam,mask,image[:50],tmp_path/"bad.ply")
    empty=tmp_path/"few.ply"
    make_scene(empty,np.array([[0,0,3],[1,0,3],[2,0,3]],np.float32))
    with pytest.raises(RuntimeError,match="Too few surrounding"):
        patch_support_plane(empty,cam,mask,image,tmp_path/"bad.ply")


def test_g3_color_qa_is_measured_after_render():
    mask=np.ones((5,5),bool)
    goal=np.ones((5,5,3),np.float32)*0.5
    actual=goal.copy()
    score=patch_image_metrics(goal,actual,mask,np.ones((5,5)))
    assert score["passed"] is True
    score=patch_image_metrics(goal,actual+0.5,mask,np.ones((5,5)))
    assert score["passed"] is False
    score=patch_image_metrics(goal,actual,mask,np.zeros((5,5)))
    assert score["passed"] is False


def test_frozen_3d_visual_hull_runs_on_cpu_without_accgs():
    pts=np.array([[0,0,3],[3,3,3]],np.float32)
    K=np.array([[20,0,16],[0,20,16],[0,0,1]],np.float32)
    mask=np.zeros((32,32),bool)
    mask[12:20,12:20]=True
    view=HullView("v",np.eye(4,dtype=np.float32),K,mask)
    selected=visual_hull_object_ids(
        pts,hero_viewmat=view.viewmat,hero_intrinsic=K,hero_mask=mask,
        views=[view],dilate_px=0,view_tolerance=0,min_seen_fraction=1
    )
    assert selected.object_ids.tolist()==[0]


ROOT=Path(__file__).resolve().parents[1]
# The exported node pack is flat: what the repository keeps in competition/ sits at its root.
PACK=ROOT/"competition" if (ROOT/"competition").is_dir() else ROOT


def test_competition_nodepack_is_accgs_free_and_exports_17_nodes():
    root=ROOT
    for folder in (root/"spede_core",PACK):
        for path in folder.rglob("*.py"):
            body=path.read_text()
            import re
            assert not re.search(r"(?m)^\s*(?:from\s+accgs|import\s+accgs)\b",body),path
            compile(body,str(path),"exec")
    assert len(NODE_CLASS_MAPPINGS)==33
    for key,klass in NODE_CLASS_MAPPINGS.items():
        assert hasattr(klass,klass.FUNCTION),key
        assert hasattr(klass,"INPUT_TYPES"),key


def test_competition_workflow_refs_and_comfy_native_types():
    for name in ("spede_local_sam31_extract_api.json","spede_local_sam31_qwen21_patch_api.json"):
        graph=json.loads((PACK/"workflows"/name).read_text())
        assert graph
        for node_id,node in graph.items():
            assert node["class_type"] in NODE_CLASS_MAPPINGS or node["class_type"] in {
                "CheckpointLoaderSimple","CLIPTextEncode","SAM3_Detect",
                "UNETLoader","CLIPLoader","VAELoader","QwenImage21Cache",
                "TextEncodeQwenImage21","KSampler","VAEDecode","SaveImage",
            }
            for k,v in node["inputs"].items():
                if isinstance(v,list) and len(v)==2:
                    assert str(v[0]) in graph, (name,node_id,k)
        assert graph["22"]["class_type"]=="SAM3_Detect"
        if "patch" in name:
            assert graph["34"]["class_type"]=="TextEncodeQwenImage21"
            assert graph["34"]["inputs"]["images.image_1"] == ["27", 0]
            assert graph["34"]["inputs"]["images.image_2"] == ["26", 0]
            assert graph["35"]["inputs"]["cfg"]==1
            assert graph["40"]["class_type"]=="SpEdE_LocalPatchPlane"
            assert graph["41"]["class_type"]=="SpEdE_LocalPatchQA"


def test_sdk_builds_full_graph_offline_and_refuses_cloud_fallback(tmp_path,monkeypatch):
    try:
        from competition.local_sdk import check_local_weights
    except ImportError:
        from local_sdk import check_local_weights
    src=tmp_path/"scene.ply"
    make_scene(src,[[0,0,3],[1,0,3]])
    packed=tmp_path/"cameras.npz"
    np.savez(packed,
        targets=np.zeros((4,64,64,3),np.uint8),
        labels=np.array(["camera_0","camera_1","camera_2","camera_3"]),
        viewmats=np.repeat(np.eye(4)[None],4,axis=0),
        intrinsics=np.repeat(np.eye(3)[None],4,axis=0),
        frame_indices=np.arange(4),
    )
    graph=build_local_graph("patch",source_ply=src,camera_targets=packed,
                            hero_view="camera_2",object_text="chair",job_name="test")
    assert graph["23"]["inputs"]["hero_view_id"]=="camera_2"
    assert graph["40"]["inputs"]["view_id"]=="camera_2"
    assert all(n["inputs"]["view_id"]=="camera_2" for n in graph.values() if "view_id" in n["inputs"])
    # SAM keeps up to four instances per camera, apart, for the selection
    # to pick the hero's object among them; the name itself stays plain.
    assert graph["21"]["inputs"]["text"]=="chair:4"
    assert graph["22"]["inputs"]["individual_masks"] is True
    assert graph["23"]["inputs"]["boxes"]==["22",1]
    assert graph["23"]["inputs"]["object_text"]=="chair"
    assert graph["24"]["inputs"]["include_footprint"] is True
    assert "hero_mask_path" not in graph["23"]["inputs"]
    hero=tmp_path/"hero.png"; hero.write_bytes(b"png")
    with_mask=build_local_graph("extract",source_ply=src,camera_targets=packed,hero_view="camera_2",
                                object_text="chair",hero_mask=hero)
    assert with_mask["23"]["inputs"]["hero_mask_path"]==str(hero.resolve())
    assert build_local_graph("extract",source_ply=src,camera_targets=packed,hero_view="camera_2",object_text="chair")["24"]["inputs"]["include_footprint"] is False
    assert graph["25"]["inputs"]["output_path"].startswith("spede/local/test/")
    assert graph["42"]["inputs"]["filename_prefix"]=="spede/local/test/qwen21_edit"
    assert graph["43"]["inputs"]["filename_prefix"]=="spede/local/test/patched_render"
    monkeypatch.delenv("COMFY_BASE_URL",raising=False)
    with pytest.raises(ValueError,match="COMFY_BASE_URL"):
        validate_local_endpoint()
    monkeypatch.setenv("COMFY_BASE_URL","https://cloud.comfy.org")
    with pytest.raises(ValueError,match="loopback"):
        validate_local_endpoint()
    monkeypatch.setenv("COMFY_BASE_URL","http://127.0.0.1:8189")
    assert validate_local_endpoint()=="http://127.0.0.1:8189"
    with pytest.raises(FileNotFoundError,match="missing"):
        check_local_weights(tmp_path,graph,"patch")



def test_standalone_bundle_private_dev_and_public_gate(tmp_path):
    exporter=pytest.importorskip("scripts.export_local_competition")   # not shipped in the exported pack
    manifest,build=exporter.manifest,exporter.build
    info=manifest()
    assert info["no_accgs_imports"] is True
    assert all("accgs/" not in x["target"].lower() for x in info["files"])
    assert all("splatedit/" not in x["target"].lower() for x in info["files"])
    with pytest.raises(PermissionError,match="rights-review"):
        build(tmp_path/"public",public_build=True,allow_dirty=True)
    private=tmp_path/"private"
    summary=build(private,private_dev=True)
    assert summary["license_mode"]=="private_dev"
    assert (private/"__init__.py").exists()
    assert (private/"spede_core"/"renderer.py").exists()
    assert (private/"workflows"/"spede_local_sam31_qwen21_patch_api.json").exists()
    assert not (private/"accgs").exists()
    assert not (private/"splatedit").exists()
    assert "polyformproject.org/licenses/noncommercial" in (private/"LICENSE").read_text()
    with pytest.raises(FileExistsError,match="empty"):
        build(private,private_dev=True)
    public=tmp_path/"public"
    summary=build(public,public_build=True,ack_rights=True,allow_dirty=True)
    assert summary["license_mode"]=="public_agpl"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (public/"LICENSE").read_text()
    assert "AGPL-3.0" in (public/"NOTICE").read_text()
    assert "polyform" not in (public/"THIRD_PARTY.md").read_text().lower()
    noncommercial=tmp_path/"pf.txt"
    noncommercial.write_text((private/"LICENSE").read_text())
    with pytest.raises(ValueError,match="AGPL"):
        build(tmp_path/"other",public_build=True,ack_rights=True,license_file=noncommercial,
              allow_dirty=True)



def test_sh_degree_three_and_extra_fields_survive_birth_and_removal(tmp_path):
    src=tmp_path/"with_sh3.ply"
    basis=["x","y","z","f_dc_0","f_dc_1","f_dc_2"]
    basis += [f"f_rest_{i}" for i in range(45)]
    basis += ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    dtype=np.dtype([(name,"<f4") for name in basis]+[("trace_id","<i4")])
    data=np.zeros(5,dtype=dtype)
    data["rot_0"]=1
    data["opacity"]=3
    for i in range(45):
        data[f"f_rest_{i}"]=i/200
    data["trace_id"]=np.arange(5)+1000
    PlyData([PlyElement.describe(data,"vertex")],text=False).write(src)
    scene=GaussianPly.load(src)
    assert scene.max_sh_degree==3
    assert scene.rest.shape==(5,15,3)
    out=tmp_path/"new_sh3.ply"
    append_planar_vertices(
        src,out,np.array([[0.3,0.2,1.5]]),
        np.array([[0.3,0.4,0.8]]),np.array([0.02])
    )
    final=PlyData.read(out)["vertex"].data
    np.testing.assert_array_equal(final[:5],data)
    assert all(final[f"f_rest_{i}"][5]==0 for i in range(45))
    assert final["trace_id"][5]==0
    cut=tmp_path/"cut_sh3.ply"
    remove_rows(out,np.array([2]),cut)
    rows=PlyData.read(cut)["vertex"].data
    np.testing.assert_array_equal(rows[:4],data[[0,1,3,4]])
    assert rows.dtype.names==data.dtype.names



def test_heldout_rgb_locality_metric_ignores_edit_region_only():
    from spede_core.quality import outside_mask_preservation
    original=np.ones((64,64,3),np.float32)*.4
    patched=original.copy()
    mask=np.zeros((64,64),bool)
    mask[20:44,20:44]=True
    patched[mask]=1.0
    score=outside_mask_preservation(original,patched,mask)
    assert score["passed"] is True
    assert score["outside_rgb_mae"]==0
    # Changing only 100/4096 pixels is below the deliberate 7% gate.
    patched[:30,:30]=.0
    score=outside_mask_preservation(original,patched,mask)
    assert score["passed"] is False
    assert score["outside_changed_fraction"]>0



def test_standalone_nodepack_imports_without_accgs_or_gpu(tmp_path):
    import subprocess
    import sys
    exporter=pytest.importorskip("scripts.export_local_competition")

    bundle=tmp_path/"ComfyUI-SpEdE-Local"
    exporter.build(bundle,private_dev=True)
    script=(
        "import importlib.util,sys,pathlib,json\n"
        "root=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location("
        "'spede_local_export',root/'__init__.py',submodule_search_locations=[str(root)])\n"
        "module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module)\n"
        "print(json.dumps({'nodes':list(module.NODE_CLASS_MAPPINGS),"
        "'accgs_loaded':any(k=='accgs' or k.startswith('accgs.') for k in sys.modules)}))\n"
    )
    result=subprocess.run(
        [sys.executable,"-c",script,str(bundle)],
        cwd=tmp_path,text=True,capture_output=True,timeout=30,
        env={"PYTHONPATH":str(bundle)},
        check=True,
    )
    info=json.loads(result.stdout.strip())
    assert "SpEdE_LocalPatchPlane" in info["nodes"]
    assert len(info["nodes"])==33
    assert info["accgs_loaded"] is False


def test_unobserved_contact_layer_is_cleared_only_under_object():
    from types import SimpleNamespace
    from spede_core.patch import PlaneEvidence, unobserved_contact_ids
    rng=np.random.default_rng(3)
    obj=np.column_stack([rng.uniform(-.5,.5,400),rng.uniform(-.5,.5,400),rng.uniform(.05,1,400)])
    contact=np.column_stack([rng.uniform(-.4,.4,50),rng.uniform(-.4,.4,50),np.full(50,.01)])
    visible_table=np.column_stack([rng.uniform(1.5,2,50),rng.uniform(-.4,.4,50),np.zeros(50)])
    under_table=np.column_stack([rng.uniform(-.4,.4,50),rng.uniform(-.4,.4,50),np.full(50,-1.0)])
    scene=SimpleNamespace(means=np.vstack([contact,visible_table,under_table]))
    camera=SimpleNamespace(
        viewmat=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,5],[0,0,0,1]],float),
        intrinsic=np.array([[100,0,50],[0,100,50],[0,0,1]],float),
    )
    mask=np.zeros((100,100),bool); mask[39:62,39:62]=True
    plane=PlaneEvidence(normal=np.array([0,0,-1.0]),offset=0.0,inlier_count=100,
                        support_count=100,inlier_fraction=1.0,rmse=0.0,center=np.zeros(3))
    ids=unobserved_contact_ids(scene,obj,plane,camera,mask,tolerance=0.02)
    assert np.array_equal(ids,np.arange(50))


# ---- hero-view check (clicks vs the object's name) ----------------------------

from spede_core.hero_check import clean_name, compare_hero_masks, focus_crop


def _box(y0,y1,x0,x1,shape=(60,80)):
    m=np.zeros(shape,bool)
    m[y0:y1,x0:x1]=True
    return m


def test_clean_name_strips_model_chatter():
    assert clean_name("The bonsai tree.\n") == "bonsai tree"
    assert clean_name("'Lego Bulldozer'") == "lego bulldozer"
    assert clean_name("a red bicycle with a basket on the back") == "red bicycle with a basket"


def test_hero_check_agrees_when_name_finds_the_clicked_object():
    click=_box(10,40,10,40)
    chosen,named,report=compare_hero_masks(click,[_box(11,40,10,41),_box(45,58,60,78)],[[20,20]])
    assert report["status"]=="agree" and report["passed"]
    assert report["matched_instances"]==[0]
    assert (chosen==click).all()


def test_hero_check_widens_a_holey_click_mask_inside_the_named_object():
    buggy=_box(10,40,10,50)
    click=_box(10,40,10,40)
    click[20:24,20:24]=False                             # the splat render's holes
    chosen,named,report=compare_hero_masks(click,[buggy],[[15,15]])
    assert report["status"]=="agree" and report["chosen"]=="name"
    assert (chosen==buggy).all()
    # An exclude click on the extra part keeps the clicked mask.
    chosen,named,report=compare_hero_masks(click,[buggy],[[15,15]],[[45,25]])
    assert report["chosen"]=="click" and (chosen==click).all()


def test_hero_check_narrows_click_mask_that_grabbed_a_neighbour():
    bonsai=_box(10,30,10,30)
    bicycle=_box(10,50,30,75)
    click=bonsai|bicycle                                 # SAM's click mask took both
    chosen,named,report=compare_hero_masks(click,[bonsai],'[{"x": 20, "y": 20}]')
    assert report["status"]=="click_mask_too_big" and not report["passed"]
    assert report["chosen"]=="name"
    assert (chosen==bonsai).all()


def test_hero_check_flags_a_name_that_finds_something_else():
    chosen,named,report=compare_hero_masks(_box(10,30,10,30),[_box(40,58,50,78)],[[20,20]])
    assert report["status"]=="name_elsewhere" and not report["passed"]
    chosen,named,report=compare_hero_masks(_box(10,30,10,30),np.zeros((0,60,80)),[[20,20]])
    assert report["status"]=="name_not_found" and named is None


def test_hero_check_keeps_several_clicked_instances():
    left,right=_box(10,30,5,25),_box(10,30,50,70)
    chosen,named,report=compare_hero_masks(left|right,[left,right],[[15,20],[60,20]])
    assert report["status"]=="agree" and report["matched_instances"]==[0,1]


def test_hero_check_exclude_click_drops_an_instance():
    left,right=_box(10,30,5,25),_box(10,30,50,70)
    chosen,named,report=compare_hero_masks(left,[left,right],[[15,20,1]],[[60,20]])
    assert report["matched_instances"]==[0] and report["passed"]


def test_focus_crop_dims_outside_and_keeps_aspect():
    img=np.full((60,80,3),200,np.uint8)
    crop=focus_crop(img,_box(20,40,30,50),size=100)
    assert max(crop.shape[:2])==100 and crop.shape[2]==3
    assert crop[0,0].max()<100                            # dimmed context


def test_hero_graph_drops_naming_when_user_names_it(tmp_path):
    try:
        from competition.local_sdk import build_hero_graph, needs_qwen
    except ImportError:
        from local_sdk import build_hero_graph, needs_qwen
    cams=tmp_path/"cams.npz"
    np.savez(cams,labels=np.array(["a","b","c","d"]),viewmats=np.repeat(np.eye(4)[None],4,0),
             intrinsics=np.repeat(np.eye(3)[None],4,0),targets=np.zeros((4,8,8,3),np.uint8),
             frame_indices=np.arange(4))
    graph=build_hero_graph(camera_targets=cams,hero_view="b",points=[[3,4],[5,6,0]],job_name="j1")
    assert graph["4"]["inputs"]["view_id"]=="b"
    assert json.loads(graph["10"]["inputs"]["positive_coords"])==[{"x":3,"y":4}]
    assert json.loads(graph["15"]["inputs"]["negative_coords"])==[{"x":5,"y":6}]
    assert graph["13"]["class_type"]=="TextGenerate" and needs_qwen(graph)
    named=build_hero_graph(camera_targets=cams,hero_view="b",points=[[3,4]],object_text="chair",job_name="j1")
    assert "13" not in named and named["21"]["inputs"]["text"]=="chair" and not needs_qwen(named)
    assert named["15"]["inputs"]["output_subdir"]=="spede/local/j1/hero"


def test_hero_nodes_registered():
    for key in ("SpEdE_LocalViewImage","SpEdE_LocalFocusCrop","SpEdE_LocalObjectName","SpEdE_LocalHeroConsistency"):
        assert key in NODE_CLASS_MAPPINGS


def test_support_plane_uses_the_base_ring_for_a_tall_object():
    from spede_core.patch import fit_view_support_plane
    # A camera looking along +z at a floor (y = 1.5, image bottom) and a wall
    # (z = 6, image top); the object mask is a tall column whose upper ring
    # sees only the wall, a plane that passes RANSAC on the full ring.
    cam=SimpleNamespace(view_id="c",viewmat=np.eye(4,dtype=np.float32),
                        intrinsic=np.array([[60,0,32],[0,60,32],[0,0,1]],np.float32),width=64,height=64)
    rng=np.random.default_rng(0)
    pts=[]
    for v in range(64):
        for u in range(64):
            ray=np.array([(u+0.5-32)/60,(v+0.5-32)/60,1.0])
            t=1.5/ray[1] if ray[1]>0.1 else 6.0           # floor below, wall above
            pts.append(ray*t+rng.normal(0,0.002,3))
    class Scene:
        means=np.asarray(pts)
        opacity_logits=np.full((len(pts),1),5.0)
    mask=np.zeros((64,64),bool)
    mask[4:52,26:38]=True
    plane,_tol=fit_view_support_plane(Scene,cam,mask,ring_px=8,minimum_births=4)
    assert plane.ring=="base"
    assert abs(abs(plane.normal[1])-1)<0.05                # the floor, not the wall
    flat=np.zeros((64,64),bool)
    flat[50:58,20:44]=True                                  # low and wide: the base ring is the floor too
    assert abs(abs(fit_view_support_plane(Scene,cam,flat,ring_px=4,minimum_births=4)[0].normal[1])-1)<0.05


def test_support_plane_ignores_gaussians_behind_the_visible_surface():
    from spede_core.patch import fit_view_support_plane
    # A tabletop (y = 1.5) seen from above, with as many Gaussians of a
    # tilted floor far behind it projecting into the same ring pixels:
    # only the surface the camera sees around the object is support.
    cam=SimpleNamespace(view_id="c",viewmat=np.eye(4,dtype=np.float32),
                        intrinsic=np.array([[60,0,32],[0,60,32],[0,0,1]],np.float32),width=64,height=64)
    rng=np.random.default_rng(1)
    pts=[]
    for v in range(64):
        for u in range(64):
            ray=np.array([(u+0.5-32)/60,(v+0.5-32)/60,1.0])
            if ray[1]>0.1:
                pts.append(ray*1.5/ray[1]+rng.normal(0,0.002,3))      # the tabletop
                far=ray*(6.0+3.0*ray[0])                               # behind it, tilted
                pts.append(far+rng.normal(0,0.002,3))
            else:
                pts.append(ray*6.0+rng.normal(0,0.002,3))
    class Scene:
        means=np.asarray(pts)
        opacity_logits=np.full((len(pts),1),5.0)
    mask=np.zeros((64,64),bool)
    mask[30:52,26:38]=True
    plane,_tol=fit_view_support_plane(Scene,cam,mask,ring_px=8,minimum_births=4)
    assert abs(abs(plane.normal[1])-1)<0.05
    assert abs(abs(plane.offset)-1.5)<0.05


def test_planar_patch_births_stay_on_the_tabletop_not_in_the_air(tmp_path):
    # A narrow tabletop (y = 1.5, depth below 5) with a wall at z = 6 behind
    # and beside it, so the base ring sees both, as a bonsai on a table with
    # a bicycle beside it does. Rays through the upper part of the removed
    # object's mask meet the tabletop's plane beyond the table, in the air:
    # no births there.
    cam=camera()
    pts=[]
    mask=np.zeros((64,64),bool)
    mask[30:60,26:38]=True
    for v in range(64):
        for u in range(64):
            if mask[v,u]:
                continue
            ray=np.array([(u+0.5-32)/60,(v+0.5-32)/60,1.0])
            pts.append(ray*1.5/ray[1] if ray[1]>0.3 and 22<=u<42 else ray*6.0)
    src=tmp_path/"table.ply"
    make_scene(src,np.asarray(pts))
    image=np.full((64,64,3),0.5,np.float32)
    out=tmp_path/"patched.ply"
    result=patch_support_plane(src,cam,mask,image,out,remaining_alpha=np.zeros(mask.shape,np.float32),
                               remaining_depth=np.zeros(mask.shape,np.float32),ring_px=8,pixel_stride=2)
    births=PlyData.read(out)["vertex"].data[len(pts):]
    assert result["births"]>=12
    assert np.abs(births["y"]-1.5).max()<0.05
    assert births["z"].max()<5.0+0.3


def test_separate_cluster_ids_drops_distant_objects_keeps_the_anchored_one():
    from spede_core.removal import separate_cluster_ids

    rng = np.random.default_rng(0)
    obj = rng.uniform(0, 1, (4000, 3))                       # the clicked object
    part = rng.uniform(0, 0.05, (50, 3)) + [1.05, 0.5, 0.5]  # a detached part next to it
    toy = rng.uniform(0, 0.3, (3000, 3)) + [6.0, 0, 0]       # a separate object far away
    big = rng.uniform(0, 1.5, (9000, 3)) + [0, 8.0, 0]       # larger, but not clicked
    means = np.vstack([obj, part, toy, big])
    ids = np.arange(len(means))
    anchor = np.zeros(len(means), bool)
    anchor[:4000] = True
    dropped = separate_cluster_ids(means, ids, anchor)
    assert set(dropped.tolist()) == set(range(4050, len(means)))
    # Without an anchor the largest piece is the object.
    assert set(separate_cluster_ids(means, ids).tolist()) == set(range(0, 7050))
    # One piece: nothing to drop.
    assert separate_cluster_ids(obj, np.arange(4000)).size == 0
