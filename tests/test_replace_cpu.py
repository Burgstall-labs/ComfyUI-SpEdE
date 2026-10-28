"""CPU contract checks for scene-aware insertion and object registration."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from spede_core.replace import image_object_assets, replacement_prior, merge_tripo_ply
try:                                    # this repository
    from competition.nodes import NODE_CLASS_MAPPINGS
    from competition.local_sdk import REPLACE_SIZES, TURBO_SIGMAS, build_local_graph, build_replace_graph, replace_noun
except ImportError:                     # the exported node pack has these at its root
    from nodes import NODE_CLASS_MAPPINGS
    from local_sdk import REPLACE_SIZES, TURBO_SIGMAS, build_local_graph, build_replace_graph, replace_noun


def ply(path, positions):
    pos=np.asarray(positions,np.float32)
    fields=[(n,"<f4") for n in
            ("x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity",
             "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3")]
    fields.append(("custom","<i4"))
    v=np.zeros(len(pos),dtype=np.dtype(fields))
    for i,k in enumerate(("x","y","z")):v[k]=pos[:,i]
    for k in ("scale_0","scale_1","scale_2"):v[k]=np.log(.04)
    v["rot_0"]=1
    v["opacity"]=3
    v["custom"]=np.arange(len(pos))+10
    PlyData([PlyElement.describe(v,"vertex")],text=False).write(str(path))
    return v


def cam():
    return SimpleNamespace(
        view_id="camera_0",width=96,height=96,
        viewmat=np.eye(4,dtype=np.float32),
        intrinsic=np.array([[90,0,48],[0,90,48],[0,0,1]],np.float32)
    )


def test_prepare_rgba_and_separate_shadow_candidate(tmp_path):
    clean=np.full((96,96,3),.7,np.float32)
    edited=clean.copy()
    mask=np.zeros((96,96),bool)
    mask[32:63,38:61]=True
    edited[mask]=[.85,.20,.11]
    # Model-suggested contact darkening immediately outside object.
    edited[63:72,38:61]*=.7
    data=image_object_assets(edited,clean,mask,tmp_path,crop_size=128)
    rgba=np.asarray(Image.open(data["image_object_rgba"]))
    shadow=np.asarray(Image.open(data["shadow_candidate_png"]))
    assert rgba.shape==(128,128,4)
    assert rgba[:,:,3].max()==255
    assert data["shadow_is_only_candidate"] is True
    assert not shadow[mask].any()
    assert shadow[65:70,40:59].max()>0
    assert data["tripo_inference_completed"] is False


def test_replacement_prior_uses_real_gaussian_world_bbox(tmp_path):
    yy,xx=np.mgrid[-2:3,-2:3]
    pts=np.column_stack([xx.ravel()*.08,yy.ravel()*.08,
                         np.full(xx.size,3.)])
    path=tmp_path/"source.ply"
    ply(path,pts)
    mask=np.zeros((96,96),bool)
    mask[37:64,36:65]=True
    data=replacement_prior(path,np.arange(len(pts)),cam(),mask)
    assert data["source_geometry_authoritative"] is True
    assert data["method"]=="old_object_bbox_and_camera_projection"
    assert data["target_uniform_extent"]>0
    assert data["new_world_center"][2]==pytest.approx(3,abs=.2)
    assert data["registration_has_been_applied"] is False


def test_register_splat_preserves_existing_scene_and_uses_extent_prior(tmp_path):
    scene=tmp_path/"background.ply"
    src=ply(scene,[[0,0,3],[1,0,3],[0,1,3]])
    generated=tmp_path/"generated.ply"
    obj=ply(generated,[[-1,-1,0],[1,1,0],[0,0,1]])
    result=merge_tripo_ply(
        scene,generated,
        {"new_world_center":[4,1,3],"target_uniform_extent":2},
        tmp_path/"merged.ply"
    )
    output=PlyData.read(tmp_path/"merged.ply")["vertex"].data
    np.testing.assert_array_equal(output[:len(src)],src)
    assert result["object_gaussians"]==len(obj)
    assert result["result_gaussians"]==6
    assert result["automatic_axis_yaw_alignment"] is False
    assert result["contact_shadow_applied"] is False
    assert np.median(output["x"][3:])==pytest.approx(4,abs=.1)


def test_replacement_rejects_bad_assets_and_invalid_scene(tmp_path):
    clean=np.ones((64,64,3),np.float32)
    empty=np.zeros((64,64),bool)
    with pytest.raises(ValueError,match="segmentation"):
        image_object_assets(clean,clean,empty,tmp_path)
    scene=tmp_path/"scene.ply"
    ply(scene,[[0,0,3],[1,0,3],[0,1,3]])
    gen=tmp_path/"gen.ply"
    ply(gen,[[0,0,0],[0,0,0],[0,0,0]])
    with pytest.raises(ValueError,match="no valid scale"):
        merge_tripo_ply(scene,gen,{
            "new_world_center":[0,0,3],"target_uniform_extent":2
        },tmp_path/"bad.ply")


def test_replace_nodes_expose_optional_staging_contract():
    assert "SpEdE_LocalPrepareReplace" in NODE_CLASS_MAPPINGS
    assert "SpEdE_LocalRegisterSplat" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["SpEdE_LocalRegisterSplat"].INPUT_TYPES()["required"]["generated_object_ply"][0]=="STRING"



def test_qwen_sam_replace_preparation_workflow_is_concretely_connected(tmp_path):
    source=tmp_path/"scene.ply"
    ply(source,[[0,0,3],[.2,0,3],[0,.2,3],[.2,.2,3]])
    camera_pack=tmp_path/"cameras.npz"
    labels=[f"camera_{i}" for i in range(4)]
    np.savez(camera_pack,
        targets=np.zeros((4,96,96,3),np.uint8),
        labels=np.array(labels),
        viewmats=np.tile(np.eye(4)[None],(4,1,1)),
        intrinsics=np.tile(cam().intrinsic[None],(4,1,1)),
        frame_indices=np.arange(4))
    graph=build_local_graph(
        "replace_prep",source_ply=source,camera_targets=camera_pack,
        hero_view="camera_2",object_text="armchair",job_name="obj"
    )
    assert graph["62"]["class_type"]=="MoGeInference"
    assert graph["64"]["class_type"]=="SpEdE_LocalPatchFromDepth"
    assert graph["81"]["class_type"]=="TextEncodeQwenImage21"
    assert graph["81"]["inputs"]["images.image_1"]==["80",0]
    assert graph["83"]["class_type"]=="VAEDecode"
    assert graph["85"]["class_type"]=="SAM3_Detect"
    assert graph["85"]["inputs"]["image"]==["83",0]
    assert graph["86"]["inputs"]["new_object_mask"]==["85",0]
    assert graph["87"]["inputs"]["images"]==["86",0]
    assert graph["80"]["inputs"]["view_id"]=="camera_2"
    assert graph["87"]["inputs"]["filename_prefix"].startswith("spede/local/obj/")


def _replace_inputs(tmp_path):
    source=tmp_path/"scene.ply"
    ply(source,[[0,0,3],[.2,0,3],[0,.2,3],[.2,.2,3]])
    extract=tmp_path/"object_only.ply"
    ply(extract,[[0,.2,3],[.2,.2,3]])
    mask=tmp_path/"hero_mask.png"
    Image.new("L",(96,96),255).save(mask)
    camera_pack=tmp_path/"cameras.npz"
    np.savez(camera_pack,targets=np.zeros((4,96,96,3),np.uint8),labels=np.array([f"camera_{i}" for i in range(4)]),
             viewmats=np.tile(np.eye(4)[None],(4,1,1)),intrinsics=np.tile(cam().intrinsic[None],(4,1,1)),
             frame_indices=np.arange(4))
    return {"source_ply":source,"camera_targets":camera_pack},extract,mask


KNOWN_REPLACE_NODES={"CheckpointLoaderSimple","CLIPTextEncode","SAM3_Detect","UNETLoader","CLIPLoader",
    "VAELoader","QwenImage21Cache","TextEncodeQwenImage21","RandomNoise","CFGGuider","KSamplerSelect",
    "ManualSigmas","SamplerCustomAdvanced","VAEDecode","SaveImage","LoadImage","TripoSplatPreprocessImage",
    "CLIPVisionLoader","TripoSplatConditioning","KSampler","VAEDecodeTripoSplat","SplatToFile3D"}


def _connected(graph):
    known=set(NODE_CLASS_MAPPINGS)|KNOWN_REPLACE_NODES
    for node_id,node in graph.items():
        assert node["class_type"] in known,node_id
        for v in node["inputs"].values():
            if isinstance(v,list) and len(v)==2:
                assert str(v[0]) in graph,(node_id,v)


def test_replace_graph_wires_qwen_sam_triposplat_and_placement(tmp_path):
    inputs,extract,_mask=_replace_inputs(tmp_path)
    graph=build_replace_graph(**inputs,object_ply=extract,hero_view="camera_1",text="a large green ceramic vase",
                              removed_label="lego bulldozer",seed=7,max_shift=0.3,job_name="rep")
    _connected(graph)
    # No reference picture: LoadImage and <image3> are gone; an extract on record: no Extract nodes.
    assert "3" not in graph and "images.image_3" not in graph["34"]["inputs"]
    assert not any(k.startswith("x") for k in graph) and "5" not in graph
    assert graph["1"]["inputs"]["source_ply"]==str(inputs["source_ply"].resolve())
    # The inverse-extract base: the source without the extract, rendered in the hero camera for Qwen.
    base=graph["10"]
    assert base["class_type"]=="SpEdE_LocalReplaceBase" and base["inputs"]["scene"]==["1",0]
    assert base["inputs"]["object_ply"]==str(extract.resolve()) and base["inputs"]["hero_image_path"]==""
    assert base["inputs"]["remove_fill_ply"]=="" and base["inputs"]["output_path"]=="spede/local/rep/replace_base.ply"
    # Qwen turbo: the cleaned base first, the original photo second; 8 steps, cfg 1.
    assert graph["30"]["inputs"]["unet_name"]=="qwen_image_2.1_turbo_int8_convrot.safetensors"
    assert graph["34"]["inputs"]["images.image_1"]==["10",1] and graph["34"]["inputs"]["images.image_2"]==["27",0]
    assert graph["44"]["inputs"]["images"]==["10",1]
    assert graph["34"]["inputs"]["resolution"]==1024
    assert graph["35d"]["inputs"]["sigmas"]==TURBO_SIGMAS and graph["35b"]["inputs"]["cfg"]==1.0
    assert graph["35a"]["inputs"]["noise_seed"]==7
    prompt=graph["34"]["inputs"]["prompt"]
    assert prompt.startswith("Replace the empty spot in <image1> with a large green ceramic vase.") and "lego bulldozer" in prompt
    assert "the large green ceramic vase is the only new thing" in prompt.lower()
    # Match (the default) asks Qwen for the old object's size too.
    assert "The large green ceramic vase is about as big as the lego bulldozer was" in prompt
    assert "<image3>" not in prompt
    # SAM on Qwen's picture by the noun; the cutout feeds TripoSplat and the placement.
    assert graph["84"]["inputs"]["text"]=="large green ceramic vase"
    assert graph["85"]["inputs"]["image"]==["36",0]
    assert graph["90"]["inputs"]["mask"]==["86",1]
    place=graph["100"]["inputs"]
    assert graph["100"]["class_type"]=="SpEdE_LocalPlaceGenerated" and place["scene"]==["1",0]
    assert place["generated"]==["98",0] and place["object_mask"]==["86",1] and place["inserted_image"]==["36",0]
    assert place["object_ply"]==str(extract.resolve()) and place["max_shift"]==0.3 and place["size"]=="match"
    assert place["output_path"]=="spede/local/rep/replaced_scene.ply"
    assert all(n["inputs"]["view_id"]=="camera_1" for n in graph.values() if "view_id" in n["inputs"])
    assert all(graph[n]["inputs"]["filename_prefix"].startswith("spede/local/rep/") for n in ("42","43","44","87"))
    # The node's size widget offers the modes the builder sends.
    sizes=NODE_CLASS_MAPPINGS["SpEdE_LocalPlaceGenerated"].INPUT_TYPES()["required"]["size"][0]
    assert sorted(sizes)==sorted(REPLACE_SIZES)


def test_replace_graph_extracts_first_without_an_extract_and_uses_a_remove(tmp_path):
    inputs,_extract,mask=_replace_inputs(tmp_path)
    removed=tmp_path/"patched.ply"
    ply(removed,[[0,0,3],[.2,0,3],[.1,.1,3.1]])
    hero=tmp_path/"qwen_edit.png"
    Image.new("RGB",(96,96)).save(hero)
    graph=build_replace_graph(**inputs,hero_mask=mask,remove_fill_ply=removed,hero_image=hero,hero_view="camera_1",
                              text="vase",removed_label="lego bulldozer",size="natural",job_name="rep")
    _connected(graph)
    # SAM 3.1 in every camera by the old name, the selection without the removal footprint, Extract with its trim.
    assert "lego bulldozer" in graph["x21"]["inputs"]["text"]
    assert graph["x22"]["inputs"]["model"]==["20",0] and graph["x22"]["inputs"]["image"]==["x3",0]
    assert graph["x23"]["inputs"]["hero_mask_path"]==str(mask.resolve())
    assert graph["x24"]["inputs"]["include_footprint"] is False and graph["x24"]["inputs"]["scene"]==["1",0]
    assert graph["x50"]["class_type"]=="SpEdE_LocalExtract"
    assert graph["x50"]["inputs"]["output_subdir"]=="spede/local/rep/extract"
    for node in ("10","100"):
        assert graph[node]["inputs"]["object_ply"]==["x50",0]
        assert graph[node]["inputs"]["remove_fill_ply"]==str(removed.resolve())
    assert graph["10"]["inputs"]["hero_image_path"]==str(hero.resolve())
    # On the Remove's result: that scene is the base (both nodes), the Remove fill is not used; the extract
    # (here made first) still measures the old object.
    on_remove=build_replace_graph(**inputs,hero_mask=mask,remove_scene_ply=removed,remove_fill_ply=removed,
                                  hero_image=hero,hero_view="camera_1",text="vase",removed_label="lego bulldozer")
    _connected(on_remove)
    for node in ("10","100"):
        assert on_remove[node]["inputs"]["remove_scene_ply"]==str(removed.resolve())
        assert on_remove[node]["inputs"]["remove_fill_ply"]=="" and on_remove[node]["inputs"]["object_ply"]==["x50",0]
    assert on_remove["1"]["inputs"]["source_ply"]==str(inputs["source_ply"].resolve())
    for name in ("SpEdE_LocalReplaceBase","SpEdE_LocalPlaceGenerated"):
        assert "remove_scene_ply" in NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["optional"]
    # Natural: Qwen's size, no size words.
    assert graph["100"]["inputs"]["size"]=="natural" and "as big as" not in graph["34"]["inputs"]["prompt"]
    with_words=build_replace_graph(**inputs,hero_mask=mask,hero_view="camera_1",text="vase",removed_label="bulldozer",
                                   size="natural",size_words=True)
    assert "about as big as the bulldozer was" in with_words["34"]["inputs"]["prompt"]
    larger=build_replace_graph(**inputs,hero_mask=mask,hero_view="camera_1",text="vase",removed_label="bulldozer",
                               size="larger")
    assert "bigger than the bulldozer was" in larger["34"]["inputs"]["prompt"]


def test_replace_graph_with_a_reference_picture_and_refusals(tmp_path):
    inputs,extract,_mask=_replace_inputs(tmp_path)
    graph=build_replace_graph(**inputs,object_ply=extract,hero_view="camera_1",text="The fruit bowl.",
                              removed_label="bulldozer",reference_image="spede_replace_p_1.png")
    assert graph["3"]["inputs"]["image"]=="spede_replace_p_1.png"
    assert graph["34"]["inputs"]["images.image_3"]==["3",0]
    prompt=graph["34"]["inputs"]["prompt"]
    assert prompt.startswith("Replace the empty spot in <image1> with the fruit bowl shown in <image3>")
    assert "The fruit bowl is about as big as the bulldozer was" in prompt
    assert graph["84"]["inputs"]["text"]=="fruit bowl"
    assert replace_noun("an old red   teapot with a long thin spout")=="old red teapot with a long"
    with pytest.raises(ValueError,match="what to put there"):
        build_replace_graph(**inputs,object_ply=extract,hero_view="camera_1",text=" the ",removed_label="x")
    with pytest.raises(ValueError,match="missing from camera pack"):
        build_replace_graph(**inputs,object_ply=extract,hero_view="nope",text="vase",removed_label="x")
    with pytest.raises(ValueError,match="max_shift"):
        build_replace_graph(**inputs,object_ply=extract,hero_view="camera_1",text="vase",removed_label="x",max_shift=2)
    with pytest.raises(ValueError,match="size must be"):
        build_replace_graph(**inputs,object_ply=extract,hero_view="camera_1",text="vase",removed_label="x",size="huge")
    with pytest.raises(ValueError,match="hero mask"):
        build_replace_graph(**inputs,hero_view="camera_1",text="vase",removed_label="x")


def test_object_cutout_fills_holes_and_keeps_the_largest_region():
    torch=pytest.importorskip("torch")
    image=torch.zeros((1,32,32,3))
    mask=torch.zeros((1,32,32))
    mask[0,4:20,4:20]=1
    mask[0,10:14,10:14]=0                # the fruit SAM left out of the bowl
    mask[0,26:29,26:29]=1                # a stray bit elsewhere
    out_image,out_mask,white=NODE_CLASS_MAPPINGS["SpEdE_LocalObjectCutout"]().cut(image,mask,"bowl")
    m=out_mask[0].numpy()>0.5
    assert m[4:20,4:20].all() and m.sum()==256
    assert float(white[0,0,0,0])==1.0 and float(white[0,8,8,0])==0.0
    with pytest.raises(ValueError,match="found no 'bowl'"):
        NODE_CLASS_MAPPINGS["SpEdE_LocalObjectCutout"]().cut(image,torch.zeros((1,32,32)),"bowl")
