#!/usr/bin/env python3
"""Run a full local SpEdE competition workflow with the official Comfy SDK v2.

No cloud fallback: COMFY_BASE_URL must point to loopback (comfy-api-proxy).
Dry run is the default. Qwen-Image 2.1 weights are user-supplied under their
own research/noncommercial terms and never downloaded by this script.

Example (from the node pack's folder; in the development repository run
``python -m competition.local_sdk`` from the repository root instead):
    export COMFY_BASE_URL=http://127.0.0.1:8189
    python local_sdk.py patch --source-ply /data/scene.ply \
        --camera-targets /data/cameras.npz --hero-view camera_0 \
        --object-text "chair" --job-name chair01 --submit \
        --accept-qwen-research-license
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

# spede_assets sits beside this file in the exported pack and one level up
# in the development repository; either way ``python local_sdk.py`` works.
# Only the directory that holds it goes first on the path, so the repository's
# competition/ folder never shadows its top-level ``nodes`` module.
for _p in (Path(__file__).resolve().parent,Path(__file__).resolve().parent.parent):
    if (_p/"spede_assets.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0,str(_p))
        break

from spede_assets import inspect_camera_pack

_ROOT=Path(__file__).resolve().parent
WORKFLOWS={
    "extract":"spede_local_sam31_extract_api.json",
    "patch":"spede_local_sam31_qwen21_patch_api.json",
    "depthfill":"spede_local_sam31_qwen21_moge3_depthfill_api.json",
    "replace_prep":"spede_local_replace_prepare_api.json",
    "hero":"spede_local_click_sam31_qwen3vl_hero_api.json",
    "allround1":"spede_local_allround_stage1_api.json",
    "allround_hero":"spede_local_allround_hero_api.json",
    "allround2":"spede_local_allround_stage2_api.json",
    "progressive":"spede_local_allround_progressive_api.json",
    "replace":"spede_local_replace_api.json",
}
ALLROUND_WORKFLOWS={"allround1","allround_hero","allround2","progressive"}
# Built by their own builders, not build_local_graph (main() offers neither).
OWN_BUILDERS=ALLROUND_WORKFLOWS|{"replace"}
FILL_METHODS=("independent","progressive")
MODELS={
    "sam3":"checkpoints/sam3.1_multiplex_fp16.safetensors",
    "diffusion":"diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "clip":"text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "vae":"vae/qwen_image_2.1_vae_bf16.safetensors",
    "moge":"geometry_estimation/moge_3_vitl_fp16.safetensors",
    "turbo":"diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
}
# Qwen-Image 2.1 turbo: 8 Euler steps on these sigmas (as the all-around cleans).
TURBO_SIGMAS="1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
HERO_MODELS=("full","turbo")
MODEL_LICENSE_URL="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"


def validate_local_endpoint() -> str:
    url=os.getenv("COMFY_BASE_URL","").strip()
    if not url:
        raise ValueError("Set COMFY_BASE_URL to local Comfy API v2 proxy; Cloud is not the default")
    parsed=urlparse(url)
    if parsed.scheme not in {"http","https"} or parsed.hostname not in {
        "127.0.0.1","localhost","::1"
    } or parsed.username or parsed.password:
        raise ValueError("Competition local mode only permits loopback Comfy API v2")
    return url


def _safe_job_name(value:str)->str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",value):
        raise ValueError("job name must be 1–64 alphanumeric, hyphen/underscore characters")
    return value


def _named_prompt(prompt:str,object_text:str)->str:
    """The removal prompt with the object's name: "the segmented foreground
    object" alone left Qwen-Image guessing. Playroom (23 Oct): with a
    removed render still showing dark residue where the buggy was, it kept
    the buggy; told "the buggy", it removed it."""
    name=str(object_text).strip()
    return prompt.replace("the segmented foreground object",f"the {name}") if name else prompt


# SAM3_Detect keeps one detection per text prompt unless the prompt asks
# for more ("name:N", comfy/text_encoders/sam3_clip.py). With one, a camera
# where another object of the kind scores higher only has that one; with a
# few, SpEdE_LocalMasksToReport keeps them apart and the selection picks the
# ones that hold the hero's object (spede_core.selection.pick_hero_views).
SAM_INSTANCES=4


def _sam_instances_prompt(text:str,count:int=SAM_INSTANCES)->str:
    """``text`` with ``:count`` on each comma-separated prompt that has no count of its own."""
    parts=[p.strip() for p in str(text).split(",") if p.strip()]
    return ", ".join(p if re.search(r":\s*[\d.]+\s*$",p) else f"{p}:{int(count)}" for p in parts)


def build_local_graph(
    operation:str,
    *,source_ply:str|Path,camera_targets:str|Path,hero_view:str,
    object_text:str,job_name:str="local",qwen_prompt:str|None=None,
    seed:int=241007,
    checkpoint_names:dict[str,str]|None=None,
    hero_mask:str|Path|None=None,
    max_spill_fraction:float|None=None,
    hero_model:str="full",
)->dict:
    if hero_model not in HERO_MODELS:
        raise ValueError(f"hero_model must be one of {HERO_MODELS}")
    if operation not in {"extract","patch","depthfill","replace_prep"}:
        raise ValueError(f"Unknown local workflow: {operation} (the hero check is build_hero_graph)")
    job=_safe_job_name(job_name)
    source=Path(source_ply).expanduser().resolve(strict=True)
    targets=Path(camera_targets).expanduser().resolve(strict=True)
    if source.suffix.lower()!=".ply":
        raise ValueError("Expected a Gaussian .ply source file")
    ids=inspect_camera_pack(targets)["view_ids"]
    if hero_view not in ids:
        raise ValueError(f"Hero camera {hero_view!r} missing from camera pack")
    if len(ids)<4:
        raise ValueError("SpEdE needs hero plus >=3 real voting cameras")
    graph=json.loads((_ROOT/"workflows"/WORKFLOWS[operation]).read_text())
    graph["1"]["inputs"]["source_ply"]=str(source)
    graph["2"]["inputs"]["camera_targets_npz"]=str(targets)
    if not str(object_text).strip():
        raise ValueError("SAM3.1 needs an object text prompt")
    graph["21"]["inputs"]["text"]=_sam_instances_prompt(str(object_text).strip())
    graph["23"]["inputs"]["hero_view_id"]=hero_view
    graph["23"]["inputs"]["object_text"]=str(object_text).strip()
    if hero_mask is not None:
        graph["23"]["inputs"]["hero_mask_path"]=str(Path(hero_mask).expanduser().resolve(strict=True))
    # Removal needs the footprint column (pieces the hull missed under the
    # object); extraction must not take it, or the export gets cloth discs.
    graph["24"]["inputs"]["include_footprint"]=operation in {"patch","depthfill","replace_prep"}
    for n in graph.values():
        for k,v in n["inputs"].items():
            if isinstance(v,str) and v.startswith("spede/local/"):
                n["inputs"][k]=v.replace("spede/local/",f"spede/local/{job}/",1)
    if operation in {"patch","depthfill","replace_prep"}:
        for node in graph.values():
            if "view_id" in node["inputs"]:
                node["inputs"]["view_id"]=hero_view
        # The pictures land in the job's own output folder with the PLY and
        # the reports, not under a shared prefix with a running counter.
        graph["42"]["inputs"]["filename_prefix"]=f"spede/local/{job}/qwen21_edit"
        graph["43"]["inputs"]["filename_prefix"]=f"spede/local/{job}/patched_render"
        if operation=="replace_prep":
            graph["87"]["inputs"]["filename_prefix"]=f"spede/local/{job}/replace_object"
            graph["88"]["inputs"]["filename_prefix"]=f"spede/local/{job}/replace_scene_insertion"
        if qwen_prompt is not None:
            if not qwen_prompt.strip():
                raise ValueError("Qwen prompt cannot be empty")
            graph["34"]["inputs"]["prompt"]=qwen_prompt
        else:
            graph["34"]["inputs"]["prompt"]=_named_prompt(graph["34"]["inputs"]["prompt"],object_text)
        if not 0<=int(seed)<2**63:
            raise ValueError("seed must fit positive 64-bit signed integer")
        graph["35"]["inputs"]["seed"]=int(seed)
        if hero_model=="turbo":
            _use_turbo(graph)
    if operation in {"depthfill","replace_prep"}:
        graph["65"]["inputs"]["filename_prefix"]=f"spede/local/{job}/aligned_moge_depth"
        # The Qwen spill gate (SpEdE_LocalAlignMoGeDepth); the UI's "Remove
        # anyway" raises it to the node's maximum when the extra is only shadow.
        if max_spill_fraction is not None:
            graph["63"]["inputs"]["max_spill_fraction"]=float(max_spill_fraction)
    if checkpoint_names:
        if "sam3" in checkpoint_names:
            graph["20"]["inputs"]["ckpt_name"]=checkpoint_names["sam3"]
        if operation in {"patch","depthfill","replace_prep"}:
            for key,idx,arg in (
                ("diffusion","30","unet_name"),
                ("clip","31","clip_name"),
                ("vae","32","vae_name"),
            ):
                if key in checkpoint_names:
                    graph[idx]["inputs"][arg]=checkpoint_names[key]
            if operation in {"depthfill","replace_prep"} and "moge" in checkpoint_names:
                graph["61"]["inputs"]["model_name"]=checkpoint_names["moge"]
    return graph


def _use_turbo(graph:dict)->None:
    """The hero repair on Qwen-Image 2.1 turbo: node 35's 25-step KSampler
    becomes a SamplerCustomAdvanced on the turbo sigmas, same seed."""
    k=graph["35"]["inputs"]
    graph["30"]["inputs"]["unet_name"]=Path(MODELS["turbo"]).name
    graph["35a"]={"class_type":"RandomNoise","inputs":{"noise_seed":k["seed"]}}
    graph["35b"]={"class_type":"CFGGuider","inputs":{"model":k["model"],"positive":k["positive"],
                                                     "negative":k["negative"],"cfg":1.0}}
    graph["35c"]={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}}
    graph["35d"]={"class_type":"ManualSigmas","inputs":{"sigmas":TURBO_SIGMAS}}
    graph["35"]={"class_type":"SamplerCustomAdvanced","inputs":{"noise":["35a",0],"guider":["35b",0],
        "sampler":["35c",0],"sigmas":["35d",0],"latent_image":k["latent_image"]}}


def build_hero_graph(
    *,camera_targets:str|Path,hero_view:str,points:list,object_text:str|None=None,
    job_name:str="local",checkpoint_names:dict[str,str]|None=None,
)->dict:
    """The hero-view check: clicks ([x, y] or [x, y, 1/0] in hero-image
    pixels) -> SAM3.1 mask -> Qwen3-VL name -> SAM3.1 by that name ->
    SpEdE consistency. With ``object_text`` the user's own name replaces
    the Qwen3-VL naming (the VL nodes are dropped)."""
    job=_safe_job_name(job_name)
    targets=Path(camera_targets).expanduser().resolve(strict=True)
    if hero_view not in inspect_camera_pack(targets)["view_ids"]:
        raise ValueError(f"Hero camera {hero_view!r} missing from camera pack")
    rows=[(float(p[0]),float(p[1]),int(p[2]) if len(p)>2 else 1) for p in points]
    positive=[{"x":round(x),"y":round(y)} for x,y,label in rows if label]
    negative=[{"x":round(x),"y":round(y)} for x,y,label in rows if not label]
    if not positive:
        raise ValueError("click the object at least once")
    graph=json.loads((_ROOT/"workflows"/WORKFLOWS["hero"]).read_text())
    graph["2"]["inputs"]["camera_targets_npz"]=str(targets)
    graph["4"]["inputs"]["view_id"]=hero_view
    for node in ("10","15"):
        graph[node]["inputs"]["positive_coords"]=json.dumps(positive)
        graph[node]["inputs"]["negative_coords"]=json.dumps(negative)
    graph["15"]["inputs"]["output_subdir"]=f"spede/local/{job}/hero"
    graph["16"]["inputs"]["filename_prefix"]=f"spede/local/{job}/hero/preview"
    name=str(object_text or "").strip()
    if name:
        for node in ("11","12","13","14"):
            del graph[node]
        graph["21"]["inputs"]["text"]=name
        graph["15"]["inputs"]["name"]=name
    if checkpoint_names:
        if "sam3" in checkpoint_names:
            graph["20"]["inputs"]["ckpt_name"]=checkpoint_names["sam3"]
        if "clip" in checkpoint_names and "12" in graph:
            graph["12"]["inputs"]["clip_name"]=checkpoint_names["clip"]
    return graph


def build_allround_graphs(
    *,run_dir:str|Path,source_ply:str|Path,camera_targets:str|Path,object_text:str,
    job_name:str="local",photo_dir:str|Path|None=None,chain_views:int=10,orbit_frames:int=260,
    resolution:int=1024,iterations:int=10000,seed:int=241007,
    second:str="progressive",step_iterations:int=1500,final_iterations:int=6000,
    method:str="independent",
)->tuple[dict,dict]:
    """The all-around fill's two prompts for a finished depth-fill run.
    ``second`` "progressive" (the product): the lean stage 1 (plan and the
    hero's 1024 clean only), then the fill node with ``method``
    "independent" (Qwen-Image turbo cleans every camera on its own, one
    optimisation; ``final_iterations``) or "progressive" (one camera at a
    time with births and short optimisations; ``step_iterations`` too).
    ``second`` "orbit": the full stage 1 (also the Qwen chain, MoGe births
    and orbit warp), then the H3 orbit video and optimisation
    (``iterations``). All write under spede/allround/<job>."""
    if second not in {"progressive","orbit"}:
        raise ValueError(f"second must be 'progressive' or 'orbit', not {second!r}")
    if method not in FILL_METHODS:
        raise ValueError(f"method must be one of {FILL_METHODS}, not {method!r}")
    job=_safe_job_name(job_name)
    run=Path(run_dir).expanduser().resolve(strict=True)
    source=Path(source_ply).expanduser().resolve(strict=True)
    targets=Path(camera_targets).expanduser().resolve(strict=True)
    if not str(object_text).strip():
        raise ValueError("the all-around fill needs the removed object's name")
    if (int(orbit_frames)-5)%17:
        raise ValueError("orbit_frames must be 17k+5")
    one=json.loads((_ROOT/"workflows"/WORKFLOWS["allround1" if second=="orbit" else "allround_hero"]).read_text())
    two=json.loads((_ROOT/"workflows"/WORKFLOWS["allround2" if second=="orbit" else "progressive"]).read_text())
    for graph in (one,two):
        for node in graph.values():
            for k,v in node["inputs"].items():
                if isinstance(v,str) and "/JOB" in v:
                    node["inputs"][k]=v.replace("/JOB",f"/{job}")
    start=one["70"]["inputs"]
    start.update(run_dir=str(run),source_ply=str(source),camera_targets_npz=str(targets),
                 object_text=str(object_text).strip(),chain_views=int(chain_views),
                 orbit_frames=int(orbit_frames),resolution=int(resolution),
                 photo_dir=str(Path(photo_dir).expanduser()) if photo_dir else "")
    one["35"]["inputs"]["seed"]=int(seed)
    one["34"]["inputs"]["prompt"]=_named_prompt(one["34"]["inputs"]["prompt"],object_text)
    if second=="orbit":
        one["71"]["inputs"]["seed"]=int(seed)
        two["91"]["inputs"]["iterations"]=int(iterations)
    else:
        two["95"]["inputs"].update(seed=int(seed),step_iterations=int(step_iterations),
                                   final_iterations=int(final_iterations),method=method)
    return one,two


# Replace: Qwen-Image 2.1 turbo draws the new object into the current scene's
# hero render (<image1>); the original photo (<image2>) only says where the
# removed object stood and how large it was; a reference picture is <image3>.
# "Add X where the bulldozer stands in <image2>" brought the bulldozer back in
# 3 of 3 seeds on kitchen (27 Oct 2026, a large vase drawn behind it); framed
# as filling the empty spot, with the old object said to be gone, 0 of 9
# (vase, teapot, cactus) and 0 of 3 with a reference picture (fruit bowl).
REPLACE_PROMPT=("Replace the empty spot in <image1> with {what}. <image1> is the scene after the {old} was taken away; "
                "the new object stands on the surface exactly where the {old} stood in <image2>. <image2> is the same "
                "camera before the edit and shows only where that spot is and how large it is: the {old} is gone and "
                "must not appear in the result.{size} The {noun} is the only new thing in the picture, entirely inside "
                "the picture, standing on the surface with a soft natural contact shadow and lit like the rest of the "
                "scene. Keep the camera, perspective and every other part of <image1> unchanged.")
REPLACE_REFERENCE_PROMPT=("Replace the empty spot in <image1> with the {noun} shown in <image3>: the same object, with "
                          "the same shape, materials and colours; ignore everything around it in <image3>. <image1> is "
                          "the scene after the {old} was taken away; the new object stands on the surface exactly where "
                          "the {old} stood in <image2>. <image2> is the same camera before the edit and shows only "
                          "where that spot is and how large it is: the {old} is gone and must not appear in the "
                          "result.{size} The {noun} is the only new thing in the picture, entirely inside the picture, "
                          "standing on the surface with a soft natural contact shadow and lit like the rest of the "
                          "scene. Keep the camera, perspective and every other part of <image1> unchanged.")
# Replace's size modes (SpEdE_LocalPlaceGenerated ``size``) and the words that steer Qwen's drawing toward
# them: a bigger drawing gives the cut-out more pixels and TripoSplat more detail when the 3D object is scaled up.
REPLACE_SIZES=("match","natural","larger")
REPLACE_SIZE_WORDS={
    "natural":"",
    "match":" The {noun} is about as big as the {old} was: it fills the same space the {old} took in <image2>.",
    "larger":" The {noun} is bigger than the {old} was: it fills more than the space the {old} took in <image2>.",
}
# The Extract nodes build_replace_graph puts in front when the project has no extract yet:
# extract graph node id -> id in the Replace graph (1, 2 and the SAM 3.1 loader 20 are shared).
_REPLACE_EXTRACT_IDS={"1":"1","2":"2","20":"20","3":"x3","21":"x21","22":"x22","23":"x23","24":"x24","50":"x50"}


def replace_noun(text:str)->str:
    """SAM 3.1's prompt for the new object: the user's words without a
    leading article, at most six words ("a large green ceramic vase" ->
    "large green ceramic vase")."""
    words=str(text).strip().rstrip(".").split()
    if words and words[0].lower() in {"a","an","the","some"}:
        words=words[1:]
    return " ".join(words[:6])


def build_replace_graph(
    *,source_ply:str|Path,camera_targets:str|Path,hero_view:str,text:str,removed_label:str,
    object_ply:str|Path|None=None,hero_mask:str|Path|None=None,remove_fill_ply:str|Path|None=None,
    hero_image:str|Path|None=None,remove_scene_ply:str|Path|None=None,reference_image:str|None=None,
    seed:int=241101,max_shift:float=0.25,
    size:str="match",size_words:bool|None=None,job_name:str="local",
    checkpoint_names:dict[str,str]|None=None,
)->dict:
    """Replace as an inverse extract: the new object goes into the source
    scene without exactly the extracted object (the mat round it untouched;
    the old contact footprint the new object leaves bare is filled). No
    Remove needed; with one, on the Remove's result (``remove_scene_ply``).

    ``object_ply`` is the project's Extract (object_only.ply); without it
    the Extract nodes (SAM 3.1 in every camera by ``removed_label``, the
    selection without the removal footprint, SpEdE_LocalExtract with its
    support trim) run first in the same job, from ``hero_mask``, writing
    under spede/local/<job>/extract. ``remove_scene_ply`` (a Remove's
    result: the all-around fill, else the hero patch) is the base as it is
    instead, the object gone and its spot filled there: nothing taken out or
    filled, the source and the extract only measure the old object (plane,
    footprint, size). ``remove_fill_ply`` (a Remove's scene, without
    ``remove_scene_ply``) gives the footprint fill its texture;
    ``hero_image`` (a Remove's cleaned hero picture) is what Qwen draws
    into, else the base rendered in the hero camera. ``text`` says what to put there; ``reference_image`` is a
    picture in ComfyUI's input folder (LoadImage name) of it. ``size``:
    match (the removed object's dominant extent), natural (Qwen's size) or
    larger; ``size_words`` (default: not for natural) asks Qwen for that
    size too. ``max_shift`` caps the move from the old footprint's centre
    (fraction of its size). Writes under spede/local/<job>/."""
    job=_safe_job_name(job_name)
    source=Path(source_ply).expanduser().resolve(strict=True)
    targets=Path(camera_targets).expanduser().resolve(strict=True)
    if hero_view not in inspect_camera_pack(targets)["view_ids"]:
        raise ValueError(f"Hero camera {hero_view!r} missing from camera pack")
    what=" ".join(str(text).split())
    noun=replace_noun(what)
    if not noun:
        raise ValueError("Replace needs a few words saying what to put there")
    old=" ".join(str(removed_label).split()) or "removed object"
    if not 0<=int(seed)<2**63:
        raise ValueError("seed must fit positive 64-bit signed integer")
    if not 0<=float(max_shift)<=1:
        raise ValueError("max_shift is a fraction of the old footprint, 0..1")
    if size not in REPLACE_SIZES:
        raise ValueError(f"size must be one of {REPLACE_SIZES}")
    if object_ply is None and hero_mask is None:
        raise ValueError("Replace needs the project's extract (object_ply) or the hero mask to extract it")
    graph=json.loads((_ROOT/"workflows"/WORKFLOWS["replace"]).read_text())
    graph["1"]["inputs"]["source_ply"]=str(source)
    graph["2"]["inputs"]["camera_targets_npz"]=str(targets)
    for node in graph.values():
        if "view_id" in node["inputs"]:
            node["inputs"]["view_id"]=hero_view
        for k,v in node["inputs"].items():
            if isinstance(v,str) and v.startswith("spede/local/"):
                node["inputs"][k]=v.replace("spede/local/",f"spede/local/{job}/",1)
    if object_ply is not None:
        objects=str(Path(object_ply).expanduser().resolve(strict=True))
    else:
        extract=build_local_graph("extract",source_ply=source,camera_targets=targets,hero_view=hero_view,
                                  object_text=old,job_name=job,hero_mask=hero_mask,checkpoint_names=checkpoint_names)
        for node_id,node in extract.items():
            new_id=_REPLACE_EXTRACT_IDS[node_id]
            if new_id in graph:
                continue                                  # shared: the same loader with the same input
            for k,v in node["inputs"].items():
                if isinstance(v,list) and len(v)==2 and isinstance(v[0],str):
                    node["inputs"][k]=[_REPLACE_EXTRACT_IDS[v[0]],v[1]]
            graph[new_id]=node
        objects=["x50",0]
    for node_id in ("10","100"):
        graph[node_id]["inputs"]["object_ply"]=objects
        if remove_scene_ply is not None:
            graph[node_id]["inputs"]["remove_scene_ply"]=str(Path(remove_scene_ply).expanduser().resolve(strict=True))
        elif remove_fill_ply is not None:
            graph[node_id]["inputs"]["remove_fill_ply"]=str(Path(remove_fill_ply).expanduser().resolve(strict=True))
    if hero_image is not None:
        graph["10"]["inputs"]["hero_image_path"]=str(Path(hero_image).expanduser().resolve(strict=True))
    words=size!="natural" if size_words is None else bool(size_words)
    sized=REPLACE_SIZE_WORDS["match" if size=="natural" else size] if words else ""
    prompt=REPLACE_REFERENCE_PROMPT if reference_image else REPLACE_PROMPT
    graph["34"]["inputs"]["prompt"]=prompt.format(what=what,noun=noun,old=old,size=sized.format(noun=noun,old=old))
    if reference_image:
        graph["3"]["inputs"]["image"]=str(reference_image)
    else:
        del graph["3"]
        del graph["34"]["inputs"]["images.image_3"]
    graph["35a"]["inputs"]["noise_seed"]=int(seed)
    graph["84"]["inputs"]["text"]=noun
    graph["86"]["inputs"]["name"]=noun
    graph["100"]["inputs"].update(max_shift=float(max_shift),size=size)
    if checkpoint_names:
        for key,idx,arg in (("sam3","20","ckpt_name"),("turbo","30","unet_name"),
                            ("clip","31","clip_name"),("vae","32","vae_name")):
            if key in checkpoint_names:
                graph[idx]["inputs"][arg]=checkpoint_names[key]
    return graph


# The loaders' node ids, the same in every SpEdE workflow that has them:
# node id -> (weights key, models/ subfolder, input naming the file).
_WEIGHT_LOADERS={
    "20":("sam3","checkpoints","ckpt_name"),
    "30":("diffusion","diffusion_models","unet_name"),
    "31":("clip","text_encoders","clip_name"),
    "32":("vae","vae","vae_name"),
    "61":("moge","geometry_estimation","model_name"),
}


def check_local_weights(comfy_root: Path, graph:dict, operation:str)->dict[str,str]:
    """The model files ``graph`` loads, found by the loader node ids present
    (a node with that id but without the file input, e.g. stage 2's
    sampler nodes, is not a loader). ``operation`` only matters for the hero
    graph, whose Qwen3-VL text encoder is node 12."""
    folder=Path(comfy_root).expanduser().resolve(strict=True)/"models"
    targets={}
    for node_id,(key,sub,arg) in _WEIGHT_LOADERS.items():
        inputs=graph.get(node_id,{}).get("inputs",{})
        if isinstance(inputs.get(arg),str):
            targets[key]=folder/sub/inputs[arg]
    if operation=="hero" and "12" in graph:
        targets["clip"]=folder/"text_encoders"/graph["12"]["inputs"]["clip_name"]
    missing={name:str(p) for name,p in targets.items() if not p.is_file()}
    if missing:
        raise FileNotFoundError(
            "Download/approve the required model weights yourself; missing: "+json.dumps(missing)
            +"; the pack's default files under models/ are: "+", ".join(sorted(MODELS.values()))
        )
    return {name:str(path) for name,path in targets.items()}


def needs_qwen(graph:dict)->bool:
    return any(n.get("class_type") in {"CLIPLoader","UNETLoader"} for n in graph.values())


def run_sdk(graph:dict,*,submit:bool=False,accept_qwen:bool=False,
            operation:str="patch",output_dir:Path|None=None)->dict:
    endpoint=validate_local_endpoint()
    # The hero check's naming loads the Qwen3-VL text encoder shipped with
    # Qwen-Image 2.1, so it asks for the same acknowledgement.
    if submit and needs_qwen(graph) and not accept_qwen:
        raise PermissionError(
            "Qwen-Image 2.1 (and its Qwen3-VL text encoder) is a Research License model. "
            "Read "+MODEL_LICENSE_URL+
            " and pass --accept-qwen-research-license only for permitted local use."
        )
    from comfy_sdk import Comfy
    client=Comfy()
    workflow=client.workflows.from_json(graph)
    if not submit:
        return {
            "mode":"dry_run", "sdk":"comfy-sdk v2",
            "endpoint":endpoint,"graph_nodes":len(graph),"submitted":False,
            "qwen_license_url":MODEL_LICENSE_URL if needs_qwen(graph) else None
        }
    job=client.run(workflow)
    exported=[]
    for index,asset in enumerate(job.outputs):
        if asset.type not in {"image","file"}:
            continue
        if output_dir is not None:
            outdir=Path(output_dir).resolve()
            outdir.mkdir(parents=True,exist_ok=True)
            safe_name=Path(asset.name).name
            destination=outdir/f"{index:02d}_{safe_name}"
            asset.to_file(destination)
            exported.append(str(destination))
    return {
        "mode":"completed",
        "job_id":str(job.id),
        "status":str(job.status),
        "output_assets":exported,
        "notice":"PLY may currently exist only in local Comfy output dir; file asset export from nodes is not yet a v2 output handle.",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation",choices=sorted(k for k in WORKFLOWS if k not in OWN_BUILDERS))
    ap.add_argument("--source-ply",type=Path,help="required for extract and patch")
    ap.add_argument("--camera-targets",type=Path,required=True)
    ap.add_argument("--hero-view",required=True)
    ap.add_argument("--hero-mask",type=Path,help="own hero-view mask PNG instead of SAM's")
    ap.add_argument("--object-text",help="required for extract and patch; for hero it skips Qwen3-VL naming")
    ap.add_argument("--points",help='hero: clicks as JSON [[x, y, 1], [x, y, 0], ...] (1 include, 0 exclude)')
    ap.add_argument("--job-name",default="local")
    ap.add_argument("--prompt",default=None)
    ap.add_argument("--seed",type=int,default=241007)
    ap.add_argument("--comfy-root",type=Path)
    ap.add_argument("--submit",action="store_true")
    ap.add_argument("--accept-qwen-research-license",action="store_true")
    ap.add_argument("--prompt-file",type=Path)
    ap.add_argument("--download-outputs",type=Path)
    args=ap.parse_args()
    if args.operation=="hero":
        if not args.points:
            ap.error("hero needs --points")
        graph=build_hero_graph(
            camera_targets=args.camera_targets,hero_view=args.hero_view,
            points=json.loads(args.points),object_text=args.object_text,job_name=args.job_name)
    else:
        if not args.source_ply or not args.object_text:
            ap.error(f"{args.operation} needs --source-ply and --object-text")
        graph=build_local_graph(
            args.operation,source_ply=args.source_ply,camera_targets=args.camera_targets,
            hero_view=args.hero_view,object_text=args.object_text,job_name=args.job_name,
            qwen_prompt=args.prompt,seed=args.seed,hero_mask=args.hero_mask
        )
    if args.comfy_root:
        check_local_weights(args.comfy_root,graph,args.operation)
    if args.prompt_file:
        args.prompt_file.parent.mkdir(parents=True,exist_ok=True)
        args.prompt_file.write_text(json.dumps(graph,indent=2)+"\n")
    print(json.dumps(run_sdk(
        graph,submit=args.submit,accept_qwen=args.accept_qwen_research_license,
        operation=args.operation,output_dir=args.download_outputs,
    ),indent=2))


if __name__=="__main__":
    main()
