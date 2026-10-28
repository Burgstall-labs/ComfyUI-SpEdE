"""The progressive all-around fill's workflow file, its SDK builder path and
the weight check. Pure CPU: no GPU, no ComfyUI."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

try:                                    # this repository
    from competition.nodes import NODE_CLASS_MAPPINGS
    from competition.local_sdk import MODELS, WORKFLOWS, build_allround_graphs, check_local_weights
except ImportError:                     # the exported node pack has these at its root
    from nodes import NODE_CLASS_MAPPINGS
    from local_sdk import MODELS, WORKFLOWS, build_allround_graphs, check_local_weights

ROOT=Path(__file__).resolve().parents[1]
# The pack's files: under competition/ here, at the root of the exported pack.
PACK=ROOT/"competition" if (ROOT/"competition"/"nodes.py").is_file() else ROOT
FOLDER=PACK/"workflows"
PROGRESSIVE=FOLDER/"spede_local_allround_progressive_api.json"
HERO=FOLDER/"spede_local_allround_hero_api.json"
NATIVE={"UNETLoader","CLIPLoader","VAELoader","QwenImage21Cache","LoadMoGeModel","PreviewImage",
        "TextEncodeQwenImage21","KSampler","VAEDecode","SaveImage"}


def links(graph):
    for node_id,node in graph.items():
        for key,value in node["inputs"].items():
            if isinstance(value,list) and len(value)==2:
                yield node_id,key,value


def assert_wired(graph):
    for node_id,node in graph.items():
        kind=node["class_type"]
        if kind.startswith("SpEdE_"):
            assert kind in NODE_CLASS_MAPPINGS,(node_id,kind)
            spec=NODE_CLASS_MAPPINGS[kind].INPUT_TYPES()
            assert set(spec["required"])<=set(node["inputs"])<=set(spec["required"])|set(spec.get("optional",{})),node_id
            for key,value in node["inputs"].items():
                kind_=(spec["required"].get(key) or spec.get("optional",{}).get(key))[0]
                if isinstance(kind_,list):
                    assert value in kind_,(node_id,key,value)
        else:
            assert kind in NATIVE,(node_id,kind)
    for node_id,key,(source,index) in links(graph):
        assert str(source) in graph,(node_id,key,source)
        if graph[str(source)]["class_type"] in NODE_CLASS_MAPPINGS:
            assert int(index)<len(NODE_CLASS_MAPPINGS[graph[str(source)]["class_type"]].RETURN_TYPES)


def test_progressive_workflow_is_wired_to_existing_nodes():
    graph=json.loads(PROGRESSIVE.read_text())
    assert WORKFLOWS["progressive"]==PROGRESSIVE.name
    assert_wired(graph)
    fill=graph["95"]
    assert fill["class_type"]=="SpEdE_LocalAllRoundProgressive"
    assert fill["inputs"]["hero_clean"]==["90",2]
    assert fill["inputs"]["method"]=="independent"
    # The fill reads only the plan and the hero clean: either stage 1 will do.
    assert graph["90"]["inputs"]["require"]=="hero_clean"
    assert NODE_CLASS_MAPPINGS["SpEdE_LocalAllRoundLoad"].RETURN_NAMES[2]=="hero_clean"
    # MoGe only for the progressive method: loaded lazily.
    spec=NODE_CLASS_MAPPINGS["SpEdE_LocalAllRoundProgressive"].INPUT_TYPES()
    assert spec["required"]["moge_model"][1]["lazy"] is True
    assert list(spec["optional"])==["device","method"]          # saved UI workflows fill widgets in order
    node=NODE_CLASS_MAPPINGS["SpEdE_LocalAllRoundProgressive"]()
    assert node.check_lazy_status(method="independent",moge_model=None)==[]
    assert node.check_lazy_status(method="progressive",moge_model=None)==["moge_model"]
    assert graph["96"]=={"class_type":"PreviewImage","inputs":{"images":["95",2]}}


def test_lean_stage1_is_the_full_stage1_up_to_the_hero_clean():
    hero=json.loads(HERO.read_text())
    full=json.loads((FOLDER/WORKFLOWS["allround1"]).read_text())
    assert WORKFLOWS["allround_hero"]==HERO.name
    assert_wired(hero)
    # The same plan and the same hero clean (nodes, inputs, seed) as the full stage 1.
    for node_id in ("70","30","31","32","33","34","35","36"):
        assert hero[node_id]==full[node_id],node_id
    assert hero["74"]=={"class_type":"SpEdE_LocalAllRoundHero","inputs":{"allround":["70",0],"hero_clean":["36",0]}}
    assert hero["80"]["inputs"]["images"]==["74",1]
    assert set(hero)=={"70","30","31","32","33","34","35","36","74","80"}
    manifest=json.loads((FOLDER/"manifest.json").read_text())
    assert manifest["files"][HERO.name]["output_node_ids"]==["80"]


def test_the_progressive_load_takes_a_lean_stage1_folder(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    folder=tmp_path/"job"
    folder.mkdir()
    (folder/"state.json").write_text(json.dumps({"resolution":1024,"frames":260}))
    load=NODE_CLASS_MAPPINGS["SpEdE_LocalAllRoundLoad"]()
    with pytest.raises(FileNotFoundError,match="hero_clean.png"):
        load.load(str(folder),require="hero_clean")
    Image.fromarray(np.full((8,8,3),200,np.uint8)).save(folder/"hero_clean.png")
    allround,warp,hero,frames,resolution=load.load(str(folder),require="hero_clean")
    assert allround.folder==folder and frames==0 and resolution==1024
    assert tuple(hero.shape)==(1,8,8,3) and abs(float(hero.max())-200/255)<1e-6
    # Stage 2 still needs the warp.
    with pytest.raises(FileNotFoundError,match="not a finished"):
        load.load(str(folder))


def test_manifest_lists_every_workflow_file():
    manifest=json.loads((FOLDER/"manifest.json").read_text())
    on_disk={p.name for p in FOLDER.glob("*_api.json")}
    assert set(manifest["files"])==on_disk
    assert len(on_disk)==10
    assert manifest["files"][PROGRESSIVE.name]["output_node_ids"]==["96"]
    assert set(WORKFLOWS.values())<=on_disk


@pytest.fixture
def run_inputs(tmp_path):
    run=tmp_path/"run"; run.mkdir()
    ply=tmp_path/"scene.ply"; ply.write_bytes(b"ply")
    cameras=tmp_path/"cameras.npz"; cameras.write_bytes(b"npz")
    return {"run_dir":run,"source_ply":ply,"camera_targets":cameras,"object_text":"bonsai"}


def string_inputs(graph):
    return [v for node in graph.values() for v in node["inputs"].values() if isinstance(v,str)]


def test_build_allround_graphs_progressive(run_inputs):
    one,two=build_allround_graphs(**run_inputs,job_name="bonsai01",seed=7,
                                  step_iterations=11,final_iterations=22)
    for graph in (one,two):
        assert not [v for v in string_inputs(graph) if "/JOB" in v]
    assert one["70"]["class_type"]=="SpEdE_LocalAllRoundStart"
    # The lean stage 1: plan and hero clean only.
    assert one["74"]["class_type"]=="SpEdE_LocalAllRoundHero" and "71" not in one and "73" not in one
    assert one["35"]["inputs"]["seed"]==7
    assert one["80"]["inputs"]["filename_prefix"]=="spede/allround/bonsai01/hero_clean"
    assert two["90"]["inputs"]["allround_dir"]=="spede/allround/bonsai01"
    fill=two["95"]["inputs"]
    assert fill["output_path"]=="spede/allround/bonsai01/progressive/progressive_scene.ply"
    assert (fill["seed"],fill["step_iterations"],fill["final_iterations"])==(7,11,22)
    defaults=build_allround_graphs(**run_inputs,job_name="bonsai01")[1]["95"]["inputs"]
    assert (defaults["step_iterations"],defaults["final_iterations"])==(1500,6000)
    assert defaults["method"]=="independent"
    assert build_allround_graphs(**run_inputs,method="progressive")[1]["95"]["inputs"]["method"]=="progressive"
    with pytest.raises(ValueError,match="method"):
        build_allround_graphs(**run_inputs,method="chain")


def test_build_allround_graphs_orbit_and_bad_second(run_inputs):
    one,two=build_allround_graphs(**run_inputs,job_name="k1",second="orbit",iterations=123)
    assert one["71"]["class_type"]=="SpEdE_LocalAllRoundQwenChain" and one["73"]["class_type"]=="SpEdE_LocalAllRoundWarp"
    assert two["91"]["class_type"]=="SpEdE_LocalAllRoundOptimize"
    assert two["91"]["inputs"]["iterations"]==123
    assert two["91"]["inputs"]["output_path"]=="spede/allround/k1/allround_scene.ply"
    assert "95" not in two
    with pytest.raises(ValueError,match="second"):
        build_allround_graphs(**run_inputs,second="video")


def test_check_local_weights_on_progressive_graph(tmp_path):
    graph=json.loads(PROGRESSIVE.read_text())
    models=tmp_path/"models"
    needed=[MODELS[k] for k in ("turbo","clip","vae","moge")]
    for rel in needed:
        (models/rel).parent.mkdir(parents=True,exist_ok=True)
        (models/rel).write_bytes(b"w")
    found=check_local_weights(tmp_path,graph,"progressive")
    assert set(found)=={"diffusion","clip","vae","moge"}
    assert found["diffusion"]==str((models/MODELS["turbo"]).resolve())
    (models/MODELS["turbo"]).unlink()
    with pytest.raises(FileNotFoundError,match="qwen_image_2.1_turbo_int8_convrot") as missing:
        check_local_weights(tmp_path,graph,"progressive")
    message=str(missing.value)
    assert '"diffusion"' in message.split("; the pack's default files")[0]
    # The error lists every default file name the pack expects.
    assert all(rel in message for rel in MODELS.values())


def test_check_local_weights_ignores_non_loader_ids(tmp_path):
    # Stage 2 has nodes 20/30/31/32 that are not loaders (H3 sampler nodes).
    graph=json.loads((FOLDER/WORKFLOWS["allround2"]).read_text())
    (tmp_path/"models").mkdir()
    assert check_local_weights(tmp_path,graph,"allround2")=={}


def test_no_development_hooks_or_prints():
    nodes=(PACK/"nodes.py").read_text()
    assert "importlib.reload(" not in nodes
    for path in (PACK/"nodes.py",ROOT/"spede_core/multiview_fill.py",
                 ROOT/"spede_core/progressive_fill.py",ROOT/"spede_core/surface_fill.py"):
        assert not re.search(r"\bprint\(",path.read_text()),path
        assert not re.search(r"\blog\s*(?::\s*Callable\s*)?=\s*print\b",path.read_text()),path
