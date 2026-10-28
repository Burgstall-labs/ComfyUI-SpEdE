"""CPU-only tests for the SpEdE UI (the Studio subset shipped in the node pack)."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
# The UI lives in competition/studio in the repository and in studio/ in the exported pack.
STUDIO = next(p for p in (ROOT / "competition" / "studio", ROOT / "studio") if p.is_dir())
PACK = STUDIO.parent
PRIVATE_IMPORT = re.compile(r"(?m)^\s*(?:from\s+(?:accgs|splatedit)\b|import\s+(?:accgs|splatedit)\b)")

SPEDE_ROUTES = {
    "/api/status", "/api/scenes", "/api/scenes/{scene_key}/cameras", "/api/scenes/{scene_key}/camera/{label}.png",
    "/api/scenes/{scene_key}/explore", "/api/projects", "/api/projects/{name}", "/api/projects/{name}/select",
    "/api/projects/{name}/label", "/api/projects/{name}/hero_choice", "/api/projects/{name}/hero",
    "/api/projects/{name}/pick", "/api/projects/{name}/extract", "/api/projects/{name}/remove",
    "/api/projects/{name}/explore", "/api/projects/{name}/extract/explore",
    "/api/projects/{name}/replace", "/api/projects/{name}/replace_image",
    "/api/jobs", "/api/jobs/{job_id}", "/api/jobs/{job_id}/cancel", "/files/{path:path}",
    "/", "/explore.html", "/compare.html",
}
STUDIO_ONLY = ("poi", "leftover", "propose", "generate", "adjust", "undo", "preview", "describe", "/api/runs",
               "/api/projects/{name}/patch")


def _write_scene(folder: Path, views: int = 2):
    """A three-Gaussian scene and a camera pack (two cameras unless asked),
    enough for the CPU paths."""
    folder.mkdir(parents=True, exist_ok=True)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.zeros(3, dtype=np.dtype([(name, "<f4") for name in props]))
    rows["x"] = [0, 1, 0]; rows["y"] = [0, 0, 1]; rows["z"] = 3
    rows["opacity"] = [4.2, 4.2, -30.0]           # the third is invisible
    for key in ("scale_0", "scale_1", "scale_2"):
        rows[key] = np.log(0.03)
    rows["rot_0"] = 1
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(folder / "point_cloud.ply"))
    labels = ["cam_a", "cam_b", "cam_c", "cam_d", "cam_e", "cam_f"][:views]
    np.savez(folder / "cameras.npz",
             targets=np.full((views, 32, 48, 3), 90, np.uint8), labels=np.array(labels),
             viewmats=np.repeat(np.eye(4)[None], views, axis=0),
             intrinsics=np.repeat(np.array([[40.0, 0, 24], [0, 40.0, 16], [0, 0, 1]])[None], views, axis=0),
             frame_indices=np.arange(views))


def _write_config(tmp_path, *, views: int = 2, scene: dict | None = None) -> Path:
    scenes = tmp_path / "scenes"
    _write_scene(scenes / "toy", views=views)
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / "output").mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "comfy_host": "127.0.0.1", "comfy_port": 1, "comfy_sdk_url": "http://127.0.0.1:2",
        "comfy_root": str(comfy_root), "runs": str(tmp_path / "runs"), "gpu_turns": False,
        "scene_root": str(scenes),
        "scenes": {"toy": scene if scene is not None else {
            "title": "Toy", "source_ply": "toy/point_cloud.ply", "cameras": "toy/cameras.npz", "hero": "cam_a"}},
    }))
    return config


def _load_server(tmp_path, monkeypatch, views: int = 2):
    pytest.importorskip("fastapi")
    config = _write_config(tmp_path, views=views)
    monkeypatch.setenv("SPEDE_CONFIG", str(config))
    for name in ("spede_ui_server_under_test", "comfy_backend", "gpu_turn"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("spede_ui_server_under_test", STUDIO / "server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ui_sources_are_free_of_private_code_and_studio_only_features():
    for path in [*STUDIO.glob("*.py"), PACK / "geometry_cli.py"]:
        body = path.read_text()
        assert not PRIVATE_IMPORT.search(body), path
        compile(body, str(path), "exec")
    pages = "\n".join((STUDIO / name).read_text() for name in ("index.html", "explore.html", "compare.html"))
    assert "Splatedit Studio" not in pages
    assert "<title>SpEdE — Splat Edit Engine</title>" in (STUDIO / "index.html").read_text()
    # Replace (beta) is in the UI since 27 Oct 2026; Studio's other tools are not.
    for fragment in (r"\bPOIs?\b", "/pois", "leftover-", "mode=leftover", "propose", "Klein", "/undo", "/adjust"):
        assert not re.search(fragment, pages), fragment


def test_ui_server_exposes_exactly_the_spede_routes(tmp_path, monkeypatch):
    private = lambda: {k for k in sys.modules if k.startswith(("accgs", "splatedit"))}
    before = private()                             # other tests in this process may have loaded them
    server = _load_server(tmp_path, monkeypatch)
    paths = {route.path for route in server.app.routes if hasattr(route, "path")}
    paths.discard("/viewer")                       # the static mount
    assert SPEDE_ROUTES <= paths, SPEDE_ROUTES - paths
    for fragment in STUDIO_ONLY:
        assert not any(fragment in p for p in paths), fragment
    assert private() == before, "importing the UI loaded private code"


def test_ui_api_cpu_paths_without_comfy(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    server = _load_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    status = client.get("/api/status").json()
    assert status["comfy"] is False and status["sdk"] is False and status["scenes"] == 1
    scenes = client.get("/api/scenes").json()
    assert scenes["toy"]["size"] == [48, 32] and scenes["toy"]["heldout"] is False
    cameras = client.get("/api/scenes/toy/cameras").json()
    assert [c["id"] for c in cameras] == ["cam_a", "cam_b"] and cameras[0]["hero"] is True
    png = client.get("/api/scenes/toy/camera/cam_b.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert client.get("/api/scenes/toy/camera/nope.png").status_code == 404

    assert client.post("/api/projects", json={"scene": "nope"}).status_code == 400
    project = client.post("/api/projects", json={"scene": "toy", "title": "Toy edit"}).json()
    name = project["name"]
    assert name.startswith("spede_toy_toy_edit_") and project["view"] == "cam_a" and project["stage"] == "select"
    assert [p["name"] for p in client.get("/api/projects").json()] == [name]
    detail = client.get(f"/api/projects/{name}").json()
    assert detail["files"] == {} and detail["edited"] is False and detail["job"] is None

    assert client.post(f"/api/projects/{name}/select", json={"points": []}).status_code == 400
    assert client.post(f"/api/projects/{name}/select", json={"points": [[3, 4, 0]]}).status_code == 400
    # A real selection needs ComfyUI: refused up front, nothing queued.
    refused = client.post(f"/api/projects/{name}/select", json={"points": [[3, 4, 1]], "label": "toy block"})
    assert refused.status_code == 503 and "ComfyUI" in refused.json()["detail"]
    assert client.get("/api/jobs").json() == []
    assert client.post(f"/api/projects/{name}/extract").status_code == 400          # select first
    assert client.post(f"/api/projects/{name}/remove").status_code == 400
    assert client.post(f"/api/projects/{name}/hero_choice", json={"use": "name"}).status_code == 400
    assert client.post(f"/api/projects/{name}/hero", json={"view": "cam_b"}).status_code == 400
    assert client.get(f"/api/projects/{name}/explore").status_code == 400          # nothing edited yet
    assert client.get(f"/api/projects/{name}/extract/explore").status_code == 404
    # Without clicks on record a name is stored directly.
    named = client.post(f"/api/projects/{name}/label", json={"label": "  toy   block "}).json()
    assert named["label"] == "toy block" and named["label_source"] == "user"
    assert client.post(f"/api/projects/{name}/label", json={"label": " "}).status_code == 400

    explore = client.get("/api/scenes/toy/explore?view=cam_b").json()
    assert explore["camera"]["view"] == "cam_b" and explore["splat"].startswith("/viewer/_scenes/toy/original.splat")
    splat = client.get(explore["splat"].split("?")[0])
    assert splat.status_code == 200 and len(splat.content) == 2 * 32      # the invisible Gaussian is dropped
    assert client.get("/files/../config.json").status_code == 404
    assert client.get("/").status_code == 200 and "SpEdE" in client.get("/").text


def test_extract_explore_orbits_the_object_centre(tmp_path, monkeypatch):
    """The extracted object's 3D view pivots on the object (median of its visible
    Gaussians), and a far camera moves toward it along its line of sight."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import shutil

    server = _load_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    name = client.post("/api/projects", json={"scene": "toy", "title": "Toy edit"}).json()["name"]
    (tmp_path / "runs" / name / "extract").mkdir(parents=True)
    shutil.copy(tmp_path / "scenes/toy/point_cloud.ply", tmp_path / "runs" / name / "extract/object_only.ply")
    info = client.get(f"/api/projects/{name}/extract/explore?view=cam_a").json()
    centre = np.array([0.5, 0.0, 3.0])                       # the invisible third Gaussian is ignored
    assert np.allclose(info["camera"]["lookAt"], centre) and np.allclose(info["object"]["centre"], centre)
    assert info["object"]["radius"] == pytest.approx(0.5)
    # The real camera (origin) is farther than the fit (diameter = half the frame height): moved toward the object.
    position = np.array(info["camera"]["position"])
    fit = 2 * 0.5 / (16 / 40)
    assert np.linalg.norm(position - centre) == pytest.approx(fit)
    assert np.allclose(np.cross(position - centre, -centre), 0, atol=1e-9)   # on the line to the real camera
    # The scene view keeps the real camera and looks straight ahead.
    scene = client.get("/api/scenes/toy/explore?view=cam_a").json()
    assert scene["camera"]["position"] == [0, 0, 0] and "object" not in scene


def test_ply_to_splat_drops_invisible_gaussians(tmp_path):
    from spede_core.splat_export import ply_to_splat

    _write_scene(tmp_path)
    kept = ply_to_splat(tmp_path / "point_cloud.ply", tmp_path / "out" / "scene.splat")
    assert kept == 2
    data = np.fromfile(tmp_path / "out" / "scene.splat",
                       dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    assert len(data) == 2 and (data["c"][:, 3] > 250).all()
    np.testing.assert_allclose(data["s"], 0.03, rtol=1e-5)


def test_geometry_cli_parses_both_commands():
    spec = importlib.util.spec_from_file_location("spede_geometry_cli_under_test", PACK / "geometry_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    rank = parser.parse_args(["rank-views", "--source-ply", "s.ply", "--camera-targets", "c.npz",
                              "--hero-view-id", "cam_a", "--hero-mask", "m.png"])
    assert rank.func is module.rank_views_command and rank.device == "cuda"
    pick = parser.parse_args(["pick-camera", "--source-ply", "s.ply", "--camera-targets", "c.npz",
                              "--pose-json", "p.json", "--render-out", "v.png", "--device", "cpu"])
    assert pick.func is module.pick_camera_command and pick.render_out == "v.png"
    with pytest.raises(SystemExit):
        parser.parse_args(["rank-views", "--source-ply", "s.ply"])
    colmap = parser.parse_args(["import-colmap", "--sparse-dir", "sparse/0", "--images-dir", "images_4",
                                "--out-dir", "cams", "--hero", "DSCF5565.JPG"])
    assert colmap.func is module.import_colmap_command and colmap.size == 512 and colmap.test_every == 8


def test_colmap_import_writes_square_camera_packs_and_keeps_the_hero(tmp_path):
    from PIL import Image

    from spede_core.colmap_import import ColmapView, split_heldout, square_crop, write_camera_pack

    # Sixteen landscape photos with a gradient, a pinhole whose principal point is off-centre.
    views = []
    for index in range(16):
        path = tmp_path / f"IMG_{index:02d}.JPG"
        image = np.zeros((60, 100, 3), np.uint8)
        image[..., 0] = np.linspace(0, 255, 100, dtype=np.uint8)[None, :]
        image[..., 1] = index * 10
        Image.fromarray(image).save(path)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[2, 3] = index
        views.append(ColmapView(path.name, viewmat, np.array([[80, 0, 52], [0, 80, 31], [0, 0, 1]], np.float32), path))
    crop, k = square_crop(views[0], 30)
    assert crop.shape == (30, 30, 3)
    # Centre crop of 100x60 → 60x60 from x0=20: cx 52-20=32, then scaled by 30/60.
    np.testing.assert_allclose(k[0, 2], 16.0)
    np.testing.assert_allclose(k[1, 2], 15.5)
    np.testing.assert_allclose(k[0, 0], 40.0)

    edit, heldout = split_heldout(views, every=8, heldout_count=8, edit_count=6, hero="IMG_08.JPG")
    assert edit[0].name == "IMG_08.JPG" and len(edit) == 6
    assert {v.name for v in heldout} == {"IMG_00.JPG"}          # IMG_08 is the hero, never held out
    assert not ({v.name for v in edit} & {v.name for v in heldout})
    with pytest.raises(ValueError, match="hero"):
        split_heldout(views, hero="nope.JPG")

    pack = write_camera_pack(edit, tmp_path / "cams" / "edit_cameras.npz", size=30)
    with np.load(pack) as data:
        assert data["targets"].shape == (6, 30, 30, 3) and data["targets"].dtype == np.uint8
        assert list(data["labels"]) == [Path(v.name).stem for v in edit]
        assert data["viewmats"].shape == (6, 4, 4) and data["intrinsics"].shape == (6, 3, 3)
        assert list(data["frame_indices"]) == list(range(6)) and data["is_source"].all()
        assert int(data["face_size"]) == 30 and data["rgb_weights"].shape == (6, 30, 30)
    # The pack is what the rest of the engine reads.
    from spede_assets import inspect_camera_pack
    assert inspect_camera_pack(pack)["view_ids"][0] == "IMG_08"


def test_hero_check_compares_name_masks_of_another_resolution():
    from spede_core.hero_check import compare_hero_masks

    click = np.zeros((60, 80), bool)
    click[10:40, 10:40] = True
    small = np.zeros((30, 40), bool)                 # the same box at half resolution
    small[5:20, 5:20] = True
    chosen, named, report = compare_hero_masks(click, [small], [[20, 20]])
    assert report["status"] == "agree" and report["passed"] is True
    assert report["resized_name_masks"] == 1 and report["iou"] > 0.9
    assert named.shape == click.shape


def test_export_bundles_the_ui_and_never_a_machine_config(tmp_path):
    exporter = pytest.importorskip("scripts.export_local_competition")
    targets = {item["target"] for item in exporter.manifest()["files"]}
    for required in ("studio/server.py", "studio/comfy_backend.py", "studio/gpu_turn.py", "studio/index.html",
                     "studio/explore.html", "studio/compare.html", "studio/config.example.json",
                     "studio/requirements.txt", "geometry_cli.py", "tests/test_competition_cpu.py",
                     "tests/test_spede_ui_cpu.py"):
        assert required in targets, required
    assert "studio/config.json" not in targets
    bundle = tmp_path / "bundle"
    exporter.build(bundle, private_dev=True)
    assert (bundle / "studio" / "server.py").exists() and not (bundle / "studio" / "config.json").exists()
    assert "from competition" not in (bundle / "studio" / "server.py").read_text()


def _named_project(client, server, patch_mode: str | None, label: str | None = "toy block") -> tuple[str, Path]:
    name = client.post("/api/projects", json={"scene": "toy", "title": "Toy edit"}).json()["name"]
    if label:
        client.post(f"/api/projects/{name}/label", json={"label": label})
    root = server.RUNS / name
    if patch_mode:
        (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": patch_mode, "comfy_output": str(root)}))
    return name, root


def test_ui_allround_route_refusals_and_remove_state(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    server = _load_server(tmp_path, monkeypatch)
    assert server.PATCH_MODE == "depthfill" and server.ALLROUND_MODE == "independent"
    assert {"allround", "comfy_allround.json", "comfy_allround.log"} <= set(server.REMOVE_STATE)
    assert server.PROJECT_FILES["allround_ply"] == "allround/allround_scene.ply"
    assert server.PROJECT_FILES["allround_sheet"] == "allround/progressive_sheet.png"
    assert server.PROJECT_FILES["allround_hero_clean"] == "allround/hero_clean.png"
    client = TestClient(server.app)
    name, _root = _named_project(client, server, "patch")
    planar = client.post(f"/api/projects/{name}/allround")
    assert planar.status_code == 400
    assert planar.json()["detail"] == ("the all-around fill builds on the depth fill (patch_mode depthfill); "
                                       "this result is a planar patch")
    unnamed, _root = _named_project(client, server, "depthfill", label=None)
    refused = client.post(f"/api/projects/{unnamed}/allround")
    assert refused.status_code == 400 and refused.json()["detail"] == "name the selection first"
    assert client.get("/api/jobs").json() == []
    # Remove again sets the all-around result aside and drops its hero-only baseline,
    # but only once the job is accepted: a 503 refusal leaves the previous result intact.
    monkeypatch.setattr(server, "QWEN_ACK", True)
    again, root = _named_project(client, server, "depthfill")
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs/hero_mask.png").write_bytes(b"mask")
    (root / "session_manifest.json").write_text("{}")
    (root / "allround").mkdir()
    (root / "comfy_allround.json").write_text("{}")
    (server.VIEWER / again).mkdir(parents=True)
    (server.VIEWER / again / "hero_only.splat").write_bytes(b"old")
    assert client.post(f"/api/projects/{again}/remove").status_code == 503
    assert (root / "allround").is_dir() and (root / "comfy_allround.json").exists()
    assert (root / "session_manifest.json").exists()
    assert (server.VIEWER / again / "hero_only.splat").read_bytes() == b"old"
    monkeypatch.setattr(server, "_comfy_alive", lambda: True)
    monkeypatch.setattr(server, "_sdk_alive", lambda: True)
    queued = client.post(f"/api/projects/{again}/remove")
    assert queued.status_code == 200
    server.QUEUE.remove(queued.json()["id"])         # no worker thread in tests
    assert not (root / "allround").exists() and not (root / "comfy_allround.json").exists()
    assert len(list(root.glob("_previous_*/comfy_allround.json"))) == 1
    assert not (server.VIEWER / again / "hero_only.splat").exists()
    assert client.get(f"/api/projects/{again}").json()["allround"] is None
    for page in ("/", "/explore.html", "/compare.html"):
        assert client.get(page).status_code == 200, page
        assert client.get(page).headers["cache-control"] == "no-cache", page


def test_ui_allround_mode_is_validated_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEDE_ALLROUND_MODE", "bogus")
    with pytest.raises(SystemExit, match="allround_mode must be independent"):
        _load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("SPEDE_ALLROUND_MODE", "orbit")
    assert _load_server(tmp_path / "again", monkeypatch).ALLROUND_MODE == "orbit"
    monkeypatch.setenv("SPEDE_ALLROUND_MODE", "progressive")
    assert _load_server(tmp_path / "third", monkeypatch).ALLROUND_MODE == "progressive"


def _progressive_report() -> dict:
    return {"order": ["cam_a", "cam_b", "cam_c"], "surface": {"discs": 4200},
            "steps": [{"view": "cam_b", "geometry": {"born": 1200}}, {"view": "cam_c", "geometry": {"born": 300}},
                      {"view": "cam_x", "geometry": {}}],
            "final": {"trainable": 9000, "pruned": 17, "total_gaussians": 123456, "seconds": 610.5, "held_out": [],
                      "after": {"cam_a": {"mae": 0.021, "texture": 0.98}, "cam_b": {"mae": 0.034, "texture": 0.9},
                                "cam_c": {"mae": 0.05, "texture": 0.8}}}}


def test_progressive_summary_fields():
    sys.path.insert(0, str(STUDIO))
    import comfy_backend

    common = {"backend": "comfy", "comfy_job": "job2", "stage1_job": "job1", "job_name": "toy_allround_1",
              "comfy_output": "/out", "total_gaussians": 99, "allround_ply": "/runs/p/allround/allround_scene.ply",
              "pictures": {"progressive_sheet": "progressive_sheet.png"}, "finished": 1.0}
    summary = comfy_backend.progressive_summary(_progressive_report(), **common)
    assert summary["method"] == "progressive"
    assert summary["cameras"] == ["cam_a", "cam_b", "cam_c"] and summary["cameras_filled"] == 3
    assert summary["surface_discs"] == 4200 and summary["births"] == 1500
    assert (summary["trainable"], summary["pruned"], summary["total_gaussians"], summary["seconds"]) == (9000, 17, 123456, 610.5)
    assert summary["after"]["cam_b"] == {"mae": 0.034, "texture": 0.9}
    for key, value in common.items():
        if key != "total_gaussians":
            assert summary[key] == value, key
    for orbit_only in ("usable_frames", "orbit_frames", "held_out", "pruned_drifted"):
        assert orbit_only not in summary
    # Without a report count the PLY's own count is used.
    report = _progressive_report()
    del report["final"]["total_gaussians"]
    report["surface"] = None
    fallback = comfy_backend.progressive_summary(report, **common)
    assert fallback["total_gaussians"] == 99 and fallback["surface_discs"] is None
    independent = comfy_backend.progressive_summary({**_progressive_report(), "method": "independent"}, **common)
    assert independent["method"] == "independent"
    orbit = comfy_backend.orbit_summary({"chain": ["cam_a", "cam_b"], "held_out": ["cam_b"], "usable_frames": 200,
                                         "orbit_frames": 260, "pruned_drifted": 3, "after": {}}, **common)
    assert orbit["method"] == "orbit" and orbit["cameras_filled"] == 2 and orbit["usable_frames"] == 200


def test_allround_step_progressive_paths_and_hero_only_baseline(tmp_path, monkeypatch):
    sys.path.insert(0, str(STUDIO))
    import comfy_backend

    _write_scene(tmp_path / "scene")
    ply = tmp_path / "scene" / "point_cloud.ply"
    root, comfy_output, viewer = tmp_path / "runs/p", tmp_path / "output", tmp_path / "viewer"
    root.mkdir(parents=True)
    (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": "depthfill", "comfy_output": str(tmp_path)}))
    (root / "session_manifest.json").write_text(json.dumps({"generation": 1, "current_ply": str(ply)}))
    (viewer / "p").mkdir(parents=True)
    (viewer / "p" / "edited.splat").write_bytes(b"hero-only")
    seen = {}

    def fake_graphs(**kwargs):
        seen.update(kwargs)
        return {"stage": 1}, {"stage": 2}

    class Job:
        def __init__(self, id_):
            self.id = id_

    def fake_run(graph, log, timeout):
        out = comfy_output / "spede/allround" / seen["job_name"]
        if graph["stage"] == 1:
            out.mkdir(parents=True, exist_ok=True)
            (out / "hero_clean.png").write_bytes(b"png1")
            return Job("stage1-id")
        (out / "progressive").mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(ply, out / "progressive/progressive_scene.ply")
        (out / "progressive/progressive_scene.progressive_report.json").write_text(json.dumps(_progressive_report()))
        (out / "progressive/progressive_sheet.png").write_bytes(b"png2")
        return Job("stage2-id")

    monkeypatch.setattr(comfy_backend, "build_allround_graphs", fake_graphs)
    monkeypatch.setattr(comfy_backend, "run_graph", fake_run)
    step = comfy_backend.allround_step(project={"name": "p", "label": "toy block"},
                                       scene={"source_ply": ply, "cameras": tmp_path / "scene/cameras.npz"},
                                       root=root, comfy_output=comfy_output, viewer=viewer)
    step()
    assert seen["second"] == "progressive" and seen["method"] == "independent"
    assert (root / "allround/allround_scene.ply").read_bytes() == ply.read_bytes()
    assert (root / "allround/progressive_sheet.png").read_bytes() == b"png2"
    assert (root / "allround/hero_clean.png").read_bytes() == b"png1"
    summary = json.loads((root / "comfy_allround.json").read_text())
    assert summary["method"] == "progressive" and summary["comfy_job"] == "stage2-id"
    assert summary["stage1_job"] == "stage1-id" and summary["births"] == 1500
    assert summary["allround_ply"] == str(root / "allround/allround_scene.ply")
    assert summary["pictures"] == {"progressive_sheet": "progressive_sheet.png", "hero_clean": "hero_clean.png"}
    manifest = json.loads((root / "session_manifest.json").read_text())
    assert manifest["allround"] is True and manifest["current_ply"] == summary["allround_ply"]
    assert (viewer / "p/hero_only.splat").read_bytes() == b"hero-only"
    assert (viewer / "p/edited.splat").read_bytes() != b"hero-only"
    # A second run keeps the hero-only baseline instead of copying the last all-around result over it.
    step()
    assert (viewer / "p/hero_only.splat").read_bytes() == b"hero-only"
    assert json.loads((root / "session_manifest.json").read_text())["generation"] == 3


def test_ui_status_reports_the_qwen_licence_acknowledgement(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("SPEDE_ACCEPT_QWEN_RESEARCH_LICENSE", raising=False)
    server = _load_server(tmp_path, monkeypatch)
    assert server.QWEN_ACK is False and server.comfy_backend.QWEN_ACK is False
    assert TestClient(server.app).get("/api/status").json()["qwen_research_license_acknowledged"] is False
    monkeypatch.setenv("SPEDE_ACCEPT_QWEN_RESEARCH_LICENSE", "1")
    server = _load_server(tmp_path / "again", monkeypatch)
    assert server.QWEN_ACK is True and server.comfy_backend.QWEN_ACK is True
    assert TestClient(server.app).get("/api/status").json()["qwen_research_license_acknowledged"] is True
    example = json.loads((STUDIO / "config.example.json").read_text())
    assert example["accept_qwen_research_license"] is False
    assert "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE" in example["accept_qwen_research_license_comment"]


def _fake_comfy_sdk(monkeypatch) -> list:
    """A stand-in comfy_sdk that records what was submitted and succeeds."""
    import types

    submitted = []

    class Job:
        id, status, error = "fake-job", "succeeded", None

        def wait(self, timeout=None):
            return self

    class Comfy:
        def __init__(self, **kwargs):
            self.workflows = types.SimpleNamespace(from_json=lambda graph: graph)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, workflow):
            submitted.append(workflow)
            return Job()

    monkeypatch.setitem(sys.modules, "comfy_sdk", types.SimpleNamespace(Comfy=Comfy))
    return submitted


def test_ui_remove_is_refused_without_the_qwen_licence(tmp_path, monkeypatch):
    """Remove (Qwen-Image 2.1) is refused with the licence text before it
    queues or reaches ComfyUI; a Qwen job that gets queued anyway fails in
    run_graph with the licence text once; Extract and a hero check with a
    typed name still submit."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from PIL import Image

    monkeypatch.delenv("SPEDE_ACCEPT_QWEN_RESEARCH_LICENSE", raising=False)
    server = _load_server(tmp_path, monkeypatch, views=4)
    backend = server.comfy_backend
    submitted = _fake_comfy_sdk(monkeypatch)
    monkeypatch.setattr(server, "_comfy_alive", lambda: True)
    monkeypatch.setattr(server, "_sdk_alive", lambda: True)
    client = TestClient(server.app)
    name, root = _named_project(client, server, None)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    Image.new("L", (48, 32), 255).save(root / "inputs/hero_mask.png")

    (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": "depthfill", "comfy_output": str(root)}))
    for refused in (client.post(f"/api/projects/{name}/remove"),
                    client.post(f"/api/projects/{name}/allround"),
                    client.post(f"/api/projects/{name}/select", json={"points": [[3, 4, 1]]})):
        assert refused.status_code == 400
        assert refused.json()["detail"] == backend.QWEN_LICENSE_TEXT
    # A removed project's selection is fixed (pick and hero are refused for that); the licence checks below
    # are about a project that is not edited yet.
    (root / "comfy_patch.json").unlink()
    # Pick always names the new mask with Qwen3-VL; another hero camera does when the project is unnamed.
    pick = {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "fov": 50, "aspect": 1.5, "x": 0.5, "y": 0.5}
    unnamed = client.post("/api/projects", json={"scene": "toy", "title": "Unnamed edit"}).json()["name"]
    unnamed_root = server.RUNS / unnamed
    assert unnamed != name and not client.get(f"/api/projects/{unnamed}").json().get("label")
    for project_root in (root, unnamed_root):
        (project_root / "view_rank.json").write_text(json.dumps(
            {"views": [{"view_id": "cam_b", "points": [[3, 4, 1]]}]}))
    for refused in (client.post(f"/api/projects/{name}/pick", json=pick),
                    client.post(f"/api/projects/{unnamed}/hero", json={"view": "cam_b"})):
        assert refused.status_code == 400
        assert refused.json()["detail"] == backend.QWEN_LICENSE_TEXT
    assert not (root / "inputs/pick_pose.json").exists()
    assert client.get("/api/jobs").json() == [] and submitted == []
    # A named project's hero change runs SAM 3.1 only: accepted.
    named_hero = client.post(f"/api/projects/{name}/hero", json={"view": "cam_b"})
    assert named_hero.status_code == 200, named_hero.json()
    server.QUEUE.remove(named_hero.json()["id"])
    server.JOBS.pop(named_hero.json()["id"])

    # The enforcement point: a Qwen graph queued anyway fails with the licence text, logged once.
    scene = server.SCENES["toy"]
    common = dict(source_ply=scene["source_ply"], camera_targets=scene["cameras"], hero_view="cam_a",
                  object_text="toy block", hero_mask=root / "inputs/hero_mask.png")
    depthfill = backend.build_local_graph("depthfill", **common)
    queued = server._job("remove", name, [("comfy_patch", lambda: backend.run_graph(
        depthfill, root / "comfy_patch.log"), None)])
    job = server.JOBS[queued["id"]]
    server.QUEUE.remove(job["id"])                 # no worker thread in tests: run it here
    server._run_job(job)
    assert job["status"] == "failed" and submitted == []
    for text in ("Research License", backend.MODEL_LICENSE_URL, "accept_qwen_research_license"):
        assert text in job["error"], text
    assert job["error"].count("Research License") == 1
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "failed"

    # A typed name skips Qwen3-VL naming: select is accepted.
    named_select = client.post(f"/api/projects/{name}/select", json={"points": [[3, 4, 1]], "label": "toy block"})
    assert named_select.status_code == 200
    server.QUEUE.remove(named_select.json()["id"])

    # The gate itself: the Qwen graphs are refused, Extract and the named hero check pass.
    extract = backend.build_local_graph("extract", **common)
    unnamed = backend.build_hero_graph(camera_targets=scene["cameras"], hero_view="cam_a", points=[[3, 4, 1]])
    named = backend.build_hero_graph(camera_targets=scene["cameras"], hero_view="cam_a", points=[[3, 4, 1]],
                                     object_text="toy block")
    for graph in (depthfill, unnamed):
        with pytest.raises(backend.StepFailed, match="accept_qwen_research_license"):
            backend.run_graph(graph, tmp_path / "gate.log")
    assert submitted == []
    for graph in (extract, named):
        backend.run_graph(graph, tmp_path / "gate.log")
    assert submitted == [extract, named]
    # With the acknowledgement the Qwen graph is submitted too.
    monkeypatch.setattr(backend, "QWEN_ACK", True)
    backend.run_graph(depthfill, tmp_path / "gate.log")
    assert submitted[-1] is depthfill


SPILL_NODE_ERROR = ("ComfyUI job 17fc failed: \"code='node_execution_error' message='Qwen-Image removed more than the "
                    "selection: it changed 7566 px outside the 3434-px selection (220% of it; magenta in edit_spill.png). "
                    "The 3D scene would keep what the picture lost.\\\\n' node_id='63' class_type='SpEdE_LocalAlignMoGeDepth'\"")


def test_spill_message_explains_the_qwen_spill_gate():
    sys.path.insert(0, str(STUDIO))
    import comfy_backend

    assert comfy_backend.spill_message("ComfyUI job x failed: out of memory") is None
    text = comfy_backend.spill_message(SPILL_NODE_ERROR)
    assert text.startswith("Qwen removed more than the selection: 7,566 px beyond the 3,434-px selection")
    assert "use your clicks or add include points" in text and "Remove anyway" in text
    # The node before the selection size was in its message: estimated from the percentage.
    old = comfy_backend.spill_message("Qwen-Image removed more than the selection: it changed 7566 px outside it "
                                      "(220% of the selection; magenta in edit_spill.png).")
    assert "7,566 px beyond the 3,439-px selection" in old


def test_remove_seed_spill_override_and_failed_pictures(tmp_path, monkeypatch):
    sys.path.insert(0, str(STUDIO))
    import comfy_backend

    _write_scene(tmp_path / "scene", views=4)
    scene = {"source_ply": tmp_path / "scene/point_cloud.ply", "cameras": tmp_path / "scene/cameras.npz"}
    root, comfy_output = tmp_path / "runs/p", tmp_path / "output"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/hero_mask.png").write_bytes(b"mask")
    (root / "comfy_failed").mkdir()
    (root / "comfy_failed/stale.png").write_bytes(b"old")
    graphs = []

    def fake_run(graph, log, **kwargs):
        graphs.append(graph)
        out = comfy_output / "spede/local" / graph["42"]["inputs"]["filename_prefix"].split("/")[2]
        (out / "depth").mkdir(parents=True, exist_ok=True)
        (out / "depth/edit_spill.png").write_bytes(b"spill")
        (out / "qwen21_edit_00001_.png").write_bytes(b"edit")
        raise comfy_backend.StepFailed(SPILL_NODE_ERROR)

    monkeypatch.setattr(comfy_backend, "run_graph", fake_run)
    step = comfy_backend.remove_patch_step(project={"name": "p", "label": "toy block", "view": "cam_a"}, scene=scene,
                                           root=root, comfy_output=comfy_output, viewer=tmp_path / "viewer",
                                           mode="depthfill", seed=12345, ignore_spill=True)
    with pytest.raises(comfy_backend.StepFailed, match="7,566 px beyond the 3,434-px selection"):
        step()
    assert graphs[0]["35"]["inputs"]["seed"] == 12345
    assert graphs[0]["63"]["inputs"]["max_spill_fraction"] == comfy_backend.SPILL_FRACTION_ANYWAY
    assert sorted(p.name for p in (root / "comfy_failed").iterdir()) == ["edit_spill.png", "qwen_edit.png"]
    assert (root / "comfy_failed/qwen_edit.png").read_bytes() == b"edit"
    assert "SpEdE: Qwen removed more than the selection" in (root / "comfy_patch.log").read_text()
    # Without a seed or the override the workflow's own values stay.
    with pytest.raises(comfy_backend.StepFailed):
        comfy_backend.remove_patch_step(project={"name": "p", "label": "toy block", "view": "cam_a"}, scene=scene,
                                        root=root, comfy_output=comfy_output, viewer=tmp_path / "viewer",
                                        mode="depthfill")()
    assert graphs[1]["35"]["inputs"]["seed"] == 241007
    assert "max_spill_fraction" not in graphs[1]["63"]["inputs"]


def test_ui_remove_request_error_text_and_pick_miss(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    server = _load_server(tmp_path, monkeypatch, views=4)
    assert server.PROJECT_FILES["edit_spill"] == "comfy_failed/edit_spill.png"
    assert server.PROJECT_FILES["failed_qwen_edit"] == "comfy_failed/qwen_edit.png"
    # A step's own explanation leads the error, before a node's refusal in the same log.
    log = "12:00:00 submitted\n" + repr(RuntimeError(SPILL_NODE_ERROR)) + "\nSpEdE: Qwen removed more, 7,566 px\n"
    assert server._error_text(log).split("\n\n")[0] == "Qwen removed more, 7,566 px"
    monkeypatch.setattr(server, "QWEN_ACK", True)
    monkeypatch.setattr(server, "_comfy_alive", lambda: True)
    monkeypatch.setattr(server, "_sdk_alive", lambda: True)
    client = TestClient(server.app)
    name, root = _named_project(client, server, None)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs/hero_mask.png").write_bytes(b"mask")
    seen = {}
    monkeypatch.setattr(server.comfy_backend, "remove_patch_step", lambda **kw: seen.update(kw) or (lambda: None))
    queued = client.post(f"/api/projects/{name}/remove", json={"seed": 77, "ignore_spill": True})
    assert queued.status_code == 200, queued.json()
    assert queued.json()["request"] == {"seed": 77, "ignore_spill": True}
    assert seen["seed"] == 77 and seen["ignore_spill"] is True
    server.QUEUE.remove(queued.json()["id"])
    server.JOBS.pop(queued.json()["id"])
    plain = client.post(f"/api/projects/{name}/remove")          # no body: the defaults
    assert plain.status_code == 200 and seen["seed"] is None and seen["ignore_spill"] is False
    server.QUEUE.remove(plain.json()["id"])
    server.JOBS.pop(plain.json()["id"])
    # A 3D pick that hits nothing says so, not where it failed.
    pick = {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "fov": 50, "aspect": 1.5, "x": 0.5, "y": 0.5}
    job = server.JOBS[client.post(f"/api/projects/{name}/pick", json=pick).json()["id"]]
    assert job["request"] == pick
    assert client.get(f"/api/projects/{name}").json()["pick_pose"] == pick     # Back to 3D starts there
    server.QUEUE.remove(job["id"])
    job["steps"] = [s for s in job["steps"] if s["name"] == "comfy_hero"]
    (root / "pick.json").write_text(json.dumps({"status": "missed", "reason": "only sky there"}))
    server._run_job(job)
    assert job["status"] == "failed"
    assert job["error"].split("\n\n")[0] == "No object under the click: only sky there"


def test_ui_scene_missing_cameras_exits_naming_the_scene(tmp_path):
    pytest.importorskip("fastapi")
    import os
    import subprocess

    config = _write_config(tmp_path, scene={"title": "Toy", "source_ply": "toy/point_cloud.ply", "hero": "cam_a"})
    env = {**os.environ, "SPEDE_CONFIG": str(config)}
    proc = subprocess.run([sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1], run_name='check')",
                           str(STUDIO / "server.py")], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "'toy'" in proc.stderr and "cameras" in proc.stderr and "KeyError" not in proc.stderr


def _png_data_url(rgba) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgba, np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_ui_replace_upload_refusals_and_queue(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from PIL import Image

    server = _load_server(tmp_path, monkeypatch)
    assert {"replace", "comfy_replace.json"} <= set(server.REMOVE_STATE)
    assert server.PROJECT_FILES["replace_qwen"] == "replace/qwen_insert.png"
    assert server.PROJECT_FILES["replace_render"] == "replace/replaced_render.png"
    client = TestClient(server.app)
    name, root = _named_project(client, server, None)
    # Nothing to say; nothing selected yet.
    assert client.post(f"/api/projects/{name}/replace", json={"text": "  "}).status_code == 400
    refused = client.post(f"/api/projects/{name}/replace", json={"text": "a vase"})
    assert refused.status_code == 400 and "select an object first" in refused.json()["detail"]
    # The reference picture: a data URL, stored as RGB PNG (transparent -> white) for ComfyUI and the UI.
    assert client.post(f"/api/projects/{name}/replace_image", json={"data": "hello"}).status_code == 400
    assert client.post(f"/api/projects/{name}/replace_image", json={"data": "data:image/png;base64,bm9wZQ=="}).status_code == 400
    rgba = np.zeros((8, 12, 4), np.uint8)
    rgba[2:6, 3:9] = [200, 10, 10, 255]
    stored = client.post(f"/api/projects/{name}/replace_image", json={"data": _png_data_url(rgba)}).json()
    assert re.fullmatch(r"spede_replace_[\w-]+\.png", stored["image"]) and stored["size"] == [12, 8]
    comfy_copy = server.COMFY_ROOT / "input" / stored["image"]
    picture = np.asarray(Image.open(comfy_copy))
    assert picture.shape == (8, 12, 3) and tuple(picture[0, 0]) == (255, 255, 255) and tuple(picture[3, 4]) == (200, 10, 10)
    assert client.get(stored["url"]).status_code == 200
    # A confirmed selection is enough: Replace needs no Remove (it extracts first when there is no extract).
    (root / "inputs").mkdir(exist_ok=True)
    (root / "inputs/hero_mask.png").write_bytes(b"mask")
    unknown = client.post(f"/api/projects/{name}/replace", json={"text": "a vase", "image": "../config.json"})
    assert unknown.status_code == 400 and "reference picture" in unknown.json()["detail"]
    assert client.post(f"/api/projects/{name}/replace", json={"text": "a vase", "size": "huge"}).status_code == 422
    monkeypatch.setattr(server, "QWEN_ACK", False)
    assert "Research License" in client.post(f"/api/projects/{name}/replace", json={"text": "a vase"}).json()["detail"]
    monkeypatch.setattr(server, "QWEN_ACK", True)
    assert client.post(f"/api/projects/{name}/replace", json={"text": "a vase"}).status_code == 503   # no ComfyUI
    assert client.get("/api/jobs").json() == []
    monkeypatch.setattr(server, "_comfy_alive", lambda: True)
    monkeypatch.setattr(server, "_sdk_alive", lambda: True)
    queued = client.post(f"/api/projects/{name}/replace",
                         json={"text": " a  green vase ", "image": stored["image"], "seed": 5}).json()
    server.QUEUE.remove(queued["id"])                 # no worker thread in tests
    assert queued["kind"] == "replace" and [s["name"] for s in queued["steps"]] == ["comfy_replace"]
    assert queued["request"] == {"text": "a green vase", "image": stored["image"], "seed": 5, "size": "match"}
    busy = client.post(f"/api/projects/{name}/replace", json={"text": "a vase"})
    assert busy.status_code == 409
    server.JOBS[queued["id"]]["status"] = "done"
    # After a Replace the all-around fill would rebuild the scene without the new object: refused.
    (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": "depthfill", "comfy_output": str(root)}))
    (root / "comfy_replace.json").write_text("{}")
    later = client.post(f"/api/projects/{name}/allround")
    assert later.status_code == 400 and "before Replace" in later.json()["detail"]
    assert client.get(f"/api/projects/{name}").json()["replace"] == {}


def test_replace_step_is_an_inverse_extract_and_a_retry_swaps_the_object(tmp_path, monkeypatch):
    import shutil

    sys.path.insert(0, str(STUDIO))
    import comfy_backend
    from PIL import Image

    _write_scene(tmp_path / "scene", views=4)
    ply = tmp_path / "scene/point_cloud.ply"
    scene = {"source_ply": ply, "cameras": tmp_path / "scene/cameras.npz"}
    root, comfy_output, viewer = tmp_path / "runs/p", tmp_path / "output", tmp_path / "viewer"
    (root / "inputs").mkdir(parents=True)
    Image.new("L", (48, 32), 255).save(root / "inputs/hero_mask.png")
    graphs = []

    class Job:
        id = "replace-job"

    def fake_run(graph, log, timeout):
        graphs.append(graph)
        if len(graphs) == 4:
            raise comfy_backend.StepFailed("ComfyUI job x failed: SAM 3.1 found no 'vase'")
        out = comfy_output / Path(graph["100"]["inputs"]["output_path"]).parent
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ply, out / "replaced_scene.ply")
        if "x50" in graph:                            # the job extracted first
            (out / "extract").mkdir(exist_ok=True)
            shutil.copyfile(ply, out / "extract/object_only.ply")
            for sub in ("masks", "selection"):
                (out / sub).mkdir(exist_ok=True)
        (out / "replaced_scene.replace_report.json").write_text(json.dumps({
            "total_gaussians": 9, "object_gaussians": 5, "shadow_gaussians": 2, "fill_gaussians": 7,
            "object_ply": "obj.ply", "fill": {"object_gaussians": 2, "source": "plane", "filled_fraction": 0.6},
            "placement": {"yaw_deg": 90.0, "size": "match", "height_vs_removed": 1.4, "height_natural_vs_removed": 0.5,
                          "shift_fraction": 0.1, "shift_capped": False}}))
        for prefix in ("replace_qwen", "replace_render", "replace_clean_hero", "replace_cutout"):
            (out / f"{prefix}_00001_.png").write_bytes(prefix.encode())
        return Job()

    monkeypatch.setattr(comfy_backend, "run_graph", fake_run)
    project = {"name": "p", "label": "toy block", "view": "cam_a", "scene": "toy"}
    step = lambda **kw: comfy_backend.replace_step(project=project, scene=scene, root=root, comfy_output=comfy_output,
                                                   viewer=viewer, **kw)
    # Straight after the selection: no extract, no Remove. The job extracts first, from the hero mask.
    assert comfy_backend.replace_inputs(root) == {"object_ply": None, "remove_scene_ply": None, "hero_image": None}
    step(text="a green vase")()
    first = graphs[0]
    assert first["1"]["inputs"]["source_ply"] == str(ply.resolve())
    assert first["x50"]["class_type"] == "SpEdE_LocalExtract" and first["10"]["inputs"]["object_ply"] == ["x50", 0]
    assert first["100"]["inputs"]["object_ply"] == ["x50", 0] and first["100"]["inputs"]["size"] == "match"
    assert first["x23"]["inputs"]["hero_mask_path"] == str((root / "inputs/hero_mask.png").resolve())
    assert "as big as the toy block" in first["34"]["inputs"]["prompt"]
    assert sorted(p.name for p in (root / "replace").iterdir()) == [
        "clean_hero.png", "cutout.png", "qwen_insert.png", "replaced_render.png", "replaced_scene.ply"]
    assert (root / "extract/object_only.ply").exists()           # the project's Extract from now on
    summary = json.loads((root / "comfy_replace.json").read_text())
    assert summary["text"] == "a green vase" and summary["size"] == "match" and summary["fill_source"] == "plane"
    assert summary["placement"]["height_natural_vs_removed"] == 0.5 and summary["qwen_input"] == "base render"
    assert summary["base"] == "source" and summary["remove_scene_ply"] is None
    assert "source without the extract, footprint filled" in (root / "comfy_replace.log").read_text()
    manifest = json.loads((root / "session_manifest.json").read_text())
    assert manifest["current_ply"] == str(root / "replace/replaced_scene.ply") and manifest["generation"] == 1
    assert manifest["source_ply"] == str(ply) and "replace_base" not in manifest
    assert (viewer / "p/edited.splat").exists() and (viewer / "p/original.splat").exists()
    # Try again (new seed), natural size: again from the source scene and the extract on record, so the
    # new object is swapped, never stacked; the earlier result set aside.
    step(text="a green vase", seed=99, size="natural")()
    second = graphs[1]
    assert second["1"]["inputs"]["source_ply"] == str(ply.resolve()) and second["35a"]["inputs"]["noise_seed"] == 99
    assert "x50" not in second and second["100"]["inputs"]["object_ply"] == str((root / "extract/object_only.ply").resolve())
    assert second["100"]["inputs"]["size"] == "natural" and "as big as" not in second["34"]["inputs"]["prompt"]
    kept = list(root.glob("_previous_replace_*"))
    assert len(kept) == 1 and (kept[0] / "comfy_replace.json").exists() and (kept[0] / "replace/qwen_insert.png").exists()
    assert json.loads((root / "session_manifest.json").read_text())["generation"] == 2
    # With a Remove on record: its result is the base as it is, its cleaned hero is Qwen's input; the extract
    # on record only measures the old object.
    (root / "comfy").mkdir()
    shutil.copyfile(ply, root / "comfy/patched_scene.ply")
    (root / "comfy/qwen_edit.png").write_bytes(b"png")
    (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": "depthfill",
                                                      "patched_ply": str(root / "comfy/patched_scene.ply")}))
    step(text="a green vase", size="larger")()
    third = graphs[2]
    patched = str((root / "comfy/patched_scene.ply").resolve())
    assert third["1"]["inputs"]["source_ply"] == str(ply.resolve()) and "x50" not in third
    for node in ("10", "100"):
        assert third[node]["inputs"]["remove_scene_ply"] == patched and third[node]["inputs"]["remove_fill_ply"] == ""
        assert third[node]["inputs"]["object_ply"] == str((root / "extract/object_only.ply").resolve())
    assert third["10"]["inputs"]["hero_image_path"] == str((root / "comfy/qwen_edit.png").resolve())
    assert "bigger than the toy block" in third["34"]["inputs"]["prompt"]
    summary = json.loads((root / "comfy_replace.json").read_text())
    assert summary["base"] == "remove" and summary["remove_scene_ply"] == str(root / "comfy/patched_scene.ply")
    assert f"builds on the Remove result {root / 'comfy/patched_scene.ply'}" in (root / "comfy_replace.log").read_text()
    # The 3D compare stays original vs replaced; the manifest points at the replaced scene.
    assert json.loads((root / "session_manifest.json").read_text())["current_ply"] == str(root / "replace/replaced_scene.ply")
    # The all-around result, once there, wins over the hero patch; Replace again builds on it, never on the
    # previous Replace's scene.
    (root / "allround").mkdir()
    shutil.copyfile(ply, root / "allround/allround_scene.ply")
    (root / "comfy_allround.json").write_text(json.dumps({"allround_ply": str(root / "allround/allround_scene.ply")}))
    assert comfy_backend.replace_inputs(root)["remove_scene_ply"] == str(root / "allround/allround_scene.ply")
    # A failed Replace leaves the last one as it was.
    before = (root / "comfy_replace.json").read_text()
    with pytest.raises(comfy_backend.StepFailed, match="found no 'vase'"):
        step(text="a vase", image="spede_replace_p.png")()
    assert graphs[3]["3"]["inputs"]["image"] == "spede_replace_p.png"
    assert graphs[3]["100"]["inputs"]["remove_scene_ply"] == str((root / "allround/allround_scene.ply").resolve())
    assert (root / "comfy_replace.json").read_text() == before and (root / "replace/replaced_scene.ply").exists()


def test_ui_edited_and_replaced_flags_lock_the_selection(tmp_path, monkeypatch):
    """edited means a Remove ran, replaced a Replace: a Replace without a Remove is not a removal,
    but either fixes the selection (pick and another camera are refused), and the Replace-only
    project carries everything its Result page needs."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    server = _load_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "QWEN_ACK", True)
    client = TestClient(server.app)
    name, root = _named_project(client, server, None)
    (root / "view_rank.json").write_text(json.dumps({"views": [{"view_id": "cam_b", "points": [[3, 4, 1]]}]}))
    detail = client.get(f"/api/projects/{name}").json()
    assert detail["edited"] is False and detail["replaced"] is False
    # Replace only: the manifest exists (the 3D compare and the walk-through use it), but nothing was removed.
    (root / "replace").mkdir()
    (root / "replace/replaced_scene.ply").write_bytes(b"ply")
    (root / "comfy_replace.json").write_text(json.dumps({"text": "a green vase", "size": "match", "finished": 5.0}))
    (root / "session_manifest.json").write_text(json.dumps({"current_ply": str(root / "replace/replaced_scene.ply"),
                                                            "generation": 1}))
    detail = client.get(f"/api/projects/{name}").json()
    assert detail["edited"] is False and detail["replaced"] is True and detail["comfy"] is None
    assert detail["replace"]["text"] == "a green vase" and "replace_ply" in detail["files"]
    pick = {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "fov": 50, "aspect": 1.5, "x": 0.5, "y": 0.5}
    for refused in (client.post(f"/api/projects/{name}/pick", json=pick),
                    client.post(f"/api/projects/{name}/hero", json={"view": "cam_b"})):
        assert refused.status_code == 400 and "already has an edit" in refused.json()["detail"]
    assert not (root / "inputs/pick_pose.json").exists() and client.get("/api/jobs").json() == []
    # With a Remove on record the project is edited too.
    (root / "comfy_patch.json").write_text(json.dumps({"patch_mode": "depthfill", "comfy_output": str(root)}))
    detail = client.get(f"/api/projects/{name}").json()
    assert detail["edited"] is True and detail["replaced"] is True


def test_compare_page_treats_a_missing_export_as_none():
    """camera.json answers 404 with a JSON body when a project has no 3D export: the compare page must
    check the status, or it builds two viewers on missing splats (two black panes)."""
    page = (STUDIO / "compare.html").read_text()
    assert re.search(r"cam = res\.ok \? await res\.json\(\) : null", page)
    assert "No 3D export for this project yet" in page


def test_gpu_wait_reasons_read_as_words():
    sys.path.insert(0, str(STUDIO))
    import gpu_turn

    assert gpu_turn.wait_text({"state": "queued", "reason": "another owner holds the turn"}) == "another job is using the GPU"
    assert gpu_turn.wait_text({"state": "queued", "reason": "earlier eligible request"}) == "another job is ahead in the GPU queue"
    assert gpu_turn.wait_text({"state": "queued", "reason": ["ComfyUI queue is not empty"]}) == "ComfyUI is busy with another prompt"
    assert gpu_turn.wait_text({"state": "queued", "blocked_by": "ram", "reason": ["x"]}).startswith("waiting for free memory")


def test_ui_page_labels_for_first_time_users():
    """The renamed actions are in the page, the jargon labels they replace are not."""
    page = (STUDIO / "index.html").read_text()
    for text in ("Looks right → Remove", "Replace it with…", "Not right?", "Try another version", "Search for this name",
                 "Edit another object (new project)", "Show test runs", "Edit camera"):
        assert text in page, text
    for text in ("Try again (new seed)", "re-runs SAM", "The GPU turn reserves", "hero_only.splat", "the mat round it"):
        assert text not in page, text
