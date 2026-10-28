"""SpEdE: the web UI of the SpEdE ComfyUI node pack.

Edit a Gaussian-splat scene by pointing at it: click an object in a camera
photo (or in the free 3D view), let SAM 3.1 and Qwen3-VL agree on what it is,
then extract it as a clean object PLY, or remove it and have the surface it
hid rebuilt from Qwen-Image 2.1's repair as real Gaussians on MoGe depth
aligned to the scene (the depth fill; a planar patch is the fallback),
optionally make that fill hold from every direction, and put a new object
where the old one stood (Replace, beta: Qwen-Image 2.1 draws it, TripoSplat
makes it 3D). Every GPU step is one ComfyUI job submitted through the official Comfy
SDK v2 (comfy-api-proxy); camera ranking and 3D picks render with gsplat in a
subprocess. The server itself never touches the GPU.

    python studio/server.py --host 127.0.0.1 --port 8090          # exported node pack
    python competition/studio/server.py --host 127.0.0.1 --port 8090   # this repository

Configuration: ``studio/config.json`` (copy ``config.example.json``) or the
file ``SPEDE_CONFIG`` points at. The scalar keys (comfy_host, comfy_port,
comfy_sdk_url, comfy_root, runs, scene_root, cuda_device, patch_mode,
allround_mode, accept_qwen_research_license, gpu_turns, gpu_turns_root,
gpu_turns_ram_gb) can be overridden by ``SPEDE_<KEY>`` in the environment;
``scenes`` is read from the file only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STUDIO = Path(__file__).resolve().parent
PACK = STUDIO.parent              # the node pack root (export) or competition/ (repository)
for _p in (PACK, PACK.parent):    # spede_core and geometry_cli next to the pack root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(STUDIO) not in sys.path:
    sys.path.insert(0, str(STUDIO))
import comfy_backend  # noqa: E402
import gpu_turn  # noqa: E402
from spede_core.splat_export import ply_to_splat  # noqa: E402

# ---- configuration ---------------------------------------------------------------
CONFIG_PATH = Path(os.environ.get("SPEDE_CONFIG", STUDIO / "config.json")).expanduser()
if not CONFIG_PATH.is_file():
    raise SystemExit(f"SpEdE needs a configuration: copy {STUDIO / 'config.example.json'} to {CONFIG_PATH} "
                     "(or point SPEDE_CONFIG at a copy) and set comfy_root, scene_root and scenes")
CONFIG: dict = json.loads(CONFIG_PATH.read_text())


def _setting(key: str, default=None):
    env = os.environ.get("SPEDE_" + key.upper())
    return env if env is not None else CONFIG.get(key, default)


def _flag(key: str, default: bool) -> bool:
    return str(_setting(key, default)).lower() in {"1", "true", "yes", "on"}


RUNS = Path(str(_setting("runs", PACK / "runs"))).expanduser()
VIEWER = RUNS / "viewer"              # .splat files and the viewer's JS libraries
CACHE = RUNS / "camera_cache"         # the capture photos as PNG
VIEWER.mkdir(parents=True, exist_ok=True)
SCENE_ROOT = Path(str(_setting("scene_root", "."))).expanduser()


def _scene_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCENE_ROOT / path


SCENE_KEYS = ("hero", "source_ply", "cameras")      # required in every scene


def _scene(key: str, scene) -> dict:
    if not isinstance(scene, dict):
        raise SystemExit(f"SpEdE scene {key!r} in {CONFIG_PATH} must be an object with {', '.join(SCENE_KEYS)}")
    missing = [k for k in SCENE_KEYS if not scene.get(k)]
    if missing:
        raise SystemExit(f"SpEdE scene {key!r} in {CONFIG_PATH} is missing {', '.join(missing)} "
                         f"(every scene needs {', '.join(SCENE_KEYS)})")
    return {"title": scene.get("title", key.title()), "hero": scene["hero"],
            "source_ply": _scene_path(scene["source_ply"]), "cameras": _scene_path(scene["cameras"]),
            "heldout": _scene_path(scene["heldout"]) if scene.get("heldout") else None,
            # Full-resolution capture photos, for the all-around fill's 1024 px hero.
            "photos": _scene_path(scene["photos"]) if scene.get("photos") else None}


SCENES = {key: _scene(key, scene) for key, scene in (CONFIG.get("scenes") or {}).items()}
COMFY_URL = f"http://{_setting('comfy_host', '127.0.0.1')}:{_setting('comfy_port', '8188')}"
COMFY_SDK_URL = str(_setting("comfy_sdk_url", "http://127.0.0.1:8189"))
COMFY_ROOT = Path(str(_setting("comfy_root", ""))).expanduser()
if not str(_setting("comfy_root", "")) or not COMFY_ROOT.is_dir():
    raise SystemExit(f"comfy_root ({COMFY_ROOT}) must be the ComfyUI directory: its output/ folder receives the SpEdE results")
COMFY_OUTPUT = COMFY_ROOT / "output"
CUDA_DEVICE = str(_setting("cuda_device", "0"))
PATCH_MODE = str(_setting("patch_mode", "depthfill")).strip().lower()
if PATCH_MODE not in {"patch", "depthfill"}:
    raise SystemExit("SpEdE patch_mode must be depthfill (the default) or patch (the planar fallback)")
# Qwen-Image 2.1 for the hero repair in Remove: full (25 steps) or turbo (8 steps).
HERO_MODEL = str(_setting("hero_model", "turbo")).strip().lower()
if HERO_MODEL not in {"full", "turbo"}:
    raise SystemExit("SpEdE hero_model must be turbo (the default) or full")
ALLROUND_MODE = str(_setting("allround_mode", "independent")).strip().lower()
if ALLROUND_MODE not in {"independent", "progressive", "orbit"}:
    raise SystemExit("SpEdE allround_mode must be independent (the default), progressive (camera by camera) "
                     "or orbit (experimental MiniMax H3)")
# Qwen-Image 2.1 and its Qwen3-VL text encoder are Research License models:
# every step that would load them is refused until the owner acknowledges
# the licence (comfy_backend.check_qwen_license). Extract and a hero check
# with a typed name do not load them.
QWEN_ACK = _flag("accept_qwen_research_license", False)
PY = sys.executable
GEOMETRY_CLI = PACK / "geometry_cli.py"
# The shared-GPU turn system of the owner's workstation (gpu-turns); off unless configured.
gpu_turn.configure(enabled=_flag("gpu_turns", False), comfy_url=COMFY_URL,
                   root=_setting("gpu_turns_root"), ram_gb=_setting("gpu_turns_ram_gb"))
comfy_backend.configure(COMFY_SDK_URL, accept_qwen=QWEN_ACK)

app = FastAPI(title="SpEdE")


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_json(path: Path, payload) -> None:
    """A reader (the UI polling a project) never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _comfy_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as res:
            return res.status == 200
    except Exception:
        return False


def _sdk_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_SDK_URL}/api/v2/health", timeout=3) as res:
            return res.status == 200
    except Exception:
        return False


VIEWER_LIBS = {
    "three.module.js": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "gaussian-splats-3d.module.js":
        "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js",
}


def ensure_viewer_libs() -> dict[str, str]:
    """The 3D views import three.js and GaussianSplats3D (both MIT) from
    runs/viewer; fetched once when missing."""
    out = {}
    for name, url in VIEWER_LIBS.items():
        path = VIEWER / name
        if path.exists() and path.stat().st_size > 0:
            out[name] = "present"
            continue
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                data = res.read()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
            out[name] = f"downloaded from {url}"
        except Exception as error:
            out[name] = f"missing ({error}); the 3D views will not load until it is placed in {VIEWER}"
    return out


@app.get("/api/status")
def api_status():
    return {"comfy_url": COMFY_URL, "comfy": _comfy_alive(), "sdk_url": COMFY_SDK_URL, "sdk": _sdk_alive(),
            "gpu_turns": gpu_turn.SETTINGS["enabled"], "scenes": len(SCENES), "runs": str(RUNS),
            "qwen_research_license_acknowledged": QWEN_ACK,
            "viewer_libs": all((VIEWER / n).exists() for n in VIEWER_LIBS)}


@app.get("/api/scenes")
def api_scenes():
    return {k: {"title": v["title"], "hero": v["hero"], "size": _camera_size(k), "heldout": v["heldout"] is not None}
            for k, v in SCENES.items()}


@app.get("/files/{path:path}")
def files(path: str):
    target = (RUNS / path).resolve()
    if RUNS.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


# ---- jobs (one GPU: strictly one at a time) ---------------------------------

JOBS: dict[str, dict] = {}
QUEUE: list[str] = []
_lock = threading.Lock()
ACTIVE = {"queued", "running"}
JOBS_KEPT = 30                     # per project, in runs/<project>/jobs.json
# A subprocess step that ran but changed nothing says why in its JSON.
NOTHING_DONE = {"nothing_there", "no_camera", "empty_selection"}
NOTHING_TEXT = {"pick": "No object under the click"}     # what such a step's failure means to the user


def _error_text(log: str) -> str:
    """The step's exception line first (not the warnings a log starts with),
    then the end of the log."""
    if "Connection refused" in log and not _comfy_alive():
        return (f"ComfyUI stopped during this step ({COMFY_URL} refuses connections). Start it again and rerun.\n\n"
                + log[-600:])
    # A step's own explanation ("SpEdE: ..." in its log) comes first.
    own = re.findall(r"(?m)^SpEdE: (.+)$", log)
    if own:
        return own[-1] + "\n\n" + log[-1200:]
    # A ComfyUI node's own refusal reads best on its own, without the wrapper.
    node = re.findall(r"message=\\*'(.+?)(?:\\+n)?\\*' node_id", log)
    if node:
        return node[-1] + "\n\n" + log[-1200:]
    lines = [line for line in log.strip().splitlines() if line.strip()]
    final = next((line for line in reversed(lines)
                  if "Error" in line.split(":", 1)[0] or "Exception" in line.split(":", 1)[0]), "")
    return (final + "\n\n" if final else "") + log[-1200:]


def _persist(job: dict) -> None:
    """The job's public record on disk, so a restarted server still knows
    what ran, what failed and why."""
    root = RUNS / job["name"]
    try:
        root.mkdir(parents=True, exist_ok=True)
        with _lock:
            mine = sorted((j for j in JOBS.values() if j["name"] == job["name"]), key=lambda j: j["created"])
            records = [_public(j, locked=True) for j in mine[-JOBS_KEPT:]]
        _write_json(root / "jobs.json", records)
    except Exception:
        pass                            # bookkeeping must never fail a job


def _recover_jobs() -> None:
    """On start: the jobs.json of every project; a job that was queued or
    running when the server stopped is marked interrupted with its log tail."""
    for path in RUNS.glob("*/jobs.json"):
        for job in _json(path) or []:
            if not isinstance(job, dict) or "id" not in job:
                continue
            if job.get("status") in ACTIVE:
                job["status"] = "interrupted"
                step = next((s for s in job.get("steps", []) if s.get("status") == "running"), None)
                if step:
                    step["status"] = "interrupted"
                    log = path.parent / f"{step['name']}.log"
                    tail = log.read_text()[-600:] if log.exists() else ""
                    job["error"] = f"SpEdE restarted during '{step['name']}'. " + tail
                else:
                    job["error"] = "SpEdE restarted before this job ran"
                job["finished"] = job.get("finished") or time.time()
            job["steps"] = [{**s} for s in job.get("steps", [])]
            JOBS[job["id"]] = job
        if _json(path):
            _persist({"name": path.parent.name})


def _active_job(project: str) -> dict | None:
    with _lock:
        return next((j for j in JOBS.values() if j["name"] == project and j["status"] in ACTIVE), None)


def _job(kind: str, project: str, steps: list[tuple], after=None, prepare=None, request: dict | None = None) -> dict:
    """Queue a job: steps are (name, command, output) where command is a
    callable (runs in the worker), an argv list (a subprocess whose stdout is
    the JSON report ``output``) or None. ``prepare`` runs only once the job
    passed the ComfyUI and busy checks, just before it is queued. ``request``
    is the body it was started with, for the UI's Try again."""
    if not _comfy_alive():
        raise HTTPException(503, f"ComfyUI is not running at {COMFY_URL}. Start it and try again.")
    if not _sdk_alive():
        raise HTTPException(503, f"The Comfy SDK proxy is not answering at {COMFY_SDK_URL}. "
                                 "Start comfy-api-proxy and try again.")
    busy = _active_job(project)
    if busy:
        raise HTTPException(409, f"this project is busy ({busy['kind']} is {busy['status']}); wait for it or cancel it")
    if prepare is not None:
        prepare()
    job_id = uuid.uuid4().hex[:10]
    if gpu_turn.SETTINGS["enabled"]:
        steps = [("gpu_turn", None, None), *steps]
    job = {"id": job_id, "kind": kind, "name": project, "status": "queued", "created": time.time(),
           "steps": [{"name": n, "cmd": c, "out": o, "status": "pending"} for n, c, o in steps],
           "after": after, "request": request or {}}
    with _lock:
        JOBS[job_id] = job
        QUEUE.append(job_id)
    _persist(job)
    return _public(job)


def _cancel(job: dict) -> None:
    """A queued job leaves the queue; a running one has its ComfyUI job
    cancelled and its subprocess ended."""
    with _lock:
        if job["id"] in QUEUE:
            QUEUE.remove(job["id"])
        job["cancel"] = True
        proc = job.get("proc")
        running = job["status"] == "running"
    if running:
        comfy_backend.cancel_active()
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


def _worker():
    while True:
        job_id = None
        with _lock:
            if QUEUE:
                job_id = QUEUE.pop(0)
        if job_id is None:
            time.sleep(0.5)
            continue
        _run_job(JOBS[job_id])


def _run_job(job: dict) -> None:
    """Run one job's steps in this thread (the worker's unit of work; tests call it directly)."""
    if job.get("cancel"):
        job["status"], job["finished"] = "cancelled", time.time()
        _persist(job)
        return
    job["status"] = "running"
    _persist(job)
    root = RUNS / job["name"]
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_DEVICE}
    try:
        _run_steps(job, root, env)
    finally:
        turn, gpu_turn.CURRENT = gpu_turn.CURRENT, None
        if turn is not None:
            try:
                turn.release()
            except Exception as error:
                with open(root / "gpu_turn.log", "a") as handle:
                    handle.write(f"release failed: {error!r}\n")
    if job["status"] == "done" and job.get("after"):
        try:
            job["after"]()
        except Exception as error:
            # The steps ran, the bookkeeping after them did not: that is a failure, not a quiet done.
            job["status"], job["error"] = "failed", f"after the steps: {error!r}"
    job["finished"] = time.time()
    _persist(job)


def _run_steps(job: dict, root: Path, env: dict) -> None:
    for step in job["steps"]:
        step["status"], step["started"] = "running", time.time()
        _persist(job)
        log_path = root / f"{step['name']}.log"
        try:
            cmd = step["cmd"]
            if step["name"] == "gpu_turn":
                log_path.write_text("")
                turn = gpu_turn.GpuTurn(job["name"], job["kind"], log_path,
                                        cancelled=lambda: bool(job.get("cancel")),
                                        note=lambda text, step=step: step.__setitem__("note", text))
                turn.acquire()
                gpu_turn.CURRENT = turn
                code = 0
            elif callable(cmd):
                cmd()
                code = 0
            else:
                out = root / (step["out"] or f"{step['name']}.json")
                tmp = out.with_name(out.name + ".running")
                with open(tmp, "w") as stdout, open(log_path, "w") as stderr:
                    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=str(PACK), env=env)
                    job["proc"] = proc
                    gpu_turn.attach_pid(proc.pid)
                    code = proc.wait()
                    job["proc"] = None
                result = _json(tmp) or {}
                if code == 0 and result.get("status") in NOTHING_DONE:
                    with open(log_path, "a") as handle:
                        handle.write(f"\n{step['name']}: {result['status']} " + json.dumps(
                            {k: v for k, v in result.items() if k != "status"})[:600] + "\n")
                        if step["name"] in NOTHING_TEXT:
                            handle.write(f"SpEdE: {NOTHING_TEXT[step['name']]}: {result.get('reason') or result['status']}\n")
                    code = 1
                # The previous report survives a failed step: the result
                # becomes the report only on success.
                os.replace(tmp, out if code == 0 else out.with_name(out.stem + ".failed.json"))
        except Exception as error:          # a Python step failed
            with open(log_path, "a") as handle:
                handle.write(repr(error) + "\n")
            code = 1
        step["seconds"] = time.time() - step["started"]
        if job.get("cancel"):
            step["status"], job["status"] = "cancelled", "cancelled"
            job["error"] = "cancelled"
            break
        step["status"] = "done" if code == 0 else "failed"
        if code != 0:
            job["status"] = "failed"
            job["error"] = _error_text(log_path.read_text()) if log_path.exists() else "failed"
            break
    else:
        job["status"] = "done"


def _public(job: dict, *, locked: bool = False) -> dict:
    """A copy safe to serialise while the worker mutates the job."""
    def snapshot():
        return {**{k: v for k, v in job.items() if k not in {"steps", "after", "proc", "cancel"}},
                "steps": [{k: v for k, v in s.items() if k not in {"cmd", "out"}} for s in job["steps"]],
                "position": QUEUE.index(job["id"]) + 1 if job["id"] in QUEUE else 0}
    if locked:
        return snapshot()
    with _lock:
        return snapshot()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return _public(job)


@app.get("/api/jobs")
def api_jobs(limit: int = 60):
    with _lock:
        jobs = sorted(JOBS.values(), key=lambda j: -j["created"])[:max(1, min(limit, 500))]
    return [_public(j) for j in jobs]


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] not in ACTIVE:
        return _public(job)
    _cancel(job)
    return _public(job)


# ---- scene cameras (CPU: the photos stored with the camera targets) ----------

_SIZES: dict[str, list[int]] = {}


def _camera_labels(scene_key: str) -> list[str]:
    import numpy as np

    with np.load(SCENES[scene_key]["cameras"], allow_pickle=True) as data:
        return [str(v) for v in data["labels"]]


def _camera_size(scene_key: str) -> list[int]:
    """[width, height] of the capture photos (the click coordinates' frame)."""
    import numpy as np

    if scene_key not in _SIZES:
        with np.load(SCENES[scene_key]["cameras"], allow_pickle=True) as data:
            shape = data["targets"].shape
        _SIZES[scene_key] = [int(shape[2]), int(shape[1])]
    return _SIZES[scene_key]


@app.get("/api/scenes/{scene_key}/cameras")
def api_cameras(scene_key: str):
    if scene_key not in SCENES:
        raise HTTPException(404, "unknown scene")
    hero = SCENES[scene_key]["hero"]
    return [{"id": label, "image": f"/api/scenes/{scene_key}/camera/{label}.png", "hero": label == hero}
            for label in _camera_labels(scene_key)]


@app.get("/api/scenes/{scene_key}/camera/{label}.png")
def api_camera_image(scene_key: str, label: str):
    import numpy as np
    from PIL import Image

    if scene_key not in SCENES or not re.fullmatch(r"[\w-]+", label):
        raise HTTPException(404, "unknown camera")
    path = CACHE / scene_key / f"{label}.png"
    if not path.exists():
        with np.load(SCENES[scene_key]["cameras"], allow_pickle=True) as data:
            labels = [str(v) for v in data["labels"]]
            if label not in labels:
                raise HTTPException(404, "unknown camera")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(data["targets"][labels.index(label)]).save(path)
    return FileResponse(path)


# ---- projects -------------------------------------------------------------------

class NewProject(BaseModel):
    scene: str
    view: str | None = None
    title: str | None = None


class Selection(BaseModel):
    points: list[list[float]]           # [[x, y, 1 | 0], ...] in capture-photo pixels
    label: str | None = None


class LabelRequest(BaseModel):
    label: str


class RemoveRequest(BaseModel):
    seed: int | None = None             # Qwen-Image's seed; Try again sends a new one
    ignore_spill: bool = False          # Remove anyway: what Qwen also removed is only shadow


class ReplaceRequest(BaseModel):
    text: str                           # what to put there
    image: str | None = None            # a reference picture, as /replace_image named it
    seed: int | None = None             # Qwen-Image's seed; Try again sends a new one
    # The new object's size: match the removed object's dominant extent, Qwen's own (natural), or 1.3x match.
    size: Literal["match", "natural", "larger"] = "match"


class ReplaceImage(BaseModel):
    data: str                           # the picture as a data URL (data:image/...;base64,...)


class HeroChoice(BaseModel):
    use: str                            # "click" or "name"


class HeroRequest(BaseModel):
    view: str


class PickRequest(BaseModel):
    position: list[float]           # three.js camera, world frame of the splat
    quaternion: list[float]         # x, y, z, w
    fov: float                      # vertical, degrees
    aspect: float
    x: float                        # click, 0..1 of the viewer width
    y: float                        # click, 0..1 of the viewer height


def _project_path(name: str) -> Path:
    if not re.fullmatch(r"[\w-]+", name):
        raise HTTPException(404, "unknown project")
    return RUNS / name / "project.json"


def _load_project(name: str) -> dict:
    path = _project_path(name)
    if not path.exists():
        raise HTTPException(404, "unknown project")
    for attempt in range(3):               # writes are atomic; a read can still race a rename
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.05 * (attempt + 1))
    raise HTTPException(503, "project file is being written; try again")


def _save_project(project: dict) -> None:
    _write_json(_project_path(project["name"]), project)


def _set_stage(name: str, stage: str, **extra):
    def apply():
        project = _load_project(name)
        project["stage"] = stage
        project.update(extra)
        project.setdefault("history", []).append({"stage": stage, "at": time.time()})
        _save_project(project)
    return apply


@app.post("/api/projects")
def api_new_project(req: NewProject):
    if req.scene not in SCENES:
        raise HTTPException(400, "unknown scene")
    view = req.view or SCENES[req.scene]["hero"]
    if view not in _camera_labels(req.scene):
        raise HTTPException(400, "unknown camera")
    slug = re.sub(r"[^a-z0-9]+", "_", (req.title or "edit").lower()).strip("_")[:24] or "edit"
    name = f"spede_{req.scene}_{slug}_{time.strftime('%m%d_%H%M%S')}"
    project = {"name": name, "scene": req.scene, "view": view, "title": req.title or "Untitled edit",
               "stage": "select", "created": time.time(), "history": []}
    _save_project(project)
    return project


@app.get("/api/projects")
def api_projects():
    items = []
    for path in RUNS.glob("*/project.json"):
        project = _json(path)
        if project and project.get("scene") in SCENES:
            items.append(project)
    return sorted(items, key=lambda p: -p["created"])


PROJECT_FILES = {
    "selection": "inputs/hero_mask.png", "selection_overlay": "inputs/selection.png",
    "click_mask": "inputs/click_mask.png", "name_mask": "inputs/name_mask.png",
    "extract_ply": "extract/object_only.ply", "extract_semantic_ply": "extract/object_semantic.ply",
    "extract_rescued_ply": "extract/object_rescued.ply", "extract_sheet": "extract/extract_sheet.png",
    "extract_heldout": "extract/extract_heldout.png",
    "comfy_qwen_edit": "comfy/qwen_edit.png", "comfy_patched_render": "comfy/patched_render.png",
    "comfy_patched_ply": "comfy/patched_scene.ply",
    "allround_chain": "allround/chain_sheet.png",
    "allround_warp": "allround/warp_sheet.png", "allround_result": "allround/result_sheet.png",
    "allround_ply": "allround/allround_scene.ply",
    "allround_sheet": "allround/progressive_sheet.png", "allround_hero_clean": "allround/hero_clean.png",
    # Replace (comfy_backend.replace_step): Qwen's insert, the 3D result in the hero camera, the reference.
    "replace_qwen": "replace/qwen_insert.png", "replace_render": "replace/replaced_render.png",
    "replace_clean_hero": "replace/clean_hero.png", "replace_cutout": "replace/cutout.png",
    "replace_ply": "replace/replaced_scene.ply",
    # The last Remove that failed (comfy_backend.remove_patch_step): Qwen's picture and its spill.
    "edit_spill": "comfy_failed/edit_spill.png", "failed_qwen_edit": "comfy_failed/qwen_edit.png",
}


@app.get("/api/projects/{name}")
def api_project(name: str):
    project = _load_project(name)
    root = RUNS / name
    files = {}
    for key, rel in PROJECT_FILES.items():
        if (root / rel).exists():
            files[key] = f"/files/{name}/{rel}?t={int((root / rel).stat().st_mtime)}"
    busy = _active_job(name)
    return {**project, "files": files, "comfy": _json(root / "comfy_patch.json"),
            "allround": _json(root / "comfy_allround.json"), "replace": _json(root / "comfy_replace.json"),
            "job": _public(busy) if busy else None,
            # edited: the object was removed (a Remove ran); replaced: a Replace ran (with or without
            # a Remove). Either locks the selection: another object is a new project.
            "edited": (root / "comfy_patch.json").exists(), "replaced": (root / "comfy_replace.json").exists(),
            "extract": _json(root / "extract/object_extract_report.json"),
            "view_rank": _json(root / "view_rank.json"), "gates": _gates(root),
            # The 3D viewer's pose at the last pick: Back to 3D reopens it there.
            "pick_pose": _json(root / "inputs/pick_pose.json")}


def _gates(root: Path) -> dict:
    """Stage gates (G1 removal, G2 plane, G3 patch, H1 held-out) of the last
    Remove+Patch; the removal gate is stale once a new selection was made."""
    stored = _json(root / "gates.json")
    if not stored:
        return {}
    mask, hull = root / "inputs/hero_mask.png", root / "hull"
    if "removal" in stored and mask.exists() and hull.exists() and mask.stat().st_mtime > hull.stat().st_mtime:
        stored = {**stored, "removal": {**stored["removal"], "stale": True}}
    return stored


# ---- selection: clicks, the hero check, the cameras --------------------------------

def _points(points: list) -> list[list[float]]:
    return [[float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1] for p in points]


def _rank_views(name: str):
    """Score every camera for the current selection (geometry_cli rank-views);
    never fails the job, the ranking is advice."""
    def run():
        project = _load_project(name)
        root = RUNS / name
        scene = SCENES[project["scene"]]
        tmp = root / "view_rank.json.running"
        with open(tmp, "w") as out, open(root / "rank.log", "w") as err:
            proc = subprocess.Popen([PY, str(GEOMETRY_CLI), "rank-views", "--source-ply", str(scene["source_ply"]),
                                     "--camera-targets", str(scene["cameras"]), "--hero-view-id", project["view"],
                                     "--hero-mask", str(root / "inputs/hero_mask.png")],
                                    stdout=out, stderr=err, cwd=str(PACK),
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_DEVICE})
            gpu_turn.attach_pid(proc.pid)
            code = proc.wait()
        if code == 0 and (_json(tmp) or {}).get("status") == "ranked":
            os.replace(tmp, root / "view_rank.json")
        else:
            tmp.unlink(missing_ok=True)
            (root / "view_rank.json").unlink(missing_ok=True)
    return run


def _hero_step(name: str, points: list, label: str | None, label_source: str | None = None):
    project = _load_project(name)
    return comfy_backend.hero_step(name=name, scene=SCENES[project["scene"]], root=RUNS / name,
                                   comfy_output=COMFY_OUTPUT, points=_points(points), label=label,
                                   label_source=label_source, load=_load_project, save=_save_project)


def _hero_job(name: str, points: list, label: str | None, label_source: str | None = None) -> dict:
    """The ComfyUI hero-view check (clicks, name, agreement) and the camera ranking."""
    return _job("select", name, [("comfy_hero", _hero_step(name, points, label, label_source), None),
                                 ("rank", _rank_views(name), None)],
                request={"points": points, "label": label})


@app.post("/api/projects/{name}/select")
def api_select(name: str, req: Selection):
    """SAM 3.1 masks the clicks in the hero camera; Qwen3-VL names the mask
    unless the user did; SAM 3.1 then searches the same view by that name and
    SpEdE compares the two before any other camera is searched."""
    _load_project(name)
    if not req.points:
        raise HTTPException(400, "click the object at least once")
    if not any(len(p) < 3 or int(p[2]) for p in req.points):
        raise HTTPException(400, "place at least one include point on the object")
    label = (req.label or "").strip() or None
    if label is None:
        _require_qwen_ack()                 # Qwen3-VL would name the mask
    return _hero_job(name, req.points, label)


@app.post("/api/projects/{name}/label")
def api_label(name: str, req: LabelRequest):
    """Name (or rename) the selection: the other cameras search for the object
    by this name, so with clicks on record the name is checked against them
    again (SAM 3.1 only, no Qwen3-VL)."""
    project = _load_project(name)
    label = " ".join(req.label.strip().split())[:60]
    if not label:
        raise HTTPException(400, "type a name, e.g. bonsai tree")
    if project.get("points"):
        _hero_job(name, project["points"], label)
        return _load_project(name)
    project.update(label=label, label_source="user")
    _save_project(project)
    return project


@app.post("/api/projects/{name}/hero_choice")
def api_hero_choice(name: str, req: HeroChoice):
    """After the clicks and the name disagree, the user picks the mask that
    Extract and Remove carry to the other cameras."""
    import numpy as np
    from PIL import Image

    project = _load_project(name)
    root = RUNS / name
    source = {"click": root / "inputs/click_mask.png", "name": root / "inputs/name_mask.png"}.get(req.use)
    if source is None or not source.exists():
        raise HTTPException(400, f"no {req.use} mask to use")
    if _active_job(name):
        raise HTTPException(409, "this project is busy; wait for it or cancel it")
    mask = np.asarray(Image.open(source).convert("L")) > 127
    Image.fromarray((mask * 255).astype(np.uint8)).save(root / "inputs/hero_mask.png")
    image = np.asarray(Image.open(api_camera_image(project["scene"], project["view"]).path).convert("RGB"))
    overlay = image.astype(np.float32)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([215, 38, 61])
    Image.fromarray(overlay.astype(np.uint8)).save(root / "inputs/selection.png")
    check = dict(project.get("hero_check") or {}, chosen=req.use)
    project.update(hero_check=check, hero_confirmed=True, selection_pixels=int(mask.sum()))
    _save_project(project)
    return project


def _require_qwen_ack() -> None:
    """Refuse a step that would load Qwen-Image 2.1 or Qwen3-VL before it
    queues (and takes a GPU turn); comfy_backend.check_qwen_license in
    run_graph stays the enforcement point."""
    if not QWEN_ACK:
        raise HTTPException(400, comfy_backend.QWEN_LICENSE_TEXT)


def _hero_unconfirmed(project: dict) -> None:
    check = project.get("hero_check")
    if check and not project.get("hero_confirmed"):
        raise HTTPException(400, "the clicks and the name disagree (" + str(check.get("advice") or check.get("status"))
                                 + "): pick the clicked or the named mask, or rename the object")


def _locked(root: Path) -> bool:
    """A Remove or a Replace ran: the selection is fixed, another object is a new project."""
    return (root / "comfy_patch.json").exists() or (root / "comfy_replace.json").exists()


@app.post("/api/projects/{name}/hero")
def api_hero(name: str, req: HeroRequest):
    """Make another camera the hero; the selection moves with click points on
    the object's visible part in that camera, and the name is checked there."""
    project = _load_project(name)
    root = RUNS / name
    if _locked(root):
        raise HTTPException(400, "this project already has an edit; start a new edit to use another camera")
    if req.view not in _camera_labels(project["scene"]):
        raise HTTPException(400, "unknown camera")
    rank = _json(root / "view_rank.json") or {}
    row = next((r for r in rank.get("views", []) if r["view_id"] == req.view), None)
    if not row or not row.get("points"):
        raise HTTPException(400, "the selected object is not visible in that camera")
    if not project.get("label"):
        _require_qwen_ack()                 # Qwen3-VL would name the mask in that camera

    def set_view():
        current = _load_project(name)
        current.update(view=req.view, hero_check=None, hero_confirmed=False)
        _save_project(current)
    return _job("hero", name, [("hero", set_view, None),
                               ("comfy_hero", _hero_step(name, row["points"], project.get("label") or None,
                                                         project.get("label_source") or None), None),
                               ("rank", _rank_views(name), None)], request=req.model_dump())


@app.post("/api/projects/{name}/pick")
def api_pick(name: str, req: PickRequest):
    """A click in the free 3D view: the point under it, the real camera that
    sees it best from about where the user looked, then the hero check there."""
    project = _load_project(name)
    root = RUNS / name
    if _locked(root):
        raise HTTPException(400, "this project already has an edit; start a new edit to select another object")
    _require_qwen_ack()                     # a pick clears the name: Qwen3-VL names the new mask
    scene = SCENES[project["scene"]]
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    _write_json(root / "inputs/pick_pose.json", req.model_dump())

    def use_pick():
        result = _json(root / "pick.json") or {}
        if result.get("status") != "picked":
            reason = f"No object under the click: {result.get('reason') or 'nothing picked'}"
            with open(root / "comfy_hero.log", "a") as handle:
                handle.write(f"SpEdE: {reason}\n")
            raise RuntimeError(reason)
        current = _load_project(name)
        current.update(view=result["view_id"], points=[], label="", label_source="", hero_check=None,
                       hero_confirmed=False)
        _save_project(current)
        u, v = result["pixel"]
        _hero_step(name, [[round(u), round(v), 1]], None)()

    return _job("pick", name, [
        ("pick", [PY, str(GEOMETRY_CLI), "pick-camera", "--source-ply", str(scene["source_ply"]),
                  "--camera-targets", str(scene["cameras"]), "--pose-json", str(root / "inputs/pick_pose.json")],
         "pick.json"),
        ("comfy_hero", use_pick, None),
        ("rank", _rank_views(name), None)], request=req.model_dump())


# ---- extract, remove + patch -------------------------------------------------------

def _ready_for_comfy(name: str) -> tuple[dict, Path]:
    project = _load_project(name)
    root = RUNS / name
    if not (root / "inputs/hero_mask.png").exists():
        raise HTTPException(400, "select an object first")
    if not project.get("label"):
        raise HTTPException(400, "name the selection first: SAM 3.1 finds it in the other cameras by its name")
    _hero_unconfirmed(project)
    return project, root


@app.post("/api/projects/{name}/extract")
def api_extract(name: str):
    """Extract the current selection as an object-only Gaussian PLY (SAM 3.1 in
    every camera, visual hull, rescue and depth clean); the scene is untouched."""
    project, root = _ready_for_comfy(name)
    return _job("extract", name, [("comfy_extract", comfy_backend.extract_step(
        project=project, scene=SCENES[project["scene"]], root=root, comfy_output=COMFY_OUTPUT), None)], request={})


REMOVE_STATE = ("segment", "hull", "comfy", "comfy_patch.json", "session_manifest.json", "gates.json",
                "allround", "comfy_allround.json", "comfy_allround.log", *comfy_backend.REPLACE_STATE)


@app.post("/api/projects/{name}/remove")
def api_remove(name: str, req: RemoveRequest | None = None):
    """One ComfyUI job: SAM 3.1 in every camera, the visual hull, the deletion,
    Qwen-Image 2.1's repair of the hero view, Gaussian births on the scene
    (MoGe depth in the depth fill, a support plane in the planar patch) and
    the G1/G2/G3/H1 gates. Removing again keeps the earlier result, including
    an all-around fill, aside. ``seed`` varies Qwen-Image's repair;
    ``ignore_spill`` relaxes the spill gate (the extra is only shadow)."""
    req = req or RemoveRequest()
    project, root = _ready_for_comfy(name)
    _require_qwen_ack()

    def keep_previous():
        # Only once the job is accepted: a 503/409 refusal leaves the previous result intact.
        if (root / "session_manifest.json").exists():
            keep = root / f"_previous_{time.strftime('%H%M%S')}"
            keep.mkdir(exist_ok=True)
            for item in REMOVE_STATE:
                if (root / item).exists():
                    (root / item).rename(keep / item)
        # The hero-only baseline belongs to the previous all-around fill.
        (VIEWER / name / "hero_only.splat").unlink(missing_ok=True)
    return _job("remove", name, [("comfy_patch", comfy_backend.remove_patch_step(
        project=project, scene=SCENES[project["scene"]], root=root, comfy_output=COMFY_OUTPUT, viewer=VIEWER,
        mode=PATCH_MODE, hero_model=HERO_MODEL, seed=req.seed, ignore_spill=req.ignore_spill), None)],
        after=_set_stage(name, "patch"), prepare=keep_previous, request=req.model_dump())


@app.post("/api/projects/{name}/allround")
def api_allround(name: str):
    """The all-around fill on the finished depth fill. Independent (the
    default): Qwen-Image 2.1 turbo cleans every camera that sees the hole on
    its own photo, the surface under the object is rebuilt and painted from
    above, and one optimisation fits the fill to every cleaned camera (about
    10 min on the RTX 5090 after a short first stage: the kitchen experiment
    of 23 Oct 2026). ``allround_mode`` progressive is the earlier
    camera-by-camera fill (10–25 min, about 26 GB of RAM: peak of the 21 Oct
    2026 run; the 23 Oct runs ran under a 30 GB reservation); orbit is the
    experimental MiniMax H3 orbit-video path."""
    project = _load_project(name)
    root = RUNS / name
    patch = _json(root / "comfy_patch.json")
    if not patch:
        raise HTTPException(400, "remove and patch the object first")
    if patch.get("patch_mode") != "depthfill":
        raise HTTPException(400, "the all-around fill builds on the depth fill (patch_mode depthfill); "
                                 "this result is a planar patch")
    if not project.get("label"):
        raise HTTPException(400, "name the selection first")
    if (root / "comfy_replace.json").exists():
        raise HTTPException(400, "Replace builds on the all-around fill, not the other way round: "
                                 "remove again to start over, then run the all-around fill before Replace")
    _require_qwen_ack()
    # The job kind picks the GPU turn's estimate (gpu_turn.NEEDS).
    kind = "allround_orbit" if ALLROUND_MODE == "orbit" else "allround"
    return _job(kind, name, [("comfy_allround", comfy_backend.allround_step(
        project=project, scene=SCENES[project["scene"]], root=root, comfy_output=COMFY_OUTPUT, viewer=VIEWER,
        mode=ALLROUND_MODE), None)],
        after=_set_stage(name, "allround"), request={})


# ---- replace (beta) ------------------------------------------------------------------

REPLACE_IMAGE_BYTES = 20 * 2 ** 20
REPLACE_IMAGE_NAME = re.compile(r"spede_replace_[\w-]+\.png")


@app.post("/api/projects/{name}/replace_image")
def api_replace_image(name: str, req: ReplaceImage):
    """A reference picture for Replace: stored as PNG (RGB, transparency on
    white, at most 2048 px) in ComfyUI's input folder, where the workflow's
    LoadImage reads it, and in the project for the UI."""
    import base64
    import io

    from PIL import Image

    _load_project(name)
    head, _, body = req.data.partition(",")
    if not head.startswith("data:image/") or ";base64" not in head:
        raise HTTPException(400, "send the picture as a data:image/...;base64 URL")
    try:
        raw = base64.b64decode(body, validate=True)
    except ValueError:
        raise HTTPException(400, "the picture is not valid base64") from None
    if len(raw) > REPLACE_IMAGE_BYTES:
        raise HTTPException(413, f"the picture is larger than {REPLACE_IMAGE_BYTES // 2 ** 20} MB")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            picture = im.convert("RGBA")
    except Exception:
        raise HTTPException(400, "not a picture SpEdE can read (PNG, JPEG or WebP)") from None
    white = Image.new("RGBA", picture.size, (255, 255, 255, 255))
    picture = Image.alpha_composite(white, picture).convert("RGB")
    picture.thumbnail((2048, 2048))
    filename = f"spede_replace_{name[-40:]}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    inputs = COMFY_ROOT / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    picture.save(inputs / filename)
    local = RUNS / name / "replace_inputs" / filename
    local.parent.mkdir(parents=True, exist_ok=True)
    picture.save(local)
    return {"image": filename, "url": f"/files/{name}/replace_inputs/{filename}", "size": list(picture.size)}


@app.post("/api/projects/{name}/replace")
def api_replace(name: str, req: ReplaceRequest):
    """Replace: put ``text`` (like the reference picture ``image``) where the
    selected object stands, at ``size``. Straight after the selection, no
    Remove needed: the new object goes into the source scene without exactly
    the extracted object (the project's Extract, made in the same job if
    there is none), and the old contact footprint it leaves bare is filled.
    After a Remove it goes into the Remove's result (the all-around fill,
    else the hero patch) as it is, the spot already filled there. One
    ComfyUI job of about a minute, a few more with the extract: Qwen-Image
    2.1 turbo draws it into the hero view, SAM 3.1 cuts it out, TripoSplat
    makes Gaussians of it and SpEdE stands them on the old object's surface
    with a contact shadow. Replacing again swaps the new object (the earlier
    result is kept aside)."""
    project, root = _ready_for_comfy(name)
    text = " ".join(req.text.split())[:120]
    if not text:
        raise HTTPException(400, "say what to put there, e.g. a green ceramic vase")
    if req.image is not None and (not REPLACE_IMAGE_NAME.fullmatch(req.image)
                                  or not (COMFY_ROOT / "input" / req.image).is_file()):
        raise HTTPException(400, "unknown reference picture; add it again")
    _require_qwen_ack()
    return _job("replace", name, [("comfy_replace", comfy_backend.replace_step(
        project=project, scene=SCENES[project["scene"]], root=root, comfy_output=COMFY_OUTPUT, viewer=VIEWER,
        text=text, image=req.image, seed=req.seed, size=req.size), None)],
        after=_set_stage(name, "replace"), request={**req.model_dump(), "text": text})


# ---- 3D views -------------------------------------------------------------------------

def _object_focus(ply: Path) -> tuple[list[float], float] | None:
    """Centre (median of the visible Gaussians' means) and radius (90th
    percentile distance from it) of an object PLY: the extracted object's
    orbit pivot."""
    import numpy as np
    from plyfile import PlyData

    v = PlyData.read(str(ply))["vertex"].data
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    visible = np.asarray(v["opacity"], np.float64) > -2.2          # sigmoid > 0.1
    if visible.sum() >= 1:
        xyz = xyz[visible]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if not len(xyz):
        return None
    centre = np.median(xyz, axis=0)
    radius = float(np.percentile(np.linalg.norm(xyz - centre, axis=1), 90))
    return centre.tolist(), radius


def _explore_payload(ply: Path, splat: Path, scene: dict, view: str | None, focus: bool = False) -> dict:
    """A splat for free navigation (converted once) and a start camera at a
    real view. ``focus`` (one object): the orbit pivot is the object's centre,
    and a camera far from a small object moves toward it along its line of sight."""
    import numpy as np

    if not splat.exists():
        ply_to_splat(Path(ply), splat)
    with np.load(scene["cameras"], allow_pickle=True) as data:
        labels = [str(v) for v in data["labels"]]
        viewmats = np.asarray(data["viewmats"], np.float64)
        intrinsics = np.asarray(data["intrinsics"], np.float64)
    index = labels.index(view) if view in labels else labels.index(scene["hero"])
    centres = np.stack([np.linalg.inv(m)[:3, 3] for m in viewmats])
    scale = float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) or 1.0
    c2w = np.linalg.inv(viewmats[index])
    k = intrinsics[index]
    fov = float(np.degrees(2 * np.arctan(k[1, 2] / k[1, 1])))
    position, look_at = c2w[:3, 3], c2w[:3, 3] + c2w[:3, 2] * 0.5 * scale
    object_info = _object_focus(ply) if focus else None
    if object_info:
        centre, radius = np.asarray(object_info[0]), object_info[1]
        look_at = centre
        # The object's diameter fills about half the frame height, never farther than the real camera.
        offset = position - centre
        distance, fit = float(np.linalg.norm(offset)), 2 * radius / np.tan(np.radians(fov) / 2)
        if fit > 0 and distance > fit:
            position = centre + offset / distance * fit
    return {"splat": f"/viewer/{splat.relative_to(VIEWER).as_posix()}?t={int(splat.stat().st_mtime)}",
            "camera": {"position": position.tolist(), "lookAt": look_at.tolist(),
                       "up": (-c2w[:3, 1]).tolist(), "fov": fov, "view": labels[index]},
            "scale": scale, **({"object": {"centre": object_info[0], "radius": object_info[1]}} if object_info else {})}


@app.get("/api/scenes/{scene_key}/explore")
def api_explore(scene_key: str, view: str | None = None):
    if scene_key not in SCENES:
        raise HTTPException(404, "unknown scene")
    scene = SCENES[scene_key]
    return _explore_payload(Path(scene["source_ply"]), VIEWER / "_scenes" / scene_key / "original.splat", scene, view)


@app.get("/api/projects/{name}/explore")
def api_project_explore(name: str, view: str | None = None):
    """The project's patched scene for free navigation."""
    project = _load_project(name)
    manifest = _json(RUNS / name / "session_manifest.json")
    if not manifest:
        raise HTTPException(400, "nothing edited yet")
    current = Path(manifest["current_ply"])
    stamp = int(current.stat().st_mtime) if current.exists() else 0
    folder = VIEWER / name
    splat = folder / f"result_g{int(manifest['generation']):04d}_{stamp}.splat"
    for old in folder.glob("result_*.splat") if folder.exists() else []:
        if old != splat:                   # Remove again reuses generation numbers
            old.unlink(missing_ok=True)
    return _explore_payload(current, splat, SCENES[project["scene"]], view or project["view"])


@app.get("/api/projects/{name}/extract/explore")
def api_extract_explore(name: str, view: str | None = None):
    project = _load_project(name)
    ply = RUNS / name / "extract/object_only.ply"
    if not ply.exists():
        raise HTTPException(404, "no extracted object yet")
    splat = VIEWER / name / f"extracted_{int(ply.stat().st_mtime)}.splat"
    for old in splat.parent.glob("extracted_*.splat") if splat.parent.exists() else []:
        if old != splat:
            old.unlink(missing_ok=True)
    return _explore_payload(ply, splat, SCENES[project["scene"]], view or project.get("view"), focus=True)


app.mount("/viewer", StaticFiles(directory=str(VIEWER)), name="viewer")


# The pages revalidate on every load: after an update the browser must not keep serving an old page.
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def index():
    return FileResponse(STUDIO / "index.html", headers=NO_CACHE)


@app.get("/explore.html")
def explore():
    return FileResponse(STUDIO / "explore.html", headers=NO_CACHE)


@app.get("/compare.html")
def compare():
    return FileResponse(STUDIO / "compare.html", headers=NO_CACHE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpEdE web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    for lib, state in ensure_viewer_libs().items():
        print(f"viewer library {lib}: {state}")
    if not SCENES:
        print(f"warning: no scenes in {CONFIG_PATH}; add them under \"scenes\"")
    print(f"SpEdE on http://{args.host}:{args.port} · ComfyUI {COMFY_URL} · SDK proxy {COMFY_SDK_URL} · runs {RUNS}")
    _recover_jobs()
    threading.Thread(target=_worker, daemon=True).start()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
