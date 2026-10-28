"""The UI's share of a GPU other agents also use (gpu-turns), optional.

On the owner's workstation the agents take turns on the RTX 5090 through
gpu-turns (file state under ~/.local/state/gpu-turns, shared by every
participant). UI jobs start when the user clicks, so the UI takes part itself:
each job that uses the GPU queues for a turn, heartbeats while waiting and
running, attaches what it launched (a subprocess PID or a ComfyUI prompt) and
releases the turn once that has really finished.

Off unless ``gpu_turns`` is enabled in the configuration and ``gpu_turns_root``
points at a gpu-turns checkout; a machine without the turn system runs jobs
directly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SETTINGS = {"enabled": False, "comfy_url": "http://127.0.0.1:8188", "root": None, "ram_gb": 16.0}
# Jobs that need more than the default: the all-around fill (a short
# Qwen-Image stage, then the independent fill, about 10 min, or the
# progressive one, 10–25 min of per-camera Qwen-Image turbo cleans and
# optimisations; about 26 GB of RAM (peak of the 21 Oct 2026 progressive
# run; the 23 Oct runs ran under a 30 GB reservation); the turn
# reserves 35 min and 30 GB as headroom), and the experimental orbit path
# (MiniMax H3, about 40 GB of weights, a 10 000-step optimisation).
NEEDS = {"allround": {"estimate_minutes": 35.0, "ram_gb": 30.0},
         # Replace: Qwen-Image turbo, SAM 3.1 and TripoSplat in one prompt, about a minute once loaded.
         "replace": {"estimate_minutes": 5.0},
         "allround_orbit": {"estimate_minutes": 60.0, "ram_gb": 40.0}}
CURRENT: "GpuTurn | None" = None      # the turn of the job the worker is running


def configure(*, enabled: bool, comfy_url: str, root: str | None = None, ram_gb: float | None = None) -> None:
    SETTINGS["enabled"] = bool(enabled)
    SETTINGS["comfy_url"] = comfy_url
    if root:
        SETTINGS["root"] = str(Path(root).expanduser())
    if ram_gb:
        SETTINGS["ram_gb"] = float(ram_gb)
    if SETTINGS["enabled"] and not SETTINGS["root"]:
        raise SystemExit("gpu_turns is enabled but gpu_turns_root (a gpu-turns checkout) is not configured")


def _log(log_path: Path, text: str) -> None:
    with open(log_path, "a") as handle:
        handle.write(time.strftime("%H:%M:%S ") + text + "\n")


def comfy_history_ids(limit: int = 8) -> list[str]:
    """Newest ComfyUI prompt ids first (ComfyUI keeps history in run order)."""
    with urllib.request.urlopen(f"{SETTINGS['comfy_url']}/history?max_items={limit}", timeout=10) as res:
        return list(json.load(res))[::-1]


def wait_text(got: dict) -> str:
    """Why a GPU turn is not granted yet, in the user's words."""
    reason = got.get("reason")
    reasons = reason if isinstance(reason, list) else [reason]
    text = " ".join(str(r) for r in reasons if r)
    if got.get("blocked_by") == "ram" or got.get("state") == "ram_queued":
        return "waiting for free memory on this machine"
    if "another owner" in text:
        return "another job is using the GPU"
    if "earlier eligible" in text:
        return "another job is ahead in the GPU queue"
    if "queue is not empty" in text:
        return "ComfyUI is busy with another prompt"
    return "waiting: " + (text[:120] or "the GPU is not free yet")


class TurnFailed(RuntimeError):
    pass


class GpuTurn:
    def __init__(self, project: str, kind: str, log_path: Path, *, cancelled=lambda: False,
                 estimate_minutes: float = 10.0, note=lambda text: None):
        root = SETTINGS["root"]
        if not root:
            raise TurnFailed("gpu_turns_root is not configured")
        if root not in sys.path:
            sys.path.insert(0, root)
        from gpu_turns.core import Store
        from gpu_turns.observe import comfy_free, comfy_job_finished, job_memory, observe, observe_memory

        url = SETTINGS["comfy_url"]
        state_dir = Path(os.environ.get("GPU_TURNS_STATE_DIR", Path.home() / ".local/state/gpu-turns"))
        self.store = Store(state_dir, observer=lambda: observe(url),
                           job_checker=lambda job: comfy_job_finished(url, job),
                           memory_observer=observe_memory, job_memory=job_memory,
                           comfy_freer=lambda: comfy_free(url))
        self.project, self.kind, self.log = project, kind, log_path
        need = NEEDS.get(kind, {})
        self.cancelled = cancelled
        self.note = note               # the wait reason in the user's words (the UI's wait overlay)
        self.estimate = need.get("estimate_minutes", estimate_minutes)
        self.ram_gb = max(float(SETTINGS["ram_gb"]), need.get("ram_gb", 0.0))
        self.request_id = self.token = None
        self.attached = 0
        self._stop = threading.Event()

    def _beat(self, progress: str) -> dict:
        return self.store.heartbeat(self.request_id, self.token, progress)

    def acquire(self) -> None:
        made = self.store.request("spede-ui", f"spede-ui-{self.project}", "ComfyUI-SpEdE",
                                  f"SpEdE UI {self.kind}", self.estimate, ram_gb=self.ram_gb,
                                  # The job runs inside ComfyUI: when only its idle model cache
                                  # keeps RAM short, the scheduler may unload it (both queues empty).
                                  comfy_free=True)
        self.request_id, self.token = made["request_id"], made["token"]
        _log(self.log, f"GPU turn requested ({self.request_id[:8]})")
        last_beat, last_reason = time.time(), None
        while True:
            if self.cancelled():
                self.store.cancel(self.request_id, self.token)
                self.request_id = None
                raise TurnFailed("cancelled while waiting for the GPU turn")
            got = self.store.acquire(self.request_id, self.token)
            if got["state"] == "acquired":
                _log(self.log, "GPU turn acquired")
                break
            reason = json.dumps(got.get("reason"))
            if reason != last_reason:
                _log(self.log, f"waiting for the GPU turn: {reason}")
                last_reason = reason
                try:
                    self.note(wait_text(got))
                except Exception:
                    pass
            if time.time() - last_beat > 60:
                self._beat("SpEdE UI job waiting for the GPU")
                last_beat = time.time()
            time.sleep(10)
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _keepalive(self) -> None:
        while not self._stop.wait(60):
            try:
                self._beat(f"SpEdE UI {self.kind} running")
            except Exception as error:
                _log(self.log, f"GPU turn heartbeat failed: {error!r}")

    def attach(self, job: dict) -> None:
        # gpu-turns replaces the attached job once the previous one has ended.
        for _attempt in range(30):
            try:
                self.store.attach(self.request_id, self.token, job)
                self.attached += 1
                return
            except Exception as error:
                last = error
                time.sleep(1)
        _log(self.log, f"GPU turn attach failed: {last!r}")

    def release(self) -> None:
        self._stop.set()
        if self.request_id is None:
            return
        if not self.attached:
            self._recover("the UI job acquired the turn but launched nothing on the GPU")
            return
        for _attempt in range(24):
            if self.store.release(self.request_id, self.token)["state"] == "released":
                _log(self.log, "GPU turn released")
                return
            time.sleep(5)
        self._recover("the UI job finished but its last GPU job stayed unverified for 2 min")

    def _recover(self, reason: str) -> None:
        # The owner pre-authorized recovering the UI's OWN stuck turns.
        result = subprocess.run(
            [str(Path(SETTINGS["root"]) / ".venv/bin/python"), "-m", "gpu_turns.cli",
             "--comfy-url", SETTINGS["comfy_url"], "recover", "--user-approved", "--reason",
             f"own turn {self.request_id[:8]} (spede-ui {self.project}): {reason}"],
            cwd=SETTINGS["root"], capture_output=True, text=True, timeout=120)
        _log(self.log, f"GPU turn recover: {(result.stdout or result.stderr).strip()[-300:]}")


def attach_pid(pid: int) -> None:
    if CURRENT is not None:
        CURRENT.attach({"kind": "pid", "pid": int(pid), "prompt_id": None})


def attach_prompt(prompt_id: str) -> None:
    if CURRENT is not None:
        CURRENT.attach({"kind": "comfy", "pid": None, "prompt_id": str(prompt_id)})
