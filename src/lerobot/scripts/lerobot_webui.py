#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser control panel for the ob15 robot.

Runs a small FastAPI server (``lerobot-webui``) on the ob15 itself so an operator can, from any
browser on the same network, start/stop:

  * ``lerobot-record``   -- teleop-driven dataset recording
  * ``lerobot-rollout --strategy.type=base``   -- autonomous policy inference
  * ``lerobot-rollout --strategy.type=dagger`` -- DAgger human-in-the-loop rollout collection

Each action is launched as a real CLI subprocess (the same console scripts you'd type by hand),
because these entry points open exclusive hardware handles (motors, cameras) and are designed to
own the process they run in. Only one job may run at a time since they all contend for the same
ob15 hardware. "Stop" sends SIGINT (the scripts catch it and shut down gracefully, saving/uploading
in progress data); "Force stop" sends SIGKILL for a hung process.
"""

import argparse
import logging
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class Job:
    mode: str
    argv: list[str]
    process: subprocess.Popen
    started_at: float
    log: deque = field(default_factory=lambda: deque(maxlen=2000))


class ProcessManager:
    """Owns at most one running subprocess at a time (the ob15 hardware is exclusive)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None

    def start(self, mode: str, argv: list[str]) -> Job:
        with self._lock:
            if self._job is not None and self._job.process.poll() is None:
                raise RuntimeError(
                    f"A job is already running: '{self._job.mode}' (pid {self._job.process.pid}). "
                    "Stop it before starting another."
                )
            process = subprocess.Popen(  # noqa: S603
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job = Job(mode=mode, argv=argv, process=process, started_at=time.time())
            self._job = job
        threading.Thread(target=self._pump_output, args=(job,), daemon=True).start()
        return job

    def _pump_output(self, job: Job) -> None:
        assert job.process.stdout is not None
        try:
            for line in job.process.stdout:
                job.log.append(line.rstrip("\n"))
        finally:
            job.process.wait()

    def status(self) -> dict:
        with self._lock:
            job = self._job
        if job is None:
            return {"running": False, "mode": None, "log": []}
        running = job.process.poll() is None
        return {
            "running": running,
            "mode": job.mode,
            "pid": job.process.pid,
            "returncode": job.process.returncode,
            "argv": job.argv,
            "started_at": job.started_at,
            "log": list(job.log)[-400:],
        }

    def send_signal(self, sig: signal.Signals) -> None:
        with self._lock:
            job = self._job
        if job is None or job.process.poll() is not None:
            raise RuntimeError("No job is currently running.")
        job.process.send_signal(sig)


manager = ProcessManager()
app = FastAPI(title="LeRobot ob15 Control Panel")


class RecordRequest(BaseModel):
    repo_id: str
    single_task: str = ""
    teleop_type: str = "xlerobot_vr"
    fps: int = 30
    num_episodes: int = 50
    episode_time_s: float = 180
    reset_time_s: float = 60
    push_to_hub: bool = True
    resume: bool = False


class InferenceRequest(BaseModel):
    policy_path: str
    duration: float = 0.0
    fps: float = 30.0


class DAggerRequest(BaseModel):
    policy_path: str
    repo_id: str
    single_task: str = ""
    teleop_type: str = "xlerobot_vr"
    input_device: str = "teleop"
    num_episodes: int = 50
    record_autonomous: bool = False
    duration: float = 0.0
    fps: float = 30.0
    push_to_hub: bool = True


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _launch(mode: str, argv: list[str]) -> dict:
    try:
        manager.start(mode, argv)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return manager.status()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status")
def status() -> dict:
    return manager.status()


@app.post("/api/record")
def start_record(req: RecordRequest) -> dict:
    if not req.repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    argv = [
        "lerobot-record",
        "--robot.type=ob15",
        f"--dataset.repo_id={req.repo_id}",
        f"--dataset.single_task={req.single_task}",
        f"--dataset.fps={req.fps}",
        f"--dataset.num_episodes={req.num_episodes}",
        f"--dataset.episode_time_s={req.episode_time_s}",
        f"--dataset.reset_time_s={req.reset_time_s}",
        f"--dataset.push_to_hub={_bool(req.push_to_hub)}",
        f"--teleop.type={req.teleop_type}",
        f"--resume={_bool(req.resume)}",
    ]
    return _launch("record", argv)


@app.post("/api/inference")
def start_inference(req: InferenceRequest) -> dict:
    if not req.policy_path:
        raise HTTPException(status_code=400, detail="policy_path is required")
    argv = [
        "lerobot-rollout",
        "--robot.type=ob15",
        f"--policy.path={req.policy_path}",
        "--strategy.type=base",
        f"--duration={req.duration}",
        f"--fps={req.fps}",
    ]
    return _launch("inference", argv)


@app.post("/api/dagger")
def start_dagger(req: DAggerRequest) -> dict:
    if not req.policy_path:
        raise HTTPException(status_code=400, detail="policy_path is required")
    if not req.repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    argv = [
        "lerobot-rollout",
        "--robot.type=ob15",
        f"--policy.path={req.policy_path}",
        "--strategy.type=dagger",
        f"--strategy.input_device={req.input_device}",
        f"--strategy.record_autonomous={_bool(req.record_autonomous)}",
        f"--strategy.num_episodes={req.num_episodes}",
        f"--teleop.type={req.teleop_type}",
        f"--dataset.repo_id={req.repo_id}",
        f"--dataset.single_task={req.single_task}",
        f"--dataset.push_to_hub={_bool(req.push_to_hub)}",
        f"--duration={req.duration}",
        f"--fps={req.fps}",
    ]
    return _launch("dagger", argv)


@app.post("/api/stop")
def stop() -> dict:
    try:
        manager.send_signal(signal.SIGINT)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return manager.status()


@app.post("/api/kill")
def kill() -> dict:
    try:
        manager.send_signal(signal.SIGKILL)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return manager.status()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ob15 control panel</title>
<style>
  /* Sized generously throughout: this page is commonly opened in the Quest browser (a
     standalone tab, not the WebXR/XLeVR scene) and driven with a controller ray-pointer
     rather than a mouse, so targets need to be bigger and text more legible at arm's
     length than a typical desktop control panel. */
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, system-ui, sans-serif;
    background: #14161a; color: #e6e6e6; font-size: 18px;
  }
  header {
    padding: 1.1rem 1.5rem; border-bottom: 1px solid #2a2d33;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
  }
  header h1 { font-size: 1.3rem; margin: 0; font-weight: 600; }
  #statusPill {
    padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.95rem; font-weight: 600;
    background: #333; color: #aaa;
  }
  #statusPill.running { background: #1f4f2c; color: #7fe0a0; }
  main { max-width: 900px; margin: 0 auto; padding: 1.5rem; }
  .tabs { display: flex; gap: 0.6rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .tab {
    padding: 0.85rem 1.3rem; border-radius: 10px; cursor: pointer; background: #1d2025;
    border: 1px solid #2a2d33; font-size: 1.05rem; min-height: 3.2rem; display: flex; align-items: center;
  }
  .tab.active { background: #2c5cc9; border-color: #2c5cc9; color: white; }
  .panel { display: none; background: #1a1c20; border: 1px solid #2a2d33; border-radius: 12px; padding: 1.3rem 1.5rem; }
  .panel.active { display: block; }
  label { display: block; font-size: 0.95rem; color: #adb3ba; margin: 1rem 0 0.35rem; }
  input[type=text], input[type=number] {
    width: 100%; padding: 0.85rem 0.9rem; border-radius: 8px; border: 1px solid #3a3e46;
    background: #101216; color: #e6e6e6; font-size: 1.05rem; min-height: 3.1rem;
  }
  .row { display: flex; gap: 0.9rem; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 140px; }
  .checkbox {
    display: flex; align-items: center; gap: 0.65rem; margin-top: 1.1rem;
    padding: 0.5rem 0.2rem; cursor: pointer; min-height: 3rem;
  }
  .checkbox input { width: 1.5rem; height: 1.5rem; accent-color: #2c5cc9; flex-shrink: 0; }
  .checkbox label { margin: 0; font-size: 1.05rem; color: #e6e6e6; cursor: pointer; }
  button.primary {
    margin-top: 1.4rem; width: 100%; padding: 1rem; border: none; border-radius: 10px;
    min-height: 3.6rem;
    background: #2c5cc9; color: white; font-size: 1.15rem; font-weight: 600; cursor: pointer;
  }
  button.primary:disabled { background: #33363b; color: #777; cursor: not-allowed; }
  .controls { display: flex; gap: 0.75rem; margin: 1.3rem 0; }
  .controls button {
    flex: 1; padding: 0.9rem; border-radius: 10px; border: 1px solid #2a2d33; cursor: pointer;
    font-weight: 600; font-size: 1.02rem; min-height: 3.4rem;
  }
  #stopBtn { background: #5c2020; color: #ffb3b3; }
  #killBtn { background: #1d2025; color: #ff8a8a; }
  #stopBtn:disabled, #killBtn:disabled { opacity: 0.4; cursor: not-allowed; }
  #log {
    margin-top: 1rem; background: #0b0c0e; border: 1px solid #2a2d33; border-radius: 8px;
    padding: 0.9rem; height: 260px; overflow-y: auto; font-family: ui-monospace, monospace;
    font-size: 0.9rem; white-space: pre-wrap; color: #b8c4d0;
  }
  .err { color: #ff8a8a; font-size: 0.95rem; margin-top: 0.6rem; min-height: 1.1em; }
</style>
</head>
<body>
<header>
  <h1>ob15 control panel</h1>
  <span id="statusPill">idle</span>
</header>
<main>
  <p style="color:#8a9099; font-size:0.95rem; margin:0 0 1.2rem;">
    Start a session here, then put on the headset to teleoperate — VR controller buttons
    (rerecord episode, exit early, stop, toggle intervention) control the session once it's
    running. This page works standalone in the Quest Browser app; it's separate from the
    XLeVR teleop scene.
  </p>
  <div class="tabs">
    <div class="tab active" data-tab="record">Record dataset</div>
    <div class="tab" data-tab="inference">Inference</div>
    <div class="tab" data-tab="dagger">DAgger rollout</div>
  </div>

  <div class="panel active" id="panel-record">
    <label>Dataset repo id</label>
    <input type="text" id="r_repo_id" placeholder="myuser/ob15_pick_place">
    <label>Task description</label>
    <input type="text" id="r_single_task" placeholder="Pick the block and drop it in the box">
    <div class="row">
      <div><label>Teleop</label><input type="text" id="r_teleop_type" value="xlerobot_vr"></div>
      <div><label>FPS</label><input type="number" id="r_fps" value="30"></div>
    </div>
    <div class="row">
      <div><label># episodes</label><input type="number" id="r_num_episodes" value="50"></div>
      <div><label>Episode time (s)</label><input type="number" id="r_episode_time_s" value="180"></div>
      <div><label>Reset time (s)</label><input type="number" id="r_reset_time_s" value="60"></div>
    </div>
    <div class="checkbox"><input type="checkbox" id="r_push_to_hub" checked><label for="r_push_to_hub">Push to hub</label></div>
    <div class="checkbox"><input type="checkbox" id="r_resume"><label for="r_resume">Resume existing dataset</label></div>
    <button class="primary" id="r_start">Start recording</button>
  </div>

  <div class="panel" id="panel-inference">
    <label>Policy path (local dir or hub id)</label>
    <input type="text" id="i_policy_path" placeholder="myuser/smolvla_ob15_pick_place">
    <div class="row">
      <div><label>Duration (s, 0 = infinite)</label><input type="number" id="i_duration" value="0"></div>
      <div><label>FPS</label><input type="number" id="i_fps" value="30"></div>
    </div>
    <button class="primary" id="i_start">Start inference</button>
  </div>

  <div class="panel" id="panel-dagger">
    <label>Policy path (local dir or hub id)</label>
    <input type="text" id="d_policy_path" placeholder="myuser/smolvla_ob15_pick_place">
    <label>Dataset repo id (for collected corrections)</label>
    <input type="text" id="d_repo_id" placeholder="myuser/ob15_dagger_round1">
    <label>Task description</label>
    <input type="text" id="d_single_task" placeholder="Pick the block and drop it in the box">
    <div class="row">
      <div><label>Teleop</label><input type="text" id="d_teleop_type" value="xlerobot_vr"></div>
      <div><label>Input device</label><input type="text" id="d_input_device" value="teleop"></div>
    </div>
    <div class="row">
      <div><label># correction episodes</label><input type="number" id="d_num_episodes" value="50"></div>
      <div><label>Duration (s, 0 = infinite)</label><input type="number" id="d_duration" value="0"></div>
      <div><label>FPS</label><input type="number" id="d_fps" value="30"></div>
    </div>
    <div class="checkbox"><input type="checkbox" id="d_record_autonomous"><label for="d_record_autonomous">Also record autonomous frames</label></div>
    <div class="checkbox"><input type="checkbox" id="d_push_to_hub" checked><label for="d_push_to_hub">Push to hub</label></div>
    <button class="primary" id="d_start">Start DAgger rollout</button>
  </div>

  <div class="err" id="errBox"></div>

  <div class="controls">
    <button id="stopBtn">Stop (graceful)</button>
    <button id="killBtn">Force stop</button>
  </div>

  <div id="log">no job running</div>
</main>

<script>
const tabs = document.querySelectorAll(".tab");
tabs.forEach(t => t.addEventListener("click", () => {
  tabs.forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  t.classList.add("active");
  document.getElementById("panel-" + t.dataset.tab).classList.add("active");
}));

const errBox = document.getElementById("errBox");
const logBox = document.getElementById("log");
const pill = document.getElementById("statusPill");
const stopBtn = document.getElementById("stopBtn");
const killBtn = document.getElementById("killBtn");

async function postJSON(url, body) {
  errBox.textContent = "";
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    errBox.textContent = detail.detail || ("Request failed: " + res.status);
    throw new Error(detail.detail || String(res.status));
  }
  return res.json();
}

document.getElementById("r_start").addEventListener("click", () => postJSON("/api/record", {
  repo_id: document.getElementById("r_repo_id").value,
  single_task: document.getElementById("r_single_task").value,
  teleop_type: document.getElementById("r_teleop_type").value,
  fps: Number(document.getElementById("r_fps").value),
  num_episodes: Number(document.getElementById("r_num_episodes").value),
  episode_time_s: Number(document.getElementById("r_episode_time_s").value),
  reset_time_s: Number(document.getElementById("r_reset_time_s").value),
  push_to_hub: document.getElementById("r_push_to_hub").checked,
  resume: document.getElementById("r_resume").checked,
}).catch(() => {}));

document.getElementById("i_start").addEventListener("click", () => postJSON("/api/inference", {
  policy_path: document.getElementById("i_policy_path").value,
  duration: Number(document.getElementById("i_duration").value),
  fps: Number(document.getElementById("i_fps").value),
}).catch(() => {}));

document.getElementById("d_start").addEventListener("click", () => postJSON("/api/dagger", {
  policy_path: document.getElementById("d_policy_path").value,
  repo_id: document.getElementById("d_repo_id").value,
  single_task: document.getElementById("d_single_task").value,
  teleop_type: document.getElementById("d_teleop_type").value,
  input_device: document.getElementById("d_input_device").value,
  num_episodes: Number(document.getElementById("d_num_episodes").value),
  record_autonomous: document.getElementById("d_record_autonomous").checked,
  duration: Number(document.getElementById("d_duration").value),
  fps: Number(document.getElementById("d_fps").value),
  push_to_hub: document.getElementById("d_push_to_hub").checked,
}).catch(() => {}));

stopBtn.addEventListener("click", () => postJSON("/api/stop", {}).catch(() => {}));
killBtn.addEventListener("click", () => postJSON("/api/kill", {}).catch(() => {}));

async function poll() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.running) {
      pill.textContent = "running: " + data.mode + " (pid " + data.pid + ")";
      pill.classList.add("running");
      stopBtn.disabled = false;
      killBtn.disabled = false;
    } else {
      pill.textContent = data.mode ? (data.mode + " exited (code " + data.returncode + ")") : "idle";
      pill.classList.remove("running");
      stopBtn.disabled = true;
      killBtn.disabled = true;
    }
    if (data.log && data.log.length) {
      logBox.textContent = data.log.join("\\n");
      logBox.scrollTop = logBox.scrollHeight;
    } else if (!data.mode) {
      logBox.textContent = "no job running";
    }
  } catch (e) {
    // ignore transient poll failures
  }
}
poll();
setInterval(poll, 1500);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Browser control panel for recording/inference/DAgger on ob15.")
    ap.add_argument("--host", default="0.0.0.0", help="Interface to bind to (0.0.0.0 = all interfaces).")
    ap.add_argument("--port", type=int, default=8420, help="Port to listen on.")
    args = ap.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ob15 web control panel on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
