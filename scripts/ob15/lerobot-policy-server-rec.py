"""lerobot-policy-server plus a recorder and optional stress-test perturbations.

Recorder: every REC_EVERY-th prediction saves the exact observation and answer to REC_DIR (one folder per
server start, at most REC_MAX files). A recording problem never affects serving.
PERTURB=<condition> changes every observation before the policy sees it, with the same code as the offline
harness (ob15_eval/eval_run.py): freeze_head / freeze_left_wrist / freeze_right_wrist (that camera shows the
frame from 1.5 s earlier), shift_head, dim, warm, bright. Unset = no change. Recordings store the
perturbed observation the policy actually saw."""
import os, sys, time, json, re
import numpy as np
from lerobot.inference.policy_runner import PolicyRunner
EVERY, MAX = int(os.environ.get("REC_EVERY", "3")), int(os.environ.get("REC_MAX", "400"))
cfg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--config_path=")), "server")
DIR = os.path.join(os.environ.get("REC_DIR", "/home/owen/ob15_experiments/recordings"), time.strftime("%m%d_%H%M%S_") + re.sub(r"^server_blockpass_|\.yaml$", "", os.path.basename(cfg)))
n = saved = 0
_orig = PolicyRunner.predict
def predict(self, observation, **kw):
    global n, saved
    out = _orig(self, observation, **kw)
    try:
        n += 1
        if n % EVERY == 1 and saved < MAX:
            os.makedirs(DIR, exist_ok=True)
            np.savez(os.path.join(DIR, f"{n:05d}.npz"), chunk=out.canonical_actions.numpy(), task=observation.task, t=time.time(), **{k.replace("/", "_"): v for k, v in observation.features.items()})
            saved += 1
    except Exception as e:
        print("recorder:", type(e).__name__, e, flush=True)
    return out
PERTURB = os.environ.get("PERTURB", "").strip()
if PERTURB:
    import collections
    import dataclasses

    import torch

    sys.path.insert(0, "/home/owen/ob15_experiments/ob15_eval")
    from eval_run import CONDITIONS, FREEZE_S, perturb as _perturb

    if PERTURB not in CONDITIONS[1:]:
        raise SystemExit(f"PERTURB must be one of {CONDITIONS[1:]}, got {PERTURB!r}")
    _history = collections.defaultdict(collections.deque)  # camera feature -> (time, frame)

    def _cam(name):
        return name.rsplit(".", 1)[-1]

    def _apply(features, now):
        out = dict(features)
        for name, arr in features.items():
            if getattr(arr, "ndim", 0) != 3:
                continue  # state
            cam = _cam(name)
            # served names may be renamed (base_0_rgb, exterior_1_left, image2...): map back by semantics order
            if PERTURB.startswith("freeze_"):
                hist = _history[name]
                hist.append((now, arr))
                while hist and now - hist[0][0] > FREEZE_S + 0.5:
                    hist.popleft()
                if CAM_OF.get(name) == PERTURB[len("freeze_"):]:
                    old = [f for t, f in hist if now - t >= FREEZE_S]
                    out[name] = old[-1] if old else hist[0][1]
            elif PERTURB == "shift_head" and CAM_OF.get(name) != "head":
                continue
            else:
                t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div(255)
                out[name] = (_perturb("shift" if PERTURB == "shift_head" else PERTURB, t) * 255).round().byte().permute(1, 2, 0).numpy()
        return out

    def _perturbed_predict(self, observation, **kw):
        observation = dataclasses.replace(observation, features=_apply(observation.features, observation.capture_time))
        return predict(self, observation, **kw)

    PolicyRunner.predict = _perturbed_predict
else:
    PolicyRunner.predict = predict

# served camera feature name -> robot camera, read from the server config's semantics (ob15-rgb-<camera>-v1)
CAM_OF = {}
try:
    import yaml

    for f in yaml.safe_load(open(cfg))["features"]:
        sem = str(f.get("semantics", ""))
        if sem.startswith("ob15-rgb-"):
            CAM_OF[f["name"]] = sem[len("ob15-rgb-") : -len("-v1")].replace("-", "_")
except Exception as e:  # noqa: BLE001
    print("perturb: could not read camera names from", cfg, e, flush=True)
from lerobot.scripts.lerobot_policy_server import main
if __name__ == "__main__":
    print("recording to", DIR, "| perturbation:", PERTURB or "none", "| cameras:", CAM_OF, flush=True); sys.exit(main())
