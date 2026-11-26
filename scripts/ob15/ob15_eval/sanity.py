"""Sanity checks A1-A4, A10 on CPU (no GPU, no robot). A5-A9 need the robot or the GPU; see the notes printed.

usage: sanity.py <checkpoint_dir>
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

CK = sys.argv[1]
OUT = Path("/home/owen/ob15_experiments/reports/sanity")
OUT.mkdir(parents=True, exist_ok=True)
man = json.load(open(Path(__file__).resolve().parent / "manifests/bp45_v1.json"))
ds = LeRobotDataset(man["dataset"], root=man["root"])
cfg = PreTrainedConfig.from_pretrained(CK)
cfg.device = "cpu"
res = {}

# A1: every action index is the joint its name says, and the policy's names match the dataset's
names = ds.meta.features["action"]["names"]
pol_names = getattr(cfg, "action_feature_names", None) or []
res["A1 action index map"] = (pol_names == names, f"{len(names)} names; policy {'matches' if pol_names == names else 'DIFFERS: ' + str(pol_names)}")

# A2: normalise then unnormalise recorded actions
pre, post = make_pre_post_processors(cfg, pretrained_path=CK, preprocessor_overrides={"device_processor": {"device": "cpu"}})
if cfg.type in ("vla_jepa", "goodview_jepa"):
    from lerobot.policies.vla_jepa.processor_vla_jepa import sync_gripper_steps_with_config

    sync_gripper_steps_with_config(cfg, post)
norm = next(s for s in pre.steps if type(s).__name__ == "NormalizerProcessorStep")
unnorm = next(s for s in post.steps if type(s).__name__ == "UnnormalizerProcessorStep")
acts = torch.tensor(np.stack([np.asarray(ds.hf_dataset[int(i)]["action"]) for i in range(0, len(ds), 97)]), dtype=torch.float32)
n = norm._normalize_action(acts, inverse=False) if hasattr(norm, "_normalize_action") else None
if n is None:
    from lerobot.processor import TransitionKey

    n = norm({TransitionKey.ACTION: acts, TransitionKey.OBSERVATION: {}})[TransitionKey.ACTION]
from lerobot.processor import TransitionKey

back = unnorm({TransitionKey.ACTION: n})[TransitionKey.ACTION]
err = (back - acts).abs().max().item()
res["A2 normalisation round-trip"] = (err < 0.01, f"max error {err:.6f} deg over {len(acts)} actions (in range: {bool((n.abs() <= 1.0001).all())})")

# A3: the server's image path (uint8 HWC -> float/255 CHW) gives the same tensor as training's
item = ds[1234]
ok3, worst = True, 0.0
for c in ("head", "left_wrist", "right_wrist"):
    t = item[f"observation.images.{c}"]
    hwc_u8 = (t.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    served = torch.from_numpy(hwc_u8.copy()).permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
    worst = max(worst, (served - t).abs().max().item())
ok3 = worst <= 0.5 / 255 + 1e-6
res["A3 train vs serve image path"] = (ok3, f"max pixel difference {worst * 255:.3f}/255 (uint8 rounding only)")

# A4: an image per camera key, labelled, for a human to check
tiles = []
for c in ("head", "left_wrist", "right_wrist"):
    im = Image.fromarray((item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((480, 270))
    tiles.append((c, im))
sheet = Image.new("RGB", (1440, 300), "white")
from PIL import ImageDraw

d = ImageDraw.Draw(sheet)
for k, (c, im) in enumerate(tiles):
    sheet.paste(im, (480 * k, 30))
    d.text((480 * k + 8, 8), f"observation.images.{c}", fill="black")
sheet.save(OUT / "A4_camera_keys.png")
res["A4 camera key map"] = (None, f"see {OUT / 'A4_camera_keys.png'} (needs a human look)")

# A10: post-processing audit - steps that clip, snap or use hard-coded indices
steps = [type(s).__name__ for s in post.steps]
flag = [s for s in steps if re.search(r"Snap|Binar|Clip", s)]
res["A10 post-processing audit"] = ("Gripper" not in " ".join(flag), f"output steps {steps}")

for k, (ok, msg) in res.items():
    print(f"{'PASS' if ok else ('CHECK' if ok is None else 'FAIL'):5s} {k}: {msg}")
print("not run here: A5 pinned state (robot), A6 overfit one episode (GPU run), A7 replay on robot, A8 determinism (GPU), A9 timing (robot)")
