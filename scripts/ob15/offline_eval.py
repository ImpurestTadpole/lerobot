"""Offline open-loop check: does each policy reproduce its own training demonstrations?

For sampled frames of training episodes, feed the recorded observation (images + state + task) through the
checkpoint's own pre/post processors, predict an action chunk, and compare it with the recorded actions.
Baseline: "hold still" (repeat the current joint positions for the whole chunk). A policy that learned the
task should beat the baseline clearly, especially on the moving joints.

usage: offline_eval.py <checkpoint_dir> <dataset_root> <repo_id> [episodes] [frames_per_episode]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

ckpt, root, repo = sys.argv[1], sys.argv[2], sys.argv[3]
n_eps = int(sys.argv[4]) if len(sys.argv) > 4 else 6
per_ep = int(sys.argv[5]) if len(sys.argv) > 5 else 10
torch.manual_seed(0)

train_cfg = json.load(open(Path(ckpt) / "train_config.json"))
rename = train_cfg.get("rename_map") or {}
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.pretrained_path = ckpt
cfg.device = "cuda"
from lerobot.policies.factory import has_peft_adapter_config  # noqa: E402
if getattr(cfg, "use_peft", False) and has_peft_adapter_config(ckpt):
    from peft import PeftConfig, PeftModel
    from lerobot.remote_inference.deployment import resolve_artifact
    adapter = PeftConfig.from_pretrained(ckpt)
    base = resolve_artifact(adapter.base_model_name_or_path, adapter.revision)
    policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg)
    policy = PeftModel.from_pretrained(policy, ckpt, config=adapter)
else:
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
policy = policy.to("cuda").eval()
pre, post = make_pre_post_processors(
    cfg,
    pretrained_path=ckpt,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": rename},
    },
)
H = getattr(cfg, "n_action_steps", None) or cfg.chunk_size
fps = 30
ds = LeRobotDataset(repo, root=root, delta_timestamps={"action": [i / fps for i in range(H)]})
names = ds.meta.features["action"]["names"]
n_total = ds.meta.total_episodes
eps = np.linspace(0, n_total - 1, n_eps).round().astype(int)

errs, base_errs, first_errs, first_bias = [], [], [], []
for ep in eps:
    start = int(ds.meta.episodes["dataset_from_index"][ep])
    end = int(ds.meta.episodes["dataset_to_index"][ep])
    for idx in np.linspace(start + 5, end - H - 1, per_ep).round().astype(int):
        item = ds[int(idx)]
        gt = item["action"].numpy()  # (H, 12)
        obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
        obs = {k: v for k, v in obs.items() if not k.endswith("_is_pad")}
        obs["task"] = [item["task"]]
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.no_grad():
            batch = pre(obs)
            chunk = policy.predict_action_chunk(batch)
            chunk = post(chunk)
        pred = chunk.squeeze(0).float().cpu().numpy()[:H, : gt.shape[1]]
        state = item["observation.state"].numpy()[: gt.shape[1]]
        errs.append(np.abs(pred - gt).mean(0))
        first_errs.append(np.abs(pred[0] - gt[0]))
        first_bias.append(pred[0] - gt[0])
        base_errs.append(np.abs(state[None] - gt).mean(0))

errs, base_errs, first_errs, first_bias = map(np.array, (errs, base_errs, first_errs, first_bias))
print(f"\n{Path(ckpt).parent.parent.parent.name if 'checkpoints' in ckpt else ckpt} | chunk {H} | "
      f"{len(errs)} samples from episodes {eps.tolist()}")
print(f"{'joint':30s} {'policy MAE':>10s} {'hold MAE':>9s} {'1st-step':>9s} {'1st bias':>9s}")
for j, n in enumerate(names):
    print(f"{n:30s} {errs[:, j].mean():10.2f} {base_errs[:, j].mean():9.2f} {first_errs[:, j].mean():9.2f} {first_bias[:, j].mean():+9.2f}")
print(f"{'MEAN':30s} {errs.mean():10.2f} {base_errs.mean():9.2f} {first_errs.mean():9.2f}")
print(f"policy beats hold-still on {(errs.mean(1) < base_errs.mean(1)).mean() * 100:.0f}% of samples")
