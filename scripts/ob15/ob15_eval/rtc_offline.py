"""Chunk-switch jump with and without guided RTC on held-out frames.

The previous chunk is predicted at t; the next request is made at t+S (S=15, a 0.5 s refill) and its
reply starts being executed `delay` steps later. Jump = |new[delay] - old[S+delay]| (arm joints, degrees).
usage: rtc_offline.py <ckpt> <device> [pairs] [delay]
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/owen/ob15_experiments/ob15_eval")
import eval_run as ER
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.rtc.configuration_rtc import RTCConfig

ck, dev = sys.argv[1], sys.argv[2]
pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
delay = int(sys.argv[4]) if len(sys.argv) > 4 else 4
S = 15
policy, pre, post, H, cfg = ER.load_policy(ck, dev)
base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
base.config.rtc_config = RTCConfig(execution_horizon=10, max_guidance_weight=10.0)
base.init_rtc_processor()
man = json.load(open("/home/owen/ob15_experiments/ob15_eval/manifests/bp45_v1.json"))
ds = LeRobotDataset(man["dataset"], root=man["root"], delta_timestamps={"action": [i / 30 for i in range(H)]})
names = ds.meta.features["action"]["names"]; arm = [j for j, n in enumerate(names) if "gripper" not in n]

def model_chunk(item, **kw):
    obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.") and not k.endswith("_is_pad")}
    obs["task"] = [item["task"]]
    torch.manual_seed(0)
    return policy.predict_action_chunk(pre(obs), **kw)

def canon(m):
    return post(m.clone()).squeeze(0).float().cpu().numpy()[:, : len(names)]

R = {"plain": [], "rtc": [], "err_plain": [], "err_rtc": []}
for ep in man["heldout"]:
    s, e = int(ds.meta.episodes["dataset_from_index"][ep]), int(ds.meta.episodes["dataset_to_index"][ep])
    for t in np.linspace(s + 10, e - H - S - 1, max(1, pairs // len(man["heldout"]) + 1)).round().astype(int)[: pairs]:
        old_m = model_chunk(ds[int(t)]); old = canon(old_m)
        nxt_item = ds[int(t) + S]; gt = nxt_item["action"].numpy()[:, : len(names)]
        new_p = canon(model_chunk(nxt_item))
        prefix = old_m[0, S:]  # unexecuted part of the previous chunk, model coordinates
        with torch.enable_grad():
            new_r = canon(model_chunk(nxt_item, inference_delay=delay, prev_chunk_left_over=prefix, execution_horizon=10))
        R["plain"].append(np.abs(new_p[delay] - old[S + delay])[arm].mean()); R["rtc"].append(np.abs(new_r[delay] - old[S + delay])[arm].mean())
        R["err_plain"].append(np.abs(new_p - gt)[:, arm].mean()); R["err_rtc"].append(np.abs(new_r - gt)[:, arm].mean())
print(json.dumps({k: round(float(np.mean(v)), 3) for k, v in R.items()} | {"pairs": len(R["plain"]), "delay": delay}))
