"""Offline evaluation of one checkpoint on the bp45_v1 split.

Per frame: predict the action chunk, compare with the recorded actions (degrees, per joint).
Conditions on held-out frames: clean, camera freeze (each camera shows the frame from FREEZE_S earlier),
head-camera shift, dim / warm / bright lighting (same change on every camera). Plus the chunk-switch
jump (prediction at t+STRIDE vs the prediction made at t, at the switch step).

usage: eval_run.py <checkpoint_dir> <run_id> [--tag NAME] [--frames N]
Writes ob15_eval/results/<run_id>__<tag>.csv (one row per model, condition, split, episode, frame)
and prints a summary; appends the summary to ob15_eval/results/summary.csv.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

HERE = Path(__file__).resolve().parent
FPS, FREEZE_S, STRIDE = 30, 1.5, 15
CAMS = ("head", "left_wrist", "right_wrist")


def load_policy(ckpt: str, device: str = "cuda"):
    train_cfg = json.load(open(Path(ckpt) / "train_config.json"))
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    from lerobot.policies.factory import has_peft_adapter_config

    if getattr(cfg, "use_peft", False) and has_peft_adapter_config(ckpt):
        from peft import PeftConfig, PeftModel

        from lerobot.remote_inference.deployment import resolve_artifact

        adapter = PeftConfig.from_pretrained(ckpt)
        base = resolve_artifact(adapter.base_model_name_or_path, adapter.revision)
        policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg)
        policy = PeftModel.from_pretrained(policy, ckpt, config=adapter)
    else:
        policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy = policy.to(device).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": train_cfg.get("rename_map") or {}},
        },
    )
    if cfg.type in ("vla_jepa", "goodview_jepa"):
        # Same as serving: gripper snap/binarize only as the config says (old checkpoints carry stale steps).
        from lerobot.policies.vla_jepa.processor_vla_jepa import sync_gripper_steps_with_config

        sync_gripper_steps_with_config(cfg, post)
    horizon = getattr(cfg, "n_action_steps", None) or cfg.chunk_size
    return policy, pre, post, horizon, cfg


def perturb(name: str, img: torch.Tensor) -> torch.Tensor:
    """img: [C, H, W] float in [0, 1]."""
    if name == "dim":
        img = TF.adjust_gamma(TF.adjust_brightness(img, 0.5), 1.4)
    elif name == "warm":
        img = TF.adjust_hue(TF.adjust_brightness(img, 0.9), 0.05)
    elif name == "bright":
        img = TF.adjust_gamma(TF.adjust_contrast(TF.adjust_brightness(img, 1.5), 1.1), 0.8)
    elif name == "shift":
        h, w = img.shape[-2:]
        img = TF.affine(img, angle=3.0, translate=[int(0.04 * w), int(0.02 * h)], scale=1.03, shear=[0.0, 0.0])
    return img.clamp(0, 1)


CONDITIONS = ["clean", "freeze_head", "freeze_left_wrist", "freeze_right_wrist", "shift_head", "dim", "warm", "bright"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("run_id")
    ap.add_argument("--tag", default="last")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--train_eps", type=int, default=8)
    ap.add_argument("--manifest", default=str(HERE / "manifests/bp45_v1.json"))
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    t0 = time.time()
    policy, pre, post, H, cfg = load_policy(a.ckpt)
    ds = LeRobotDataset(man["dataset"], root=man["root"], delta_timestamps={"action": [i / FPS for i in range(H)]})
    names = ds.meta.features["action"]["names"]
    arm = [j for j, n in enumerate(names) if "gripper" not in n]
    keys = [f"observation.images.{c}" for c in CAMS]
    rng = np.random.default_rng(0)
    train_eps = sorted(rng.choice(man["train"], a.train_eps, replace=False).tolist())

    def predict(item, images=None, seed=0):
        obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.") and not k.endswith("_is_pad")}
        if images:
            for k, v in images.items():
                obs[k] = v.unsqueeze(0)
        obs["task"] = [item["task"]]
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(seed)
        with torch.no_grad():
            out = post(policy.predict_action_chunk(pre(obs)))
        return out.squeeze(0).float().cpu().numpy()[:H, : len(names)]

    rows = []
    out_csv = HERE / "results" / f"{a.run_id}__{a.tag}.csv"
    out_csv.parent.mkdir(exist_ok=True)

    def frames_of(ep):
        s, e = int(ds.meta.episodes["dataset_from_index"][ep]), int(ds.meta.episodes["dataset_to_index"][ep])
        lo = s + int(FREEZE_S * FPS)  # leave room for the freeze condition
        return s, [int(x) for x in np.linspace(lo, e - H - STRIDE - 1, a.frames).round()]

    def record(split, ep, idx, cond, pred, gt, st):
        err = np.abs(pred - gt)
        rows.append(
            dict(run_id=a.run_id, tag=a.tag, split=split, episode=ep, frame=idx, condition=cond,
                 first_step_err=float(err[0][arm].mean()), chunk_err=float(err[:, arm].mean()),
                 gripper_err=float(np.delete(err, arm, axis=1).mean()),
                 hold_err=float(np.abs(st[None, : len(names)] - gt)[:, arm].mean()),
                 **{f"err_{n}": float(err[:, j].mean()) for j, n in enumerate(names)})
        )

    for split, eps in (("heldout", man["heldout"]), ("train", train_eps)):
        for ep in eps:
            _, idxs = frames_of(ep)
            for idx in idxs:
                item = ds[idx]
                gt, st = item["action"].numpy()[:, : len(names)], item["observation.state"].numpy()
                record(split, ep, idx, "clean", predict(item), gt, st)
                if split != "heldout":
                    continue
                old = ds[idx - int(FREEZE_S * FPS)]
                for c in CAMS:
                    record(split, ep, idx, f"freeze_{c}", predict(item, {f"observation.images.{c}": old[f"observation.images.{c}"]}), gt, st)
                record(split, ep, idx, "shift_head", predict(item, {keys[0]: perturb("shift", item[keys[0]])}), gt, st)
                for light in ("dim", "warm", "bright"):
                    record(split, ep, idx, light, predict(item, {k: perturb(light, item[k]) for k in keys}), gt, st)
                # chunk-switch jump: new chunk from t+STRIDE vs old chunk's step STRIDE
                nxt = predict(ds[idx + STRIDE])
                cur = predict(item)
                rows[-1]["chunk_jump"] = float(np.abs(nxt[0] - cur[STRIDE])[arm].mean())
    with open(out_csv, "w", newline="") as f:
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("err_"), k))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def mean(sel, key):
        v = [r[key] for r in rows if sel(r) and key in r]
        return float(np.mean(v)) if v else math.nan

    clean_h = mean(lambda r: r["split"] == "heldout" and r["condition"] == "clean", "chunk_err")
    summ = dict(
        run_id=a.run_id, tag=a.tag, ckpt=a.ckpt, horizon=H,
        heldout_first=mean(lambda r: r["split"] == "heldout" and r["condition"] == "clean", "first_step_err"),
        heldout_chunk=clean_h,
        heldout_gripper=mean(lambda r: r["split"] == "heldout" and r["condition"] == "clean", "gripper_err"),
        heldout_hold=mean(lambda r: r["split"] == "heldout" and r["condition"] == "clean", "hold_err"),
        train_first=mean(lambda r: r["split"] == "train", "first_step_err"),
        train_chunk=mean(lambda r: r["split"] == "train", "chunk_err"),
        chunk_jump=mean(lambda r: "chunk_jump" in r, "chunk_jump"),
        **{f"drop_{c}": mean(lambda r, c=c: r["split"] == "heldout" and r["condition"] == c, "chunk_err") - clean_h
           for c in CONDITIONS[1:]},
        minutes=(time.time() - t0) / 60,
    )
    s_csv = HERE / "results" / "summary.csv"
    new = not s_csv.exists()
    with open(s_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ))
        if new:
            w.writeheader()
        w.writerow(summ)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}, indent=0))


if __name__ == "__main__":
    main()
