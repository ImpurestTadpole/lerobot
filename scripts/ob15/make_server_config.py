"""Write a lerobot-policy-server config for an OB15 block-passing checkpoint (pi0.5 or Good-View/VLA-JEPA).
usage: make_server_config.py <checkpoint_dir> <out.yaml> [--rtc]   (--rtc only for checkpoints trained with rtc_training_max_delay > 0)"""
import json, sys
ck, out, rtc = sys.argv[1], sys.argv[2], "--rtc" in sys.argv
cfg = json.load(open(f"{ck}/config.json"))
info = json.load(open("/home/owen/.cache/huggingface/lerobot/Odog16/block_passing_sarm/meta/info.json"))["features"]
STATE_NAMES = info["observation.state"]["names"]; ACTION_NAMES = info["action"]["names"]
# checkpoint camera name -> (robot camera it comes from, H, W)
CAMS = {"observation.images.base_0_rgb": ("head", 540, 960), "observation.images.exterior_1_left": ("head", 540, 960),
        "observation.images.right_wrist_0_rgb": ("right_wrist", 360, 640), "observation.images.exterior_2_left": ("right_wrist", 360, 640),
        "observation.images.left_wrist_0_rgb": ("left_wrist", 360, 640), "observation.images.left_wrist": ("left_wrist", 360, 640),
        "observation.images.image": ("head", 540, 960), "observation.images.image2": ("left_wrist", 360, 640), "observation.images.image3": ("right_wrist", 360, 640),
        "observation.images.head": ("head", 540, 960), "observation.images.right_wrist": ("right_wrist", 360, 640)}
lines = ["deployment: ob15arms", "model:", f"  repo_or_path: {ck}", "  device: cuda", "execution:",
         f"  supported_modes: [{'chunk, rtc_trained' if rtc else 'chunk'}]", "  action_fps: 30.0", "  action_deadline_s: 5.0",
         "  idle_timeout_s: 10.0", "  warmup_calls: 2"]
if rtc: lines += ["  rtc:", "    enabled: true", "    mode: trained", "    execution_horizon: 12"]
lines += ["zenoh:", "  mode: peer", "  listen_endpoints: [tcp/10.0.0.122:7447]   # use tcp/127.0.0.1:7447 for same-machine tests",
          "semantics: ob15-arms-v1", "robot_type: ob15_arms", "features:"]
items = sorted(cfg["input_features"].items(), key=lambda kv: kv[1]["type"] != "STATE")  # state first, as in the working config
for name, f in items:
    if f["type"] == "STATE":
        lines += [f"  - name: {name}", f"    shape: [{len(STATE_NAMES)}]", "    dtype: float32", f"    names: [{', '.join(STATE_NAMES)}]", "    semantics: ob15-arms-v1"]
    elif f["type"] == "VISUAL":
        cam, h, w = CAMS[name]
        lines += [f"  - name: {name}", f"    shape: [{h}, {w}, 3]", "    dtype: uint8", "    kind: rgb", f"    semantics: ob15-rgb-{cam.replace('_', '-')}-v1"]
lines += ["action_feature:", "  name: action", f"  shape: [{len(ACTION_NAMES)}]", "  dtype: float32", f"  names: [{', '.join(ACTION_NAMES)}]", "  semantics: ob15-arms-v1"]
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}: inputs {list(cfg['input_features'])} | modes {'chunk, rtc_trained' if rtc else 'chunk'}")
