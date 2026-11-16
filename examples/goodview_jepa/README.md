# Good-View JEPA examples

Scripts for studying whether "good views" help JEPA-style representations on robot data. A good view
pair is two observations of the same world state, here two cameras at the same timestep. A bad view
pair is two views of one observation, here two crops of one camera frame. See the
[Good-View JEPA policy docs](https://huggingface.co/docs/lerobot/main/en/goodview_jepa) for the policy that
uses this idea inside VLA-JEPA.

| File                | What it does                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `train_encoder.py`  | Trains a ViT-S JEPA (invariance + SIGReg) on two cameras with good or bad views                      |
| `align_eval.py`     | Cross-camera and cross-seed alignment, and a ridge probe from frozen features to `observation.state` |
| `camera_dropout.py` | `drop_camera`: black out one camera in an observation at test time                                   |

The encoder study runs on one GPU and needs a `LeRobotDataset` with at least two cameras. Actions
are not used.

## 1. Train encoders

Hold out episodes for evaluation, and train each mode with several seeds: the spread between seeds
can be as large as the effect.

```bash
for s in 0 1 2; do
  for m in good bad; do
    python examples/goodview_jepa/train_encoder.py \
      --repo_id your_org/your_dataset --episodes $(seq 0 39) \
      --cams observation.images.wrist observation.images.top \
      --mode $m --seed $s --out ${m}_$s.pt
  done
done
```

- `good`: the positives are the two cameras at the same timestep.
- `bad`: the positives are two crops of one frame. The camera is picked per sample, so both modes
  see the same images and differ only in how they are paired.
- The loss is computed on a BatchNorm MLP projector. Without it, SIGReg at `--lam 0.02` did not
  prevent collapse on real robot frames. SIGReg scales with batch size, so keep the batch large
  (default 256). Use `--max_steps` for a quick smoke run.

## 2. Evaluate

```bash
python examples/goodview_jepa/align_eval.py \
  --repo_id your_org/your_dataset --episodes $(seq 40 59) \
  --cams observation.images.wrist observation.images.top \
  --encoders good_0.pt good_1.pt good_2.pt bad_0.pt bad_1.pt bad_2.pt
```

Frames are spread evenly over the held-out episodes and split by episode into a fit half and a test
half. Whitening, rotations and the probe are fitted on the fit half only. The output has:

| Metric                       | Meaning                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `cross_camera` (per encoder) | Camera 1 vs camera 2 through the same encoder. The good-view encoder is trained for this, so it favours good views by design |
| `cross_seed`                 | Two encoders of the same mode, different seeds: do independent runs converge to the same space?                              |
| `state_probe_R2`             | Ridge probe to the robot state (penalty picked by cross-validation over episodes). Neither mode trains for it                |
| `state_mean_baseline_R2`     | Predicting the fit-half mean state: the floor a useful probe must beat                                                       |
| `random_init`                | An untrained encoder: trained encoders should beat it                                                                        |

To compare good and bad views, rely on `cross_seed` and `state_probe_R2` rather than `cross_camera`.
Use at least 20 held-out episodes; with fewer, episode-to-episode differences dominate the probe.

## 3. Test with a blacked-out camera

```python
from camera_dropout import drop_camera

action = policy.select_action(preprocessor(drop_camera(observation, "observation.images.top")))
```

A black frame is not the same as a missing camera, and a policy that never saw one in training is
being tested off-distribution. Train every policy in a comparison with the same
`--policy.camera_blank_prob` (supported by `goodview_jepa`), or report the condition as "camera
blacked out, no blanking in training" for all of them.
