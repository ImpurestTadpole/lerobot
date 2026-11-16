# Good-view JEPA experiments on VLA-JEPA

Code for the plan "Good-View JEPAs for Robotics" (2026-11-15), with fixes from its review.

| File | Phase | What it does |
|---|---|---|
| `goodview_jepa.py` | 1 (E1, E2) | ViT-S JEPA on robot frames, good views (two cameras, same step) or bad views (two crops of one frame) |
| `align_eval.py` | 1 (E1, E2) | Cross-seed and cross-camera alignment, plus a ridge probe to `observation.state` |
| `camera_dropout.py` | 3 (E5) | Black out one camera at test time |
| `src/lerobot/policies/goodview_jepa/` | 2 (E3, E4), 3 (E5) | The `goodview_jepa` policy: VLA-JEPA plus `cross_view_mode`, `video_frame_stride`, `camera_blank_prob` |

## Things the original plan got wrong

1. **`freeze_qwen=true` turns the world model off** (`VLAJEPAConfig.__post_init__`). A fine-tune
   with it trains the action head only, with no JEPA loss. Leave it off for every Phase 2 run and
   for any run where the world model should matter.
2. **The V-JEPA 2 encoder is frozen.** `cross_view_mode` changes what the predictor (and Qwen's
   action tokens) must predict, not the encoder's representation. Phase 2 tests a training
   target, not the paper's claim about representations.
3. **Variant B (`same_time`) leaves the action tokens nothing to predict**, so it weakens the
   latent-action signal that VLA-JEPA is built on. Treat C (`future`) as the main candidate and B
   as a control.
4. **`num_video_frames` is not the horizon.** Frames are consecutive (`0..7`), the encoder groups
   them in pairs, and the predictor always looks one group (2 raw frames) ahead. Raising
   `num_video_frames` lengthens the clip and changes the action-token count (the checkpoint was
   pretrained with 8). Vary `video_frame_stride` instead: stride k means 2k raw frames ahead.
5. **The pretrained checkpoint has no wrist camera.** `lerobot/VLA-JEPA-Pretrain` was trained on
   `exterior_1_left` and `exterior_2_left` with 2 world-model views. View indices for
   `cross_view_source`/`cross_view_dest` follow your dataset's `image_features` order; print it first.
6. **Within one encoder, cross-camera alignment favours good views by design**, because that
   encoder is trained to match the two cameras. For Gate 1, use the cross-seed alignment and the
   state probe from `align_eval.py`.
7. **`--eval.n_episodes` is per task** (LIBERO: 40 tasks). With 10, the standard error at 97% is
   about 0.85 points over all 400 episodes. That is fine for Gate 0, too noisy for 1-point
   differences in E3. Use 50.

## Checked on local OB15 data (CPU, 2026-11-15)

On `Odog16/ob15_packing_box` (head, left_wrist and right_wrist at 480x640 in AV1, 30 fps, 15-d state):

- `goodview_jepa.py` trains on both cameras and saves checkpoints. With the original plain output
  head, the embeddings collapsed (std 0.24 -> 0.008 in 30 steps); the BatchNorm projector now in the
  script holds them at 0.18-0.50. SIGReg scales with batch size, so use a large batch.
- `align_eval.py` runs end to end. Use at least 20 held-out episodes: with 10 fit episodes the
  fit-mean baseline R^2 was -0.03, and probes on 40-step encoders scored 0.07-0.15, about the same as
  an untrained encoder (0.15). Short runs say nothing about good vs bad views.
- The VLA-JEPA world model, with the real V-JEPA 2 encoder on real clips: all three
  `cross_view_mode` values give finite losses with gradients reaching the predictor, at
  `video_frame_stride` 1 and 4. `image_features` order is head, left_wrist, right_wrist; the world
  model keeps the first two, so source 0 = head and dest 1 = left_wrist.
- Raw token-level L1 between V-JEPA targets is dominated by each token's temporal position (the B-C
  gap was 2.18 at stride 1 and 2.19 at stride 4), so it can't show how similar the B and C targets
  are. Settle that with the E3/E4 runs.

## Phase 1

```bash
# Hold out episodes for evaluation (here 0-39 train, 40-47 eval).
for s in 0 1 2; do
  for m in good bad; do
    python examples/goodview_jepa/goodview_jepa.py --repo_id ORG/DATA --episodes $(seq 0 39) \
      --cams observation.images.wrist observation.images.top --mode $m --seed $s --out ${m}_$s.pt
  done
done
python examples/goodview_jepa/align_eval.py --repo_id ORG/DATA --episodes $(seq 40 47) \
  --cams observation.images.wrist observation.images.top \
  --encoders good_0.pt good_1.pt good_2.pt bad_0.pt bad_1.pt bad_2.pt
```

Gate 1: good views higher than bad on cross-seed alignment, with an equal or better state probe.

## Phase 2

`goodview_jepa` is its own policy type, built on VLA-JEPA (`vla_jepa` itself is unchanged). It uses
the same weights, so convert the pretrained VLA-JEPA checkpoint once; `--policy.path` then loads it
as `goodview_jepa`. With `cross_view_mode=none` it trains exactly like VLA-JEPA (tested), so variant
A runs on the same code path as B and C.

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as M; print(M('HuggingFaceVLA/libero').camera_keys)"

python -m lerobot.policies.goodview_jepa.convert_from_vla_jepa \
  --src lerobot/VLA-JEPA-Pretrain --dst outputs/goodview_jepa_pretrain

# E3: same seed, steps and data; only the target changes. No freeze_qwen (it turns the world model
# off, and goodview_jepa refuses it with a cross-view mode).
for MODE in none same_time future; do
  lerobot-train --policy.path=outputs/goodview_jepa_pretrain --policy.repo_id=ORG/goodview_jepa_$MODE \
    --policy.cross_view_mode=$MODE --policy.cross_view_source=0 --policy.cross_view_dest=1 \
    --dataset.repo_id=HuggingFaceVLA/libero --steps=30000 --seed=0
  lerobot-eval --policy.path=ORG/goodview_jepa_$MODE --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.n_episodes=50 --eval.batch_size=5
done

# E4: horizon. Stride k = 2k raw frames ahead per prediction; keep num_video_frames=8.
for K in 1 2 4 8; do
  lerobot-train --policy.path=outputs/goodview_jepa_pretrain ... --policy.video_frame_stride=$K ...
done
```

Larger strides reach past the end of short episodes more often; LeRobot pads those frames by
repeating the last frame.

## Phase 3 (E5)

Train every policy in the comparison with the same `--policy.camera_blank_prob` (for example 0.1),
then black out a camera at test time with `drop_camera` from `camera_dropout.py`. Without
training-time blanking, report the condition as "camera blacked out, untrained", not "camera
removed".
