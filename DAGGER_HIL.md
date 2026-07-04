# DAgger & Human-in-the-Loop for XLerobot

How to improve a trained policy with human feedback — and turn every failure
into training data. Data conversion → [`DATA_CONVERSION.md`](./DATA_CONVERSION.md).
Training pipelines → [`COTRAINING.md`](./COTRAINING.md).

The compounding loop: **deploy → intervene → annotate → re-weight → retrain →
deploy**. Each pass improves the policy *and* concentrates the dataset on hard
states — exactly the data the next training round benefits from most.

---

## 1. DAgger with `lerobot-rollout` (any policy)

The policy runs autonomously; you take over when it goes wrong; each
correction window is saved as an episode tagged `intervention=True`. These
episodes cover exactly the failure states of the current policy with expert
corrections — far more valuable per frame than fresh demonstrations.

```bash
lerobot-rollout \
    --strategy.type=dagger \
    --strategy.input_device=teleop \
    --strategy.num_episodes=20 \
    --policy.path=outputs/train/trash_pickup_cotrain_v1/checkpoints/last/pretrained_model \
    --robot.type=xlerobot \
    --teleop.type=xlerobot_vr \
    --dataset.repo_id=Odog16/rollout_trash_pickup_dagger \
    --dataset.single_task="Pick up the trash and place it in the bin"
```

- **RIGHT A** (VR) toggles intervention: ON pauses the policy and hands
  control to VR (the teleop recalibrates its IK targets to the robot's
  current joints, so takeover is seamless); OFF resumes the policy.
- With `--strategy.input_device=keyboard` (or `pedal`), Space pauses, Tab
  toggles correction, Enter uploads. In teleop mode the keyboard stays active
  for **ESC** (stop) and upload only.
- The dataset repo id must start with `rollout_`. Add
  `--strategy.record_autonomous=true` to also record the policy's own frames
  (tagged `intervention=False`) with size-based episode rotation.
- 10–30 correction episodes per iteration is typically enough for a
  measurable success-rate jump; fine-tune the specialist on them (see
  COTRAINING.md step 5/6).

**HVLA:** the S1 runtime (`policies/hvla/s1_process.py`) has this loop built
in via its `record_dataset` / `intervention_dataset` options, plus the RLT
reward hook (keyboard R = success).

### The Quest 3 / XLeVR runtime

No separate XLeVR process is needed: `--teleop.type=xlerobot_vr` starts the
whole stack (HTTPS WebXR page on **:8443**, WSS controller stream on
**:8442**) inside the rollout process. Put on the Quest 3, open
`https://<rollout-host-ip>:8443` in its browser, accept the self-signed cert,
and enter VR. The XLeVR checkout is auto-located (`~/XLeRobot/XLeVR`);
override with the `XLEVR_PATH` env var or `--teleop.xlevr_path=...`. Don't run
the standalone XLeVR app at the same time — the ports clash.

Async/split deployment: run `lerobot-rollout` on the GPU PC with
`--robot.type=xlerobot_client` while the Jetson runs `xlerobot_host`; the
XLeVR servers then live on the PC and the Quest connects to the PC's IP.

---

## 2. Reward-weighted reuse: RA-BC + SARM

Not all demos are equal. SARM subtask/progress annotation
(`src/lerobot/data_processing/sarm_annotations/`) scores episodes; RA-BC
(`--use_rabc=true`, auto-migrated to `--sample_weighting.type=rabc`)
up-weights smooth successful segments and down-weights hesitation and
recovery noise during training:

```bash
lerobot-train \
    --dataset.repo_id=Odog16/trash_pickup \
    --policy.type=smolvla \
    --policy.pretrained_path=<generalist checkpoint> \
    --use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01 \
    --batch_size=16 --steps=20000
```

This *creates more usable data out of what you already have* — old
mixed-quality recordings keep contributing their good segments instead of
being deleted.

---

## 3. Online RL with human oversight: HIL-SERL (`src/lerobot/rl/`)

For one scoped task that needs near-perfect reliability. The stack:
`gym_manipulator` wraps the robot as a gym env with intervention support and
ROI cropping; a **reward classifier** trained from your success/failure
labels provides the sparse reward; `rl/actor.py` collects transitions on the
robot (human interventions enter the replay buffer as high-quality actions)
while `rl/learner.py` trains SAC off-policy on the GPU machine, warm-started
from your demonstrations.

**XLerobot caveat:** the RL env is single-arm — it assumes one motor bus
(`robot.bus`), a 3+1-dim EE action space, and one kinematic chain. It will
not run against the full 18-DOF robot. Scope it to **one 6-DOF arm** (own
robot config on one bus, base locked, wrist + overhead camera); that is also
where HIL-SERL works best (~1–3 h of robot time to near-100 % on a scoped
task). Every transition collected is also valid BC data for the next merge.

---

## 4. Which to use when

| Situation | Tool |
|---|---|
| New task, no data | Record 50+ VR demos → co-train (COTRAINING.md) |
| Policy ~60–80 % success, fails in specific states | DAgger corrections (§1) → RA-BC fine-tune |
| Plenty of mixed-quality demos | SARM annotate + RA-BC (§2) |
| One critical task needs near-perfect reliability | HIL-SERL on a single arm (§3) |
| Many tasks, plateauing generalist | Add converted external data, re-merge, redo the generalist |
