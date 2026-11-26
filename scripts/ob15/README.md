# OB15 block-passing scripts

Working copies of the scripts used on PC1 (training/serving, RTX 3090 Ti) and the Jetson (OB15 robot) for the
block-passing experiments. Paths point at the PC1 layout (`/home/owen/...`); adjust them for another machine.
The live copies stay in `~/ob15_experiments/` on PC1.

| File | Runs on | Purpose |
|---|---|---|
| `serve.sh` | PC1 | Start one policy server (`serve.sh jepa_future_c30_r1`); `PERTURB=<condition>` for stress tests |
| `lerobot-policy-server-rec.py` | PC1 | `lerobot-policy-server` plus an observation recorder and the `PERTURB` option |
| `make_server_config.py` | PC1 | Server YAML from a checkpoint |
| `run_blockpass.sh`, `run_dagger_blockpass.sh` | Jetson | Rollout / DAgger against the PC1 server (`POLICY_ENDPOINT` overrides the server address) |
| `offline_eval.py` | PC1 | Open-loop error of a checkpoint on recorded episodes |
| `ob15_eval/eval_run.py` | PC1 | Held-out evaluation: error, chunk jump, camera freeze / shift / lighting stress tests |
| `ob15_eval/sanity.py` | PC1 | Sanity checks A1-A4, A10 |
| `ob15_eval/rtc_offline.py` | PC1 | Chunk-switch jump with and without guided RTC |
| `ob15_eval/queue.sh`, `runs.json`, `report.py` | PC1 | Remote-week training queue, run list and reports |
| `ob15_eval/manifests/bp45_v1.json` | - | The 45/5 held-out split of `Odog16/block_passing_gv` |

Perturbation conditions (offline and server): `freeze_head`, `freeze_left_wrist`, `freeze_right_wrist` (that
camera shows the frame from 1.5 s earlier), `shift_head` (3 deg, 4%/2% shift, 1.03 zoom), `dim`, `warm`, `bright`.
