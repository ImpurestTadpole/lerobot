# EE-space generalist merge — resume notes

Status as of 2026-08-29. The merge ran for ~21 days, got to 30/31 sources
done + 56% through DROID, then **crashed** on a real DROID video-split bug.
Per-episode skip-on-error resilience + best-effort resume support was added
and thoroughly tested (see "2026-08-16 crash" section below — including an
important account of a bug in the first fix attempt that cost real
progress). The merge was relaunched with the fix (3rd DROID pass) — which
then turned out to have a THIRD, separate, pre-existing bug: DROID's real
camera footage was being silently zero-filled the entire pass due to a
stale duplicate `--camera-remap` CLI default drifting out of sync with the
code's actual default dict. Verified via direct pixel inspection (mean=0,
all-black), fixed (one default now, not two), and DROID restarted a 4th
time — this time verified with real, non-black camera frames before
trusting it. See "4th DROID restart" section below.

**Then, 12 days later (2026-08-29)**: found the 4th pass had reached 75%
(72,016/95,658 episodes) before crashing on a full disk. Freed ~684GB via
careful cleanup (see "5th DROID restart" section). Relaunching should have
resumed from ~72,017 — instead it rebuilt from episode 0, revealing a
FOURTH, separate, pre-existing bug: the real orchestrator always deletes
the align cache before giving the (already-built, already-tested) resume
logic a chance to see it, making resume unreachable in production the
whole time. Fixed; not yet verified end-to-end (flagged honestly, not
glossed over). This bug, not the disk-full crash itself, is what actually
destroyed the 75% of real progress. See "5th DROID restart" section below.

Meanwhile, 32 NEW source datasets (7 flex-pi + 25 xLeRobot community) were
converted in parallel for a later relaunch — see "New sources" section
below.

## 4th DROID restart (2026-08-17): silent camera zero-fill bug

While wiring up camera-remap entries for the new xLeRobot sources, found
that `co_training_utils.py`'s CLI had its OWN independently-maintained
`--camera-remap` default string, separate from the `_DEFAULT_CAMERA_REMAP`
dict — a pre-existing bug, not introduced this session. Since the CLI
always supplied a non-`None` default, `align_datasets_for_cotraining`'s own
`camera_remap is None` fallback to `_DEFAULT_CAMERA_REMAP` was never
reached through the CLI at all. The two dicts had drifted: the CLI's stale
copy was missing DROID's own camera-name mappings (`wrist_left`→
`left_wrist`, `exterior_1_left`→`head`).

**Impact**: DROID's real camera keys never matched anything in the stale
remap, so they passed through unchanged and landed outside all 3 canonical
slots — meaning `--camera-fill-mode zero` zero-filled ALL of DROID's
cameras, silently, no error, for the entire 3rd pass (~3,500/95,658
episodes, ~10 hours). Verified directly via `ffmpeg` frame extraction +
pixel stats (mean=0, max=0 — completely black) before acting, not assumed.

**Fix**: removed the CLI's duplicate default; `--camera-remap` now
defaults to `None` and genuinely falls back to `_DEFAULT_CAMERA_REMAP` —
one source of truth. Confirmed the other 30 already-cached sources use
none of the newly-fixed-only camera names, so their caches stayed valid
(checked `_align_tmp_*` mtimes post-relaunch — unmodified, correctly
reused, not rebuilt).

Stopped the (confirmed worthless — all-black) 3rd DROID pass, wiped its
cache, restarted a 4th time. **Verified this time, not just trusted**: at
~93 episodes into the 4th pass, extracted fresh frames — real content
(mean 69–93, max 255, high variance) and visually confirmed genuine DROID
wrist-camera footage. Full incident account in memory
`ee-space-generalist-plan.md`.

## 5th DROID restart (2026-08-29, 12 days later): disk-full crash, then a bug that made resume unreachable

Checked back in after 12 days with no active monitoring (this session's
scheduled wakeups don't survive a gap that long). The 4th pass had reached
**72,016/95,658 episodes (75%)** before crashing on
`[Errno 28] No space left on device` — disk was at 6.8GB free / 3.6TB.

**Freed ~684GB** before anything else could happen: `pip cache purge` +
`conda clean` (~40GB, zero risk), my own throwaway prewarm scratch output
(4.9GB), `~/.cache/huggingface/lerobot/hub/` (53GB raw source downloads —
verified first that converted `_ee` outputs are independent real files,
not symlinks into this cache, before deleting), 7 raw pre-conversion
source dirs already superseded by their `_ee` counterparts (~57GB), and
several other clearly-stale/deprecated items (~26GB). Deliberately did
**not** touch DROID's own raw source (`lerobot/droid_1.0.1`, 384GB — still
needed via symlink to decode the remaining 25%) or flex-pi's raw downloads
(~434GB — back the depth-preserving outputs, explicitly asked to be kept).

**Relaunched expecting a resume from ~72,017 — got a full rebuild from
episode 0 instead.** The `align_progress.json` marker was intact and
self-consistent, ruling out the already-documented "Parquet footer
corrupted by a hard kill" limitation. Traced the actual log line back to
its source and found a real, previously-undetected bug:
`align_datasets_for_cotraining` (the real production orchestrator)
unconditionally deletes the align cache directory whenever the
"fully-complete" fast-path check fails — **before** ever calling
`align_single_dataset()`, which is where all the resume-vs-wipe logic
actually lives. The caller destroying the directory first made that whole
resume branch unreachable through the real CLI path — every interruption
looked like "start fresh" regardless of how much valid progress existed.
This is why the original resume-support testing (2026-08-16) never caught
it: those tests called `align_single_dataset()` directly, bypassing the
real orchestrator entirely.

**This bug, not the disk-full crash, is what actually destroyed the
72,007-episode/75%/~516GB 4th-pass progress** — a plain crash with a
working resume path would have cost far less.

**Fix**: removed the caller's premature `shutil.rmtree()`; it now just
hands off to `align_single_dataset()`'s existing, correct logic. Verified
the fix loads cleanly. **Not yet verified end-to-end** (a live
crash/resume test was started but killed partway through to stop
competing for CPU with the actually-important live DROID run) — flagged
honestly rather than assumed working. Worth a proper verification once
there's spare capacity.

Did not relaunch a 6th time — the 5th launch had already rebuilt from
episode 0 by the time this fix landed, so there was nothing further to
protect by restarting again; left it running. The fix protects this run
(and any future one) against losing progress on its *next* crash, not this
one. Full account in memory `ee-space-generalist-plan.md`.

## New sources added 2026-08-16/17: flex-pi + xLeRobot community

Converted (not yet merged in — will be picked up on the NEXT full relaunch,
after the current DROID pass finishes):

- **7 flex-pi datasets** (`huggingface.co/flex-pi`) via new
  `scripts/convert_flexpi_ee.py`: 6 dual-arm "yam" teleop tasks + 4
  LIBERO-with-depth sim suites (single-arm Franka Panda, same embodiment as
  DROID). RGB+depth both preserved, but **kept as separate standalone
  datasets** (`Odog16/flexpi_<name>_ee`), not folded into the shared
  `generalist_ee_merged` schema — an explicit decision (asked first) since
  the merge tool requires identical feature dicts across every source, so
  adding depth would force a zero-filled-depth rebuild of all ~30
  already-aligned RGB-only sources plus the running DROID pass.
- **25 xLeRobot community datasets** (4 contributors, ~496 raw episodes)
  via new `scripts/convert_xlerobot_community.py`: same physical arm as
  this project's own `own_tasks` data, so reuses the existing
  `XLerobotArmKinematics` FK rather than a new embodiment. 2 candidate
  repos had no working version on the Hub and were excluded (not silently
  dropped). Gripper calibration turned out to differ by contributor (not
  just by a documented constant) — fixed via per-source empirical
  auto-calibration rather than a shared guessed constant.
- 5 real bugs found and fixed along the way (wrong v2.1-vs-v3.0 format
  assumption, wrong on-disk episode-metadata layout, wrong video directory
  naming, a missing error-isolation gap in gripper calibration that hit
  the same video-decode bug class as DROID's own crash, and the
  depth-vs-no-depth scoping decision itself). Full account, including the
  exact reasoning for each, in memory `ee-space-generalist-plan.md` under
  "New sources: flex-pi + xLeRobot community".
- **Still to do** once both conversion batches finish: prewarm their
  `_align_tmp_*` align caches (via `align_datasets_for_cotraining` under a
  scratch `--target-repo-id`, so the throwaway final-aggregation step
  doesn't touch the real `generalist_ee_merged` output), then include them
  in the full source list on the next relaunch after DROID's current pass
  completes. `config/generalist_source_weights.yaml` will need a new
  source group (and probably a volume cap, mirroring DROID's 30%) for
  these ~34 new small sources.

## 2026-08-16 crash: what happened, what changed

`IndexError: Invalid frame index=160411 ... must be less than 160411` from
torchcodec, at DROID episode 53,386/95,658. Root cause: that episode's data
is split across two physical parquet/video files (verified directly:
`file-100.mp4` has exactly 160,411 frames, matching the crash's own error
number, and episode 53386's video timestamp range ends exactly at that
file's boundary) — the same class of bug as the original DROID metadata
issue found earlier this project, but hitting the **video** boundary
pointer this time, which that earlier fix explicitly left unaddressed on
the (incorrect, in hindsight) assessment that it was cosmetic. Since
~42,644 of 95,658 episodes were originally flagged as split, this can
recur.

**Fix**: `align_single_dataset` (`co_training_utils.py`) now wraps each
episode in `try/except Exception` — skip and log, don't crash the whole
multi-day process — mirroring the pattern already established in
`scripts/convert_joint_to_ee.py`. A `meta/align_progress.json` marker and
`LeRobotDataset.resume()` support were also added for recovering from an
interruption without redoing already-processed episodes.

**A bug in the first version of this fix destroyed the 53,386 episodes of
real DROID progress that existed at crash time**, before it could be
stopped — `_write_align_options()` (persists `pad_fill_mode`) was only
called at the end of a successful run, so a crashed run's cache always
looked like a `pad_fill_mode` mismatch on the next launch and got wiped
within ~15 seconds, faster than a `kill -STOP` could land. Root cause found
and fixed (write it immediately after dataset creation instead), verified
via a genuine hard-crash test (`os._exit()` in a subprocess, not just a
shortened clean run — the earlier test suite's gap). That test then
surfaced a second, deeper limitation: Parquet only writes a valid footer at
`finalize()` time, so a true hard kill leaves episode metadata unreadable
regardless of buffering settings — meaning **true resume-from-a-hard-crash
isn't safely achievable** without bigger architecture changes (periodic
checkpoint batching) that were deliberately not attempted this pass, given
a real mistake had just happened. Resume is now **fail-safe** instead:
`LeRobotDataset.resume()` is wrapped in try/except, falling back to a clean
wipe-and-restart (not a crash) if it can't be honored. In practice this
means resume rarely succeeds against a genuine crash under this codebase's
architecture, but it never makes things worse either.

**Net effect**: DROID restarted from episode 0 on 2026-08-16 (the other 30
sources' caches were untouched and correctly fast-path-reused). It now has
real crash resilience against this bug class recurring. Full incident
account in memory `ee-space-generalist-plan.md`.

## Second incident, same day: concurrent-process race destroyed the restart's progress too

~2.5 hours into the restart above, a SECOND, independent
`lerobot-cotrain-align` invocation (same command, same target — origin not
conclusively identified, but not this user via any normal shell, and not
cron/systemd) started, found the first process's `_align_tmp_Odog16__droid_ee`
mid-write, misjudged it as an interrupted prior run, and raced its own
`shutil.rmtree()` against the first process's live writes — deleting
`data/`/`videos/` out from under it before the rmtree itself crashed.
~1643 episodes of the restart's progress were lost.

**Fix**: `_exclusive_align_lock()` — an `fcntl.flock()`-based lock (auto-
released by the OS on ANY process exit, including a hard kill, so it can
never leave a stale lock blocking future runs) now wraps the per-source
decision + alignment call in `align_datasets_for_cotraining`'s loop. A
second invocation targeting the same output directory is now refused with
a clear error instead of racing. Verified via genuine multi-process tests:
concurrent refusal, post-hard-kill lock release, and enforcement through
the real production call path (not just the lock mechanism in isolation).

DROID restarted from episode 0 a third time (2026-08-16) with skip-on-error
+ fail-safe-resume + the concurrency lock all in place. Full account in
memory `ee-space-generalist-plan.md`.

## Goal

Merge 4 EE-space data sources (own tasks, FastUMI, ALOHA, DROID) into one
dataset, then train a pi0.5 generalist, then validate it against the
existing SmolVLA generalist before adopting it as default.

## Current step: 4-source dataset merge

Relaunch with this exact command (nothing needs to change):

```bash
conda run -n lerobot lerobot-cotrain-align \
  --target-repo-id Odog16/generalist_ee_merged \
  --target-fps 30 --target-state-dim 14 --target-action-dim 14 \
  --target-image-size 360x640 \
  --target-camera-keys head,left_wrist,right_wrist \
  --camera-fill-mode zero \
  --match-features-from Odog16/tool_pickup_ee \
  --pad-fill-mode ref-mean --override-padded-stats \
  --push-to-hub false \
  --source-repos \
  Odog16/tool_pickup_ee Odog16/trash_pickup_merged_ee Odog16/block_sorting_single_ee Odog16/block_sorting_clean_ee Odog16/making_coffee_v1_ee Odog16/ob15_general_dataset_v1_ee Odog16/ob15_packing_box_filtered_ee \
  Odog16/umi_ee_pretrain_full_clean_desktop Odog16/umi_ee_pretrain_full_dispose_of_desktop_debris Odog16/umi_ee_pretrain_full_put_books_into_schoolbag Odog16/umi_ee_pretrain_full_pack_skincare_products Odog16/umi_ee_pretrain_full_take_bottle_and_place_on_coaster \
  Odog16/aloha_sim_insertion_human_ee Odog16/aloha_sim_transfer_cube_human_ee Odog16/aloha_sim_transfer_cube_scripted_ee Odog16/aloha_static_coffee_ee Odog16/aloha_mobile_cabinet_ee Odog16/aloha_mobile_wash_pan_ee Odog16/aloha_static_screw_driver_ee Odog16/aloha_static_candy_ee Odog16/aloha_mobile_wipe_wine_ee Odog16/aloha_static_towel_ee Odog16/aloha_static_vinh_cup_ee Odog16/aloha_static_vinh_cup_left_ee Odog16/aloha_static_ziploc_slide_ee Odog16/aloha_static_coffee_new_ee Odog16/aloha_static_cups_open_ee Odog16/aloha_static_pingpong_test_ee Odog16/aloha_static_pro_pencil_ee Odog16/aloha_sim_insertion_scripted_ee \
  Odog16/droid_ee \
  > /home/owen/lerobot/workflows/ee_pretrain/logs/14_generalist_merge.log 2>&1 &
disown
```

Progress at the moment of the stop: 10 of 30 sources fully done (7 own_tasks
+ 3 FastUMI: clean_desktop, dispose_of_desktop_debris,
put_books_into_schoolbag). Was ~9% into FastUMI source #4
(`pack_skincare_products`, 228/2539 episodes) when stopped — that source
will safely re-run from episode 0 (a small, accepted rework cost; see the
resume-safety note below). Still to go after that: 1 more FastUMI repo, 17
ALOHA repos, then DROID (95,658 episodes — the long pole, ~4-5 days alone).

**After relaunching**, re-attach a monitor so completion/failure surfaces
automatically instead of needing to poll:
```bash
tail -F -n0 /home/owen/lerobot/workflows/ee_pretrain/logs/14_generalist_merge.log 2>/dev/null | \
  grep -aE --line-buffered "Aligned dataset written to|Skipping alignment \(already merge-ready\)|Wrote co-train source manifest|Overrode merged|Traceback|Error|Killed|OOM|CRITICAL"
```

## Resume-safety note (why relaunching the same command is safe)

`lerobot-cotrain-align`'s cache-reuse check
(`_alignment_artifacts_match` in
`src/lerobot/data_processing/co_training_utils.py`) verifies both schema
*and* episode count before treating a source's aligned output as complete —
this was a real bug found and fixed on 2026-07-25 (previously it only
checked schema, so a killed mid-alignment run could have been silently
accepted as "done," truncating that source). Relaunching now will reuse the
10 already-completed sources' caches untouched and only rebuild the one that
was interrupted.

## Key decisions made (see memory `ee-space-generalist-plan.md` for full detail)

- Backbone: pi0.5 (`--policy.type=pi05`, no underscore),
  `train_expert_only=true` — chosen for the single RTX 3090 Ti's VRAM. No
  formal multi-backbone comparison was ever done; this is a working
  assumption.
- Resolution: staying at 360x640 for this generalist. A separate high-res
  merge is a later, evidence-gated follow-on, only after this generalist
  trains and validates — not blocking now.
- DROID (single-arm) padded into the 14-dim dual-arm schema: right arm
  (indices 7-13) is `ref-mean` fill data, `right_wrist` camera is
  zero-filled (black frame) — both tracked per-source in
  `meta/cotrain_sources.json`.
- Sampling: `WeightedEpisodeAwareSampler` +
  `config/generalist_source_weights.yaml` caps DROID at 30% draw share
  despite being ~90% of raw volume (own_tasks 35%, fastumi_ee 20%,
  aloha_ee 15%). This is the mechanism actually built/tested — not RABC
  (RABC needs SARM-computed progress values that don't exist yet).
- Loss masking: `DimMaskProvider`
  (`src/lerobot/utils/dim_masking.py`) excludes DROID's padded right-arm
  dims from the pi0.5 loss via `batch["action.dim_mask"]` — wired into
  `PI05Policy.forward()` only, verified as a true no-op when unset.
- Camera normalization: pi0.5 defaults `VISUAL` to `IDENTITY` (no stats used
  for images at all), so DROID's black-filled camera can't pollute other
  sources' real camera stats. Non-issue under the current config.
- **Fixed bug #1**: DROID's dataset had `observation.state`/`action` names
  that didn't match the reference dataset's naming convention (a
  shape/names-length mismatch) — would have silently discarded all of
  DROID's real motion data via a positional-padding fallback. Fixed via a
  metadata-only edit to `Odog16/droid_ee/meta/info.json` before this merge
  started.
- **Fixed bug #2**: the merge tool's cache-reuse check didn't verify
  episode count, only schema (see "Resume-safety note" above).

## Not yet done (in order)

1. Finish the merge (in progress — DROID restarted from episode 0 on
   2026-08-16 after the crash above; ETA ~5 days from restart).
2. Verify merged output: `--inspect-only`, confirm DROID's fps/resolution
   match target exactly, spot-check `cotrain_sources.json` padded-dim
   tracking.
3. Update `config/generalist_source_weights.yaml`'s `source_groups` to the
   real post-merge repo list; re-run `docs/dataset_composition_audit.md`'s
   numbers.
4. Deferred Part-A verification: real `padded_action_dims` check, short
   `--steps=50` smoke run with `--dim_masking.sources_path`, no-masking
   regression run (see `docs/missing_modality_fixes_validation.md`).
5. Decide pi0.5 training params (batch size, step budget) and launch
   training — not scheduled yet. Correct flags: `--policy.type=pi05` (no
   underscore), no `--policy.action_space` flag exists (policies are
   dimension-agnostic), multi-source weighting flag is
   `--source_sampling.group_config_path=config/generalist_source_weights.yaml`
   (not `--dataset.multi_source_config`).
6. Validate trained generalist vs existing SmolVLA generalist via a real
   fine-tune comparison before adopting as default.
7. Reward-model scoring/filtering (Robometer/TOPReward/SARM) — task #19,
   separately pending, not yet scoped for this dataset.
8. High-res merge (Part 2, deferred) — only after step 6 passes.
9. Part B (world-model synthetic data for DROID's padded right arm) —
   deferred until after a generalist has actually trained (see
   `docs/worldmodel_synthetic_data_investigation.md`).

Full detail in memory: `ee-space-generalist-plan.md`. Plan file:
`/home/owen/.claude/plans/vectorized-prancing-rabin.md`.
