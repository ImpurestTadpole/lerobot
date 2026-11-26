"""Reports for the remote week.

report.py <run_id>        write ~/ob15_experiments/reports/<run_id>.md from results/summary.csv
report.py --decide-last   print the run-11 choice: "<id> <kind> <seed> <args...>"
report.py --summary       write ~/ob15_experiments/reports/remote_week_summary_table.md
Outcome rule (stated so it can be checked): clean held-out chunk error within 0.10 deg of the baseline is a tie;
for a stress test, a change of less than 0.25 deg in the drop from clean is a tie. Everything else is better or worse.
"""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

E = Path(__file__).resolve().parent
REPORTS = Path("/home/owen/ob15_experiments/reports")
RUNS = {r["id"]: r for r in json.load(open(E / "runs.json"))}
BASE = "rw_none_base"
STRESS = ["freeze_head", "freeze_left_wrist", "freeze_right_wrist", "shift_head", "dim", "warm", "bright"]


def rows():
    p = E / "results" / "summary.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            if k in ("run_id", "tag", "ckpt"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def best(run_id):
    """The checkpoint with the lowest clean held-out chunk error (ties: later checkpoint)."""
    rs = [r for r in rows() if r["run_id"] == run_id and isinstance(r.get("heldout_chunk"), float)]
    return min(rs, key=lambda r: (r["heldout_chunk"], str(r["tag"])), default=None) if rs else None


def verdict(delta, tol):
    return "tie" if abs(delta) < tol else ("better" if delta < 0 else "worse")


def registry(run_id):
    p = E / "registry.csv"
    return next((r for r in csv.DictReader(open(p)) if r["run_id"] == run_id), {}) if p.exists() else {}


def report(run_id):
    run, b, base = RUNS.get(run_id, {"change": "?", "hyp": "", "pred": ""}), best(run_id), best(BASE)
    reg = registry(run_id)
    lines = [f"# {run_id}", "", f"Change {run['change']}. Run registry: commit `{reg.get('git_commit','?')}`, config hash `{reg.get('config_sha','?')}`, "
             f"manifest bp45_v1, seed {reg.get('seed','?')}, {reg.get('train_hours','?')} h, status {reg.get('status','?')}.", "",
             f"**Hypothesis (written before the run):** {run['hyp']}", "", f"**Prediction:** {run['pred']}", ""]
    if b is None:
        lines += ["**Result:** no evaluation (run failed or was skipped). See the queue log.", ""]
    else:
        curve = sorted((r for r in rows() if r["run_id"] == run_id), key=lambda r: str(r["tag"]))
        lines += ["## Offline results (degrees; held-out = the 5 demos never trained on)", "",
                  "| checkpoint | held-out chunk | held-out first step | train chunk | chunk jump |", "|---|---|---|---|---|"]
        lines += [f"| {r['tag']} | {r['heldout_chunk']:.3f} | {r['heldout_first']:.3f} | {r['train_chunk']:.3f} | {r['chunk_jump']:.3f} |" for r in curve]
        lines += ["", f"Best checkpoint: **{b['tag']}**.", "", "| stress test | drop from clean | baseline drop | vs baseline |", "|---|---|---|---|"]
        for s in STRESS:
            d, bd = b[f"drop_{s}"], (base or {}).get(f"drop_{s}", math.nan)
            lines.append(f"| {s} | {d:.3f} | {bd:.3f} | {verdict(d - bd, 0.25) if base and run_id != BASE else '-'} |")
        if base and run_id != BASE:
            clean = verdict(b["heldout_chunk"] - base["heldout_chunk"], 0.10)
            lines += ["", f"**Outcome:** clean held-out error {b['heldout_chunk']:.3f} vs baseline {base['heldout_chunk']:.3f} -> **{clean}**. "
                      "Robot success: not tested (no robot this week).", ""]
        lines += ["Notes: offline only, one seed, 5 held-out demos (60 frames) - small differences are within noise; "
                  "the reference 0.84 deg chunk jump was 7-step chunks every 4 steps, these are 30-step chunks every 15.", ""]
    lines += ["**Next step:** to be chosen from the change table in the remote-week summary.", ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{run_id}.md").write_text("\n".join(lines))
    tally = Path("/home/owen/ob15_experiments/blog_tally.md")
    if b is not None:
        with open(tally, "a") as f:
            f.write(f"\n- remote week {run_id}: held-out chunk {b['heldout_chunk']:.2f} deg (best {b['tag']}), jump {b['chunk_jump']:.2f}, offline only\n")
    print(f"report written: {REPORTS / (run_id + '.md')}")


def decide_last():
    base = best(BASE)
    cands = [i for i in RUNS if RUNS[i].get("kind", "jepa") == "jepa" and best(i)]
    weights = {"rw_none_wm03": "0.3", "rw_none_wm10": "1.0"}
    better_w = [w for w in weights if best(w) and base and best(w)["heldout_chunk"] < base["heldout_chunk"] - 0.10]
    if better_w:
        w = min(better_w, key=lambda i: best(i)["heldout_chunk"])
        gv = min((i for i in ("rw_gv_randpair", "rw_gv_mixed", "rw_gv_mixed_jitter") if best(i)), key=lambda i: best(i)["heldout_chunk"], default="rw_gv_mixed")
        args = [a for a in RUNS[gv]["args"]] + [f"--policy.world_model_loss_weight={weights[w]}"]
        print(f"rw_{gv[3:]}_wm{weights[w].replace('.', '')} jepa 1000 {' '.join(args)}")
        return
    top = min(cands, key=lambda i: best(i)["heldout_chunk"], default=BASE)
    print(f"rw_best_seed2_{top[3:]} jepa 2 {' '.join(RUNS[top]['args'])}")


def summary():
    rs = {r["run_id"] for r in rows() if str(r["run_id"]).startswith("rw_")}
    lines = ["| run | best ckpt | held-out chunk | held-out first | train chunk | jump | " + " | ".join(STRESS) + " |", "|" + "---|" * (6 + len(STRESS))]
    for i in sorted(rs):
        b = best(i)
        lines.append(f"| {i} | {b['tag']} | {b['heldout_chunk']:.3f} | {b['heldout_first']:.3f} | {b['train_chunk']:.3f} | {b['chunk_jump']:.3f} | " + " | ".join(f"{b['drop_'+s]:.2f}" for s in STRESS) + " |")
    (REPORTS / "remote_week_summary_table.md").write_text("\n".join(lines) + "\n")
    print("summary table written")


if __name__ == "__main__":
    a = sys.argv[1]
    {"--decide-last": decide_last, "--summary": summary}.get(a, lambda: report(a))()
