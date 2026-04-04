#!/usr/bin/env python3
"""Upload multiple training checkpoints to Hugging Face Hub.

Large files may upload via the Hub's XET path (hf_xet). If Ctrl+C produces nested
tracebacks inside huggingface_hub, that is the library aborting mid-transfer—not a
silent corrupt upload. Re-run the script; you may need to delete a half-finished
revision on the Hub in rare cases.

If XET causes hangs or odd errors, use classic LFS-style uploads:
  HF_HUB_DISABLE_XET=1 python upload_checkpoints.py
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

# ── Edit these for each run ────────────────────────────────────────────────────
RUNS = [
    # (output_dir, hf_repo_prefix, steps_to_upload)
    # trash_pickup SmolVLA — upload 15k checkpoint (20k was auto-pushed at end of training)
    (
        "outputs/train/making_coffee_v1_SmolVLA_v1",
        "Odog16/making_coffee_v1_SmolVLA_v1",
        ["020000"],
    ),
    # Uncomment to upload additional checkpoints:
    # (
    #     "outputs/train/trash_pickup_SmolVLA_v1",
    #     "Odog16/trash_pickup_SmolVLA",
    #     ["005000", "010000"],
    # ),
    # (
    #     "outputs/train/trash_pickup_ACT_v1",
    #     "Odog16/trash_pickup_ACT",
    #     ["020000", "040000", "060000", "080000"],
    # ),
    # (
    #     "outputs/train/multi_task_SmolVLA_v1",
    #     "Odog16/multi_task_SmolVLA",
    #     ["010000", "015000", "020000"],
    # ),
]


def _folder_size_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def upload_run(api: HfApi, output_dir: str, repo_prefix: str, steps: list[str]) -> None:
    for step in steps:
        step_k = int(step) // 1000
        repo_id = f"{repo_prefix}_{step_k}k"
        ckpt_dir = Path(output_dir) / "checkpoints" / step / "pretrained_model"
        if not ckpt_dir.exists():
            print(f"  Skipping {step}: {ckpt_dir} not found")
            continue
        total_b = _folder_size_bytes(ckpt_dir)
        gib = total_b / (1024**3)
        # Typical slow home uplink ~80–150 KB/s; hours scale linearly with speed.
        hours_at_100kib_s = (total_b / (100 * 1024)) / 3600
        print(f"  Uploading {step} → {repo_id} ({gib:.2f} GiB) …")
        print(
            f"    Expect ~{hours_at_100kib_s:.1f} h at ~100 KiB/s uplink; ~20% progress is ~{0.2 * gib:.2f} GiB, not a hang. gonna be a while...",
            flush=True,
        )
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        try:
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Checkpoint {step}",
            )
        except KeyboardInterrupt:
            print(
                "\n  Upload interrupted (Ctrl+C). The in-flight commit may be incomplete on the Hub; "
                "re-run to retry. For less noisy cancels, try: HF_HUB_DISABLE_XET=1 python upload_checkpoints.py",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(130) from None
        print(f"  Done: https://huggingface.co/{repo_id}")


def main() -> None:
    api = HfApi()
    for output_dir, repo_prefix, steps in RUNS:
        print(f"\n=== {repo_prefix} ===")
        upload_run(api, output_dir, repo_prefix, steps)


if __name__ == "__main__":
    main()
