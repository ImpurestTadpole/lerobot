#!/usr/bin/env python3
"""Upload multiple training checkpoints to Hugging Face Hub."""
from pathlib import Path

from huggingface_hub import HfApi

# ── Edit these for each run ────────────────────────────────────────────────────
RUNS = [
    # (output_dir, hf_repo_prefix, steps_to_upload)
    # trash_pickup SmolVLA — upload 15k checkpoint (20k was auto-pushed at end of training)
    (
        "outputs/train/trash_pickup_SmolVLA_v1",
        "Odog16/trash_pickup_SmolVLA_v2.1_rabc",
        ["015000"],
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


def upload_run(api: HfApi, output_dir: str, repo_prefix: str, steps: list[str]) -> None:
    for step in steps:
        step_k = int(step) // 1000
        repo_id = f"{repo_prefix}_{step_k}k"
        ckpt_dir = Path(output_dir) / "checkpoints" / step / "pretrained_model"
        if not ckpt_dir.exists():
            print(f"  Skipping {step}: {ckpt_dir} not found")
            continue
        print(f"  Uploading {step} → {repo_id} ...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Checkpoint {step}",
        )
        print(f"  Done: https://huggingface.co/{repo_id}")


def main() -> None:
    api = HfApi()
    for output_dir, repo_prefix, steps in RUNS:
        print(f"\n=== {repo_prefix} ===")
        upload_run(api, output_dir, repo_prefix, steps)


if __name__ == "__main__":
    main()
