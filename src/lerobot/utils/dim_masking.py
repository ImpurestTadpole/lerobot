# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Per-dimension action loss masking for co-training merges.

``lerobot-cotrain-align`` writes ``meta/cotrain_sources.json`` into a merged
dataset, recording (per source, per contiguous episode range) which
action/state dims were filled because that source's embodiment doesn't have
the corresponding DOF (e.g. DROID, a single-arm robot, merged into a dual-arm
14-dim schema: its right arm — indices 7-13 — is entirely fill data, not real
commands). Training on those fill values as if they were real targets teaches
the model a specific wrong "hold this fake pose" behavior for that source,
not an ignorable "don't know" signal.

This module builds a per-episode boolean mask from that same manifest (same
data source `SourceWeighter` already reads, see
``lerobot.utils.sample_weighting``) and exposes it to the training loop via
``batch[ACTION_DIM_MASK]`` so a policy's loss computation can exclude padded
dims for the samples they don't apply to. See ``PI05Policy.forward`` for the
consumer side.

Example usage (mirrors ``sample_weighting``'s factory pattern):
    from lerobot.utils.dim_masking import make_dim_mask_provider

    dim_mask_provider = make_dim_mask_provider(cfg.dim_masking, device, dataset_root=...)
    ...
    if dim_mask_provider is not None:
        batch[ACTION_DIM_MASK] = dim_mask_provider.compute_batch_mask(batch)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DimMaskingConfig:
    """
    Configuration for per-dimension action loss masking.

    Attributes:
        sources_path: Path to a cotrain_sources.json manifest. Auto-detected
            as <dataset_root>/meta/cotrain_sources.json when unset (same
            convention as SampleWeightingConfig.sources_path).
        action_dim: Real (unpadded) action dim. Auto-detected from the
            manifest's `action_names` length when unset.
        extra_params: Additional parameters, currently unused (kept for
            parity with the other per-source config dataclasses).
    """

    sources_path: str | None = None
    action_dim: int | None = None
    extra_params: dict = field(default_factory=dict)


def make_dim_mask_provider(
    config: DimMaskingConfig | None,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> DimMaskProvider | None:
    """Factory function to create a DimMaskProvider from config, or None to disable masking."""
    if config is None:
        return None

    sources_path = config.sources_path
    if sources_path is None:
        if dataset_root:
            sources_path = str(Path(dataset_root) / "meta" / "cotrain_sources.json")
        elif dataset_repo_id:
            sources_path = str(
                Path("~/.cache/huggingface/lerobot").expanduser()
                / dataset_repo_id
                / "meta"
                / "cotrain_sources.json"
            )
        else:
            raise ValueError(
                "dim_masking requires 'dim_masking.sources_path' or a local 'dataset.root' "
                "containing meta/cotrain_sources.json (written by lerobot-cotrain-align)."
            )

    return DimMaskProvider(
        sources_path=sources_path,
        action_dim=config.action_dim,
        device=device,
    )


class DimMaskProvider:
    """
    Per-episode, per-action-dim boolean mask: True where a dim is a real
    command for that episode's source, False where it's a co-training merge
    fill value that should not contribute to the training loss.

    Mirrors `lerobot.utils.sample_weighting.SourceWeighter`'s constructor and
    batch-lookup pattern exactly, but produces a `[total_episodes, action_dim]`
    table instead of a scalar-per-episode weight.
    """

    def __init__(
        self,
        sources_path: str | Path,
        action_dim: int | None = None,
        device: torch.device | None = None,
    ):
        self.device = device if device is not None else torch.device("cpu")

        sources_path = Path(sources_path).expanduser()
        if not sources_path.is_file():
            raise FileNotFoundError(
                f"Co-train source manifest not found: {sources_path}. It is written by "
                "lerobot-cotrain-align (meta/cotrain_sources.json in the merged dataset); "
                "re-run the merge or set dim_masking.sources_path explicitly."
            )
        with open(sources_path, encoding="utf-8") as f:
            manifest = json.load(f)

        entries = manifest.get("sources") or []
        total_episodes = int(manifest.get("total_episodes") or 0)
        if not entries or total_episodes <= 0:
            raise ValueError(f"Manifest {sources_path} has no sources/episodes.")

        if action_dim is None:
            action_names = manifest.get("action_names")
            if not action_names:
                raise ValueError(
                    f"Manifest {sources_path} has no 'action_names'; pass "
                    "dim_masking.action_dim explicitly."
                )
            action_dim = len(action_names)
        self._action_dim = action_dim

        mask = torch.ones(total_episodes, action_dim, dtype=torch.bool)
        self._per_source_padded: dict[str, list[int]] = {}
        n_masked_eps = 0
        for entry in entries:
            repo_id = entry.get("repo_id", "?")
            padded_dims = [d for d in (entry.get("padded_action_dims") or []) if 0 <= d < action_dim]
            start, end = int(entry["episode_start"]), int(entry["episode_end"])
            if padded_dims:
                mask[start:end, padded_dims] = False
                n_masked_eps += end - start
            self._per_source_padded[repo_id] = padded_dims
        self._mask = mask.to(self.device)
        self._total_episodes = total_episodes
        self._n_masked_eps = n_masked_eps

    def compute_batch_mask(self, batch: dict) -> torch.Tensor:
        """Return a `[batch_size, action_dim]` bool mask for `batch["episode_index"]`."""
        ep_idx = batch.get("episode_index")
        if ep_idx is None:
            raise KeyError(
                "DimMaskProvider needs 'episode_index' in the batch (present in standard "
                "LeRobotDataset items)."
            )
        ep_idx = ep_idx.reshape(-1).long().to(self._mask.device)
        ep_idx = ep_idx.clamp(0, self._total_episodes - 1)
        return self._mask[ep_idx]

    def get_stats(self) -> dict:
        return {
            "type": "dim_masking",
            "total_episodes": self._total_episodes,
            "masked_episodes": self._n_masked_eps,
            "action_dim": self._action_dim,
            **{
                f"n_padded_dims/{repo_id}": len(dims)
                for repo_id, dims in self._per_source_padded.items()
                if dims
            },
        }
