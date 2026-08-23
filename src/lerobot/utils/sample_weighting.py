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
Sample weighting abstraction for training.

This module provides an abstract base class for sample weighting strategies (e.g., RA-BC)
that can be used during training without polluting the training script with
policy-specific code.

Example usage:
    # In training config
    sample_weighting:
        type: rabc
        progress_path: hf://datasets/my-dataset/sarm_progress.parquet
        head_mode: sparse
        kappa: 0.01

    # In training script
    sample_weighter = make_sample_weighter(cfg.sample_weighting, policy, device, dataset_root=cfg.dataset.root, dataset_repo_id=cfg.dataset.repo_id)
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


class SampleWeighter(ABC):
    """
    Implementations compute per-sample weights that can be used to weight
    the loss during training. This enables techniques like:
    - RA-BC (Reward-Aligned Behavior Cloning)
    - Importance sampling
    - Curriculum learning
    - Quality-based filtering
    """

    @abstractmethod
    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute per-sample weights for a training batch.

        Args:
            batch: Training batch dictionary containing at minimum an "index" key
                   with global frame indices.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get global statistics about the weighting strategy.
        """


@dataclass
class SampleWeightingConfig:
    """
    Configuration for sample weighting during training.

    This is a generic config that supports multiple weighting strategies.
    The `type` field determines which implementation to use, and `extra_params`
    contains additional type-specific parameters.

    Attributes:
        type: Weighting strategy type ("rabc", "source", "uniform", etc.)
        progress_path: Path to precomputed progress values (for RABC)
        head_mode: Which model head to use for progress ("sparse" or "dense")
        kappa: Hard threshold for high-quality samples (RABC-specific)
        epsilon: Small constant for numerical stability
        sources_path: Path to a cotrain_sources.json manifest (for "source").
            Auto-detected as <dataset_root>/meta/cotrain_sources.json when unset.
        external_weight: Loss weight for episodes from non-native (padded)
            co-training sources; native episodes keep weight 1.0 (for "source").
        source_weights: Optional per-repo overrides, e.g.
            {"lerobot/aloha_mobile_cabinet": 0.5} (for "source").
        extra_params: Additional type-specific parameters passed to the weighter
    """

    type: str = "rabc"
    progress_path: str | None = None
    head_mode: str = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6
    sources_path: str | None = None
    external_weight: float = 0.3
    source_weights: dict = field(default_factory=dict)
    # Additional type-specific params can be added here or passed via extra_params
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter | None:
    """
    Factory function to create a SampleWeighter from config.

    This keeps policy-specific initialization logic out of the training script.

    Args:
        config: Sample weighting configuration, or None to disable weighting.
        policy: The policy being trained (used to extract chunk_size, etc.)
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    if config is None:
        return None

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset_root, dataset_repo_id)

    if config.type == "source":
        return _make_source_weighter(config, device, dataset_root, dataset_repo_id)

    if config.type == "uniform":
        # No-op weighter that returns uniform weights
        return UniformWeighter(device=device)

    raise ValueError(
        f"Unknown sample weighting type: '{config.type}'. Supported types: 'rabc', 'source', 'uniform'"
    )


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """Create RABC weighter with policy-specific initialization.

    Args:
        config: Sample weighting configuration.
        policy: The policy being trained (used to extract chunk_size).
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    # Import here to avoid circular imports and keep RABC code in SARM module
    from lerobot.rewards.sarm.rabc import RABCWeights

    # Extract chunk_size from policy config
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "RABC sample weighting requires a policy with 'chunk_size' in its config. "
            "This is typically set for action-chunking policies like ACT, Diffusion, PI0, etc."
        )

    # Determine progress_path: use explicit config or auto-detect from dataset
    progress_path = config.progress_path
    if progress_path is None:
        if dataset_root:
            progress_path = str(Path(dataset_root) / "sarm_progress.parquet")
        elif dataset_repo_id:
            progress_path = f"hf://datasets/{dataset_repo_id}/sarm_progress.parquet"
        else:
            raise ValueError(
                "RABC sample weighting requires 'progress_path' to be set, "
                "or dataset_root/dataset_repo_id for auto-detection. "
                "Generate progress values using: "
                "python -m lerobot.rewards.sarm.compute_rabc_weights --help"
            )

    return RABCWeights(
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        kappa=config.kappa,
        epsilon=config.epsilon,
        device=device,
        **config.extra_params,
    )


def _make_source_weighter(
    config: SampleWeightingConfig,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """Create a SourceWeighter from a merged co-training dataset's manifest."""
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
                "Source sample weighting requires 'sample_weighting.sources_path' or a local "
                "'dataset.root' containing meta/cotrain_sources.json (written by lerobot-cotrain-align)."
            )
    return SourceWeighter(
        sources_path=sources_path,
        external_weight=config.external_weight,
        source_weights=config.source_weights,
        device=device,
        epsilon=config.epsilon,
        **config.extra_params,
    )


class SourceWeighter(SampleWeighter):
    """
    Per-episode loss weights based on which co-training source an episode came from.

    ``lerobot-cotrain-align`` writes ``meta/cotrain_sources.json`` into the merged
    dataset, mapping contiguous episode ranges back to their source repos and
    recording which state/action dims were zero/mean-padded per source. Episodes
    from *native* sources (no padded dims — your own embodiment) keep weight 1.0;
    episodes from padded external sources get ``external_weight`` (< 1) so their
    supervised-to-constant padded dims contribute a proportionally smaller
    gradient without discarding their visual/language diversity.

    Per-repo overrides via ``source_weights={"repo/id": w}`` take precedence.

    Weights are normalized per batch to sum to batch size (same convention as
    RA-BC) so the overall gradient scale is preserved.
    """

    def __init__(
        self,
        sources_path: str | Path,
        external_weight: float = 0.3,
        source_weights: dict[str, float] | None = None,
        device: torch.device | None = None,
        epsilon: float = 1e-6,
        normalize: bool = True,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon
        self.normalize = normalize

        sources_path = Path(sources_path).expanduser()
        if not sources_path.is_file():
            raise FileNotFoundError(
                f"Co-train source manifest not found: {sources_path}. It is written by "
                "lerobot-cotrain-align (meta/cotrain_sources.json in the merged dataset); "
                "re-run the merge or set sample_weighting.sources_path explicitly."
            )
        with open(sources_path, encoding="utf-8") as f:
            manifest = json.load(f)

        source_weights = source_weights or {}
        entries = manifest.get("sources") or []
        total_episodes = int(manifest.get("total_episodes") or 0)
        if not entries or total_episodes <= 0:
            raise ValueError(f"Manifest {sources_path} has no sources/episodes.")

        weights = torch.ones(total_episodes, dtype=torch.float32)
        self._per_source: dict[str, float] = {}
        n_external_eps = 0
        for entry in entries:
            repo_id = entry.get("repo_id", "?")
            native = bool(entry.get("native", False))
            w = source_weights.get(repo_id, 1.0 if native else float(external_weight))
            start, end = int(entry["episode_start"]), int(entry["episode_end"])
            weights[start:end] = w
            self._per_source[repo_id] = w
            if not native:
                n_external_eps += end - start
        self._episode_weights = weights.to(self.device)
        self._total_episodes = total_episodes
        self._n_external_eps = n_external_eps

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        ep_idx = batch.get("episode_index")
        if ep_idx is None:
            raise KeyError(
                "SourceWeighter needs 'episode_index' in the batch (present in standard "
                "LeRobotDataset items)."
            )
        ep_idx = ep_idx.reshape(-1).long().to(self._episode_weights.device)
        ep_idx = ep_idx.clamp(0, self._total_episodes - 1)
        weights = self._episode_weights[ep_idx]
        stats = {
            "mean_weight": float(weights.mean()),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
        }
        if self.normalize:
            weights = weights * weights.numel() / (weights.sum() + self.epsilon)
        return weights.to(self.device), stats

    def get_stats(self) -> dict:
        return {
            "type": "source",
            "total_episodes": self._total_episodes,
            "external_episodes": self._n_external_eps,
            **{f"weight/{k}": v for k, v in self._per_source.items()},
        }


class UniformWeighter(SampleWeighter):
    """
    No-op sample weighter that returns uniform weights.

    Useful as a baseline or when you want to disable weighting without
    changing the training code structure.

    Note:
        Batch size is determined by looking for tensor values in the batch
        dictionary. The method checks common keys like "action", "index",
        and "observation.state" first, then falls back to scanning all values.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return uniform weights (all ones)."""
        batch_size = self._determine_batch_size(batch)

        weights = torch.ones(batch_size, device=self.device)
        stats = {"mean_weight": 1.0, "type": "uniform"}
        return weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        """
        Determine batch size from the batch dictionary.

        Checks common keys first, then scans all values for tensors.

        Args:
            batch: Training batch dictionary.
        """
        if not batch:
            raise ValueError("Cannot determine batch size from empty batch")

        # Check common keys first
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]

        # Scan all values for any tensor
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]

        # Last resort: return 1 (this handles non-tensor batches)
        return 1

    def get_stats(self) -> dict:
        """Return empty stats for uniform weighting."""
        return {"type": "uniform"}
