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
Config plumbing for `WeightedEpisodeAwareSampler` (see `lerobot.datasets.sampler`).

A merged co-training dataset's `meta/cotrain_sources.json` (written by
`lerobot-cotrain-align`) tags contiguous episode ranges with the individual
source *repo* they came from. For sampling control we usually want to weight
*groups* of repos (e.g. "own_tasks" = 8 own repos, "droid_ee" = 1 huge repo),
not each repo individually — see `config/generalist_source_weights.yaml` for
the reference file this module reads.

Example usage (mirrors `lerobot.utils.sample_weighting`'s factory pattern):
    from lerobot.utils.source_sampling import load_source_sampling_config, \\
        build_episode_source_groups

    group_weights, repo_to_group = load_source_sampling_config(path)
    episode_groups = build_episode_source_groups(
        cotrain_sources_path, repo_to_group, total_episodes=dataset.meta.total_episodes,
    )
    sampler = WeightedEpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_source_group=episode_groups,
        group_target_weights=group_weights,
        ...
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceSamplingConfig:
    """
    Attributes:
        group_config_path: Path to a YAML file with `source_weights` (group
            name -> target fraction of each epoch, must sum to 1.0) and
            `source_groups` (group name -> list of repo_ids belonging to it).
        sources_path: Path to a cotrain_sources.json manifest. Auto-detected
            as <dataset_root>/meta/cotrain_sources.json when unset (same
            convention as SampleWeightingConfig.sources_path).
        max_upsample_factor: Warn if any group's implied upsampling factor
            (target share / natural frame share) exceeds this.
        num_samples_per_epoch: Defaults to the dataset's real frame count.
    """

    group_config_path: str = ""
    sources_path: str | None = None
    max_upsample_factor: float = 10.0
    num_samples_per_epoch: int | None = None
    extra_params: dict = field(default_factory=dict)


def load_source_sampling_config(group_config_path: str | Path) -> tuple[dict[str, float], dict[str, str]]:
    """Parse the group-weights YAML into (group_weights, repo_id -> group_name).

    Returns repo_id -> group_name (not group_name -> [repo_ids]) since that's
    the direction `build_episode_source_groups` needs to look things up in.
    """
    group_config_path = Path(group_config_path).expanduser()
    if not group_config_path.is_file():
        raise FileNotFoundError(
            f"Source-sampling group config not found: {group_config_path}. See "
            "config/generalist_source_weights.yaml for the expected format."
        )
    with open(group_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    group_weights = cfg.get("source_weights") or {}
    source_groups = cfg.get("source_groups") or {}
    if not group_weights:
        raise ValueError(f"{group_config_path} has no 'source_weights' section.")
    total = sum(group_weights.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            f"source_weights in {group_config_path} must sum to 1.0, got {total:.4f}: {group_weights}"
        )

    missing_group_defs = set(group_weights) - set(source_groups)
    if missing_group_defs:
        raise ValueError(
            f"{group_config_path}: source_weights defines group(s) {sorted(missing_group_defs)} with "
            "no matching entry under 'source_groups' (no repos assigned to them)."
        )

    repo_to_group: dict[str, str] = {}
    for group_name, repo_ids in source_groups.items():
        for repo_id in repo_ids:
            if repo_id in repo_to_group:
                raise ValueError(
                    f"{group_config_path}: repo_id '{repo_id}' listed in both group "
                    f"'{repo_to_group[repo_id]}' and '{group_name}' — each source repo must "
                    "belong to exactly one group."
                )
            repo_to_group[repo_id] = group_name

    return group_weights, repo_to_group


def build_episode_source_groups(
    sources_path: str | Path,
    repo_to_group: dict[str, str],
    total_episodes: int,
) -> list[str]:
    """Read cotrain_sources.json's per-repo episode ranges and expand them into
    one group-name-per-episode list, the shape WeightedEpisodeAwareSampler wants.

    Raises if a repo present in the merged dataset's manifest has no group
    assignment in the YAML config — silently bucketing an unmapped repo into
    some default group would misassign its sampling weight without any signal
    that it happened.
    """
    sources_path = Path(sources_path).expanduser()
    if not sources_path.is_file():
        raise FileNotFoundError(
            f"Co-train source manifest not found: {sources_path}. It is written by "
            "lerobot-cotrain-align (meta/cotrain_sources.json in the merged dataset)."
        )
    with open(sources_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("sources") or []
    if not entries:
        raise ValueError(f"Manifest {sources_path} has no sources.")

    groups = [""] * total_episodes
    unmapped_repos = set()
    for entry in entries:
        repo_id = entry.get("repo_id", "?")
        group = repo_to_group.get(repo_id)
        if group is None:
            unmapped_repos.add(repo_id)
            continue
        start, end = int(entry["episode_start"]), int(entry["episode_end"])
        for ep in range(start, end):
            groups[ep] = group

    if unmapped_repos:
        raise ValueError(
            f"Repo(s) {sorted(unmapped_repos)} are in {sources_path} but not assigned to any group "
            "in the source-sampling YAML config's 'source_groups' section. Add them to a group (or a "
            "new one) before training — an unmapped source would otherwise get sampled with an "
            "undefined weight."
        )
    unassigned = [i for i, g in enumerate(groups) if g == ""]
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} episode(s) (e.g. {unassigned[:5]}) are not covered by any entry in "
            f"{sources_path} — the manifest's episode ranges don't cover the full dataset."
        )
    return groups
