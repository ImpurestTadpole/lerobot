#!/usr/bin/env python
"""Tests for the co-training source-sampling plumbing used by stage-1 generalist training."""

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.sampler import WeightedEpisodeAwareSampler
from lerobot.utils.source_sampling import build_episode_source_groups, load_source_sampling_config

YAML = """
source_weights:
  small: 0.5
  big: 0.5
source_groups:
  small: [own/a]
  big: [ext/b]
"""


def _write(tmp_path, yaml_text=YAML, sources=None):
    cfg = tmp_path / "groups.yaml"
    cfg.write_text(yaml_text)
    manifest = tmp_path / "cotrain_sources.json"
    manifest.write_text(
        json.dumps(
            sources
            or {
                "sources": [
                    {"repo_id": "own/a", "episode_start": 0, "episode_end": 2},
                    {"repo_id": "ext/b", "episode_start": 2, "episode_end": 10},
                ]
            }
        )
    )
    return cfg, manifest


def test_groups_map_every_episode(tmp_path):
    cfg, manifest = _write(tmp_path)
    weights, repo_to_group = load_source_sampling_config(cfg)
    assert weights == {"small": 0.5, "big": 0.5}
    groups = build_episode_source_groups(manifest, repo_to_group, total_episodes=10)
    assert groups == ["small"] * 2 + ["big"] * 8


def test_weights_must_sum_to_one(tmp_path):
    cfg, _ = _write(tmp_path, YAML.replace("big: 0.5", "big: 0.4"))
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_source_sampling_config(cfg)


def test_unmapped_repo_is_rejected(tmp_path):
    cfg, manifest = _write(tmp_path, YAML.replace("[ext/b]", "[ext/other]"))
    _, repo_to_group = load_source_sampling_config(cfg)
    with pytest.raises(ValueError, match="not assigned to any group"):
        build_episode_source_groups(manifest, repo_to_group, total_episodes=10)


def test_uncovered_episodes_are_rejected(tmp_path):
    cfg, manifest = _write(tmp_path)
    _, repo_to_group = load_source_sampling_config(cfg)
    with pytest.raises(ValueError, match="not covered"):
        build_episode_source_groups(manifest, repo_to_group, total_episodes=12)


def test_weighted_sampler_draws_target_share_not_natural_share(tmp_path):
    """The small group has 20 of 100 frames but must be drawn ~50% of the time."""
    cfg, manifest = _write(tmp_path)
    weights, repo_to_group = load_source_sampling_config(cfg)
    groups = build_episode_source_groups(manifest, repo_to_group, total_episodes=10)
    starts = [i * 10 for i in range(10)]
    ends = [s + 10 for s in starts]
    sampler = WeightedEpisodeAwareSampler(
        starts,
        ends,
        episode_source_group=groups,
        group_target_weights=weights,
        seed=0,
        num_samples_per_epoch=4000,
        max_upsample_factor=10.0,
    )
    drawn = torch.tensor(list(sampler))
    assert len(drawn) == 4000
    small_share = (drawn < 20).float().mean().item()
    assert small_share == pytest.approx(0.5, abs=0.05)
    stats = sampler.group_stats()
    assert stats["small"]["upsample_factor"] == pytest.approx(2.5, rel=0.05)
    # Reproducible per (seed, epoch)
    again = WeightedEpisodeAwareSampler(
        starts, ends, episode_source_group=groups, group_target_weights=weights, seed=0,
        num_samples_per_epoch=4000, max_upsample_factor=10.0,
    )
    assert list(again) == drawn.tolist()


def test_make_dataloaders_uses_weighted_sampler_when_configured(tmp_path):
    """Covers the lerobot_train branch that reads cfg.source_sampling (stage-1 config)."""
    cfg_path, manifest = _write(tmp_path)
    frames_per_ep = 10

    class Dataset(torch.utils.data.Dataset):
        episodes = list(range(10))
        absolute_to_relative_idx = None
        meta = SimpleNamespace(
            episodes={
                "dataset_from_index": [i * frames_per_ep for i in range(10)],
                "dataset_to_index": [(i + 1) * frames_per_ep for i in range(10)],
            },
            has_language_columns=False,
        )

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return index

    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(drop_n_first_frames=0, drop_n_last_frames=0),
        dataset=SimpleNamespace(streaming=False, root=str(tmp_path)),
        resume=False,
        seed=1,
        num_workers=0,
        batch_size=2,
        prefetch_factor=None,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
        max_eval_samples=0,
        source_sampling=SimpleNamespace(
            group_config_path=str(cfg_path),
            sources_path=str(manifest),
            max_upsample_factor=10.0,
            num_samples_per_epoch=200,
            extra_params={},
        ),
    )
    from lerobot.scripts.lerobot_train import make_dataloaders as mk

    train, _ = mk(cfg, Dataset(), Dataset(), 0, SimpleNamespace(device_type="cpu"))
    assert isinstance(train.sampler, WeightedEpisodeAwareSampler)
