#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


class WeightedEpisodeAwareSampler:
    """Like `EpisodeAwareSampler`, but draws frames so each named *source group*
    contributes a configured target proportion of every epoch, regardless of
    how many frames that group actually has.

    This is a different mechanism from `lerobot.utils.sample_weighting.SourceWeighter`:
    that class re-weights the *loss* of samples after they've already been drawn
    (so a source with 10x more frames still shows up 10x more often per epoch,
    just contributes less to the gradient each time). This sampler instead
    controls *draw frequency* directly — the actual fix for one large source
    (e.g. DROID) diluting a training run's *effective* diversity, not just its
    average gradient magnitude.

    Same external interface as `EpisodeAwareSampler` (`__iter__`, `__len__`,
    `set_epoch`, `state_dict`, `load_state_dict`) so it's a drop-in swap at the
    dataloader construction site, including the same `(seed, epoch)`-derived
    reproducibility (no shared `generator` to sync across distributed ranks).

    Sampling is with replacement (`torch.multinomial`), matching
    `torch.utils.data.WeightedRandomSampler`'s approach — the alternative
    (a fixed per-epoch quota per group, no replacement) would under-use a small
    group once its own frames are exhausted mid-epoch when it's been upsampled,
    which is exactly the case this class exists to handle well.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_source_group: list[str],
        group_target_weights: dict[str, float],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
        num_samples_per_epoch: int | None = None,
        max_upsample_factor: float = 10.0,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_source_group: Group name for each episode (same length as
                dataset_from_indices), e.g. from a repo_id -> group_name lookup
                built out of a cotrain_sources.json manifest + a group config.
            group_target_weights: {group_name: target fraction of each epoch's
                samples}. Must sum to ~1.0 over the groups actually present.
            num_samples_per_epoch: Defaults to the total real frame count (so
                epoch size stays the same order of magnitude as an unweighted
                run); pass explicitly to make an epoch longer/shorter.
            max_upsample_factor: Raise if any group's implied upsampling factor
                (target share / natural share) exceeds this — an extreme
                upsample factor usually means a config mistake, not intent.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )
        if len(episode_source_group) != len(from_indices):
            raise ValueError(
                f"episode_source_group must have one entry per episode, got "
                f"{len(episode_source_group)} for {len(from_indices)} episodes"
            )

        total_weight = sum(group_target_weights.values())
        if abs(total_weight - 1.0) > 1e-3:
            raise ValueError(
                f"group_target_weights must sum to 1.0, got {total_weight:.4f} "
                f"({group_target_weights}). Fix the source-sampling config before proceeding."
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        groups_used = [g for g, u in zip(episode_source_group, used, strict=True) if u]
        lengths_used = lengths[used]

        # Total natural frames per group (used both to spread each group's target
        # weight evenly across its own episodes/frames, and to report upsample factors).
        group_frame_totals: dict[str, int] = {}
        for g, n in zip(groups_used, lengths_used, strict=True):
            group_frame_totals[g] = group_frame_totals.get(g, 0) + int(n)
        missing = set(group_frame_totals) - set(group_target_weights)
        if missing:
            raise ValueError(
                f"Episodes present for group(s) {sorted(missing)} but no target weight configured "
                f"for them in group_target_weights={group_target_weights}. Add a weight (can be 0.0 "
                "to deliberately exclude a present source) rather than let it fall back silently."
            )

        # Per-frame weight: uniform within a group (weight[frame] = group_weight / group_frame_count),
        # so within-group sampling stays proportional to each episode's own natural length.
        per_episode_weight = np.array(
            [group_target_weights[g] / group_frame_totals[g] for g in groups_used], dtype=np.float64
        )
        self._frame_weights_by_episode = per_episode_weight  # per-frame weight, constant within an episode
        self._episode_lengths = lengths_used

        total_natural_frames = sum(group_frame_totals.values())
        self._group_stats = {}
        for g, target_w in group_target_weights.items():
            natural_frames = group_frame_totals.get(g, 0)
            natural_share = natural_frames / total_natural_frames if total_natural_frames else 0.0
            upsample_factor = (target_w / natural_share) if natural_share > 0 else float("inf")
            self._group_stats[g] = {
                "natural_frames": natural_frames,
                "natural_share": natural_share,
                "target_share": target_w,
                "upsample_factor": upsample_factor,
            }
            if natural_frames > 0 and upsample_factor > max_upsample_factor:
                logger.warning(
                    "Source group '%s' would be upsampled %.1fx (target share %.2f vs natural "
                    "share %.4f) — exceeds max_upsample_factor=%.1f. This usually means a small "
                    "source got a target weight sized for a much larger one; double-check "
                    "group_target_weights before training on this.",
                    g,
                    upsample_factor,
                    target_w,
                    natural_share,
                    max_upsample_factor,
                )

        self._num_samples_per_epoch = num_samples_per_epoch or self._num_frames
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    def group_stats(self) -> dict:
        """Natural vs. target share and implied upsample factor per group — log this at
        training start so a run's logs are self-documenting about the realized mix."""
        return self._group_stats

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        # Weight each *position* (not each episode) by its episode's per-frame weight,
        # then draw with replacement — equivalent to WeightedRandomSampler over frames,
        # but built from the same O(num_episodes) representation as EpisodeAwareSampler
        # rather than materializing a per-frame weight array up front.
        boundaries = np.concatenate([[0], self._cum_lengths])
        position_weights = np.repeat(self._frame_weights_by_episode, np.diff(boundaries))
        weights_t = torch.from_numpy(position_weights)
        draw = torch.multinomial(
            weights_t,
            num_samples=self._num_samples_per_epoch,
            replacement=True,
            generator=self._epoch_generator(epoch),
        )
        for k in range(start, len(draw)):
            yield self._frame_index(int(draw[k]))

    def __len__(self) -> int:
        return self._num_samples_per_epoch


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
