#!/usr/bin/env python
"""Phase 3 (E5): black out one camera at test time.

A zeroed frame is "camera blacked out", not "camera removed": the policy still gets an input slot,
just a black one. A policy that never saw black frames in training is being tested off-distribution,
so compare like with like:
- train every policy in the comparison with the same `--policy.camera_blank_prob` (e.g. 0.1), or
- report E5 as "camera blacked out, no blanking in training" for all of them.

Apply it to the raw observation, before the policy preprocessor (VLA-JEPA images are in [0, 1] and
use IDENTITY normalization, so 0 is black for both training-time blanking and this helper). Use the
same seed and trial count for every policy.

    action = policy.select_action(preprocessor(drop_camera(observation)))
"""


def drop_camera(obs: dict, key: str = "observation.images.top") -> dict:
    """Return a copy of `obs` with camera `key` blacked out."""
    if key not in obs:
        raise KeyError(f"{key!r} not in observation keys {list(obs)}")
    obs = dict(obs)
    obs[key] = obs[key] * 0  # works for torch tensors and numpy arrays
    return obs
