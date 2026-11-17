#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate encoders from train_encoder.py: cross-camera and cross-seed alignment, and a state probe.

What each number means:
- cross_camera (within one encoder): cam1 vs cam2 frames through the same encoder. The good-view
  encoder is trained to make these match, so this favours it by design. Report it, but don't use it
  alone to compare good and bad views.
- cross_seed (between two encoders of the same mode, different seeds): does independent training
  converge to the same space (the Platonic-representation question)? Run on the same camera and on
  different cameras (encoder A on cam1 vs encoder B on cam2).
- state_probe: ridge probe from frozen features to observation.state, fit on the fit episodes,
  scored on the test episodes (R^2, higher is better). Neither mode trains for this.

Whitening and the Procrustes rotation are fitted on the fit split and applied unchanged to the test
split. Use episodes held out from training (train_encoder.py --episodes).

Example:
    python examples/goodview_jepa/align_eval.py --repo_id ORG/DATA --episodes 40 41 42 43 44 45 46 47 \\
        --encoders good_0.pt good_1.pt good_2.pt bad_0.pt bad_1.pt bad_2.pt
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# ---- metric code (pure torch, testable without data) ----


def fit_whitener(z: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """ZCA whitening fitted on z: returns (mean, W) so that (z - mean) @ W is white."""
    mean = z.mean(0)
    zc = z - mean
    cov = zc.T @ zc / (len(z) - 1)
    e, v = torch.linalg.eigh(cov + eps * torch.eye(cov.shape[0], device=z.device, dtype=z.dtype))
    return mean, v @ torch.diag(e.rsqrt()) @ v.T


def apply_whitener(z: torch.Tensor, whitener: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, w = whitener
    return (z - mean) @ w


def procrustes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Orthogonal Q that minimises ||a Q - b|| for paired rows."""
    u, _, vt = torch.linalg.svd(a.T @ b)
    return u @ vt


def mutual_knn(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    """Mean overlap of k-NN sets in the two spaces (needs no rotation)."""

    def knn(x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        s = x @ x.T
        s.fill_diagonal_(-float("inf"))
        return s.topk(k, dim=1).indices

    na, nb = knn(a), knn(b)
    hits = (na.unsqueeze(2) == nb.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    return (hits / k).mean().item()


def recall_at_1(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    return ((a @ b.T).argmax(1) == torch.arange(len(a), device=a.device)).float().mean().item()


def alignment(a_fit, b_fit, a_test, b_test, k: int = 10) -> dict[str, float]:
    """Fit whitening + rotation on the fit pairs, score on the test pairs."""
    wa, wb = fit_whitener(a_fit), fit_whitener(b_fit)
    q = procrustes(apply_whitener(a_fit, wa), apply_whitener(b_fit, wb))
    ta, tb = apply_whitener(a_test, wa), apply_whitener(b_test, wb)
    return {"mutual_knn": mutual_knn(ta, tb, k), "R@1_rotated": recall_at_1(ta @ q, tb)}


ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def _ridge_r2(x_fit, y_fit, x_test, y_test, alpha: float) -> float:
    xm, ym = x_fit.mean(0), y_fit.mean(0)
    xs = x_fit.std(0) + 1e-6
    xf, xt = (x_fit - xm) / xs, (x_test - xm) / xs
    w = torch.linalg.solve(xf.T @ xf + alpha * torch.eye(xf.shape[1], dtype=xf.dtype), xf.T @ (y_fit - ym))
    return r2(xt @ w + ym, y_test, y_fit)


def r2(pred: torch.Tensor, y_test: torch.Tensor, y_fit: torch.Tensor) -> float:
    """Mean R^2 over target dims that actually move in the test split.

    A dim whose test spread is under 5% of its fit spread (e.g. a joint parked in the test episodes)
    makes R^2 explode on a near-zero denominator, so it is skipped.
    """
    ss_res = ((y_test - pred) ** 2).sum(0)
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    varying = y_test.std(0) > 0.05 * y_fit.std(0).clamp_min(1e-12)
    if not varying.any():
        return float("nan")
    return (1 - ss_res / ss_tot)[varying].mean().item()


def ridge_probe_r2(x_fit, y_fit, x_test, y_test, groups=None, alphas=ALPHAS) -> tuple[float, float]:
    """Ridge probe, features -> targets; mean R^2 over varying target dims on the test split.

    The penalty is picked by 2-fold cross-validation over `groups` (episode ids) inside the fit split,
    so the probe doesn't overfit when frames are few relative to the feature width. Returns
    (test R^2, chosen alpha).
    """
    if groups is None:
        groups = torch.arange(len(x_fit))
    uniq = groups.unique()
    fold_a = torch.isin(groups, uniq[: max(1, len(uniq) // 2)])
    fold_b = ~fold_a
    if fold_b.sum() < 2:
        best = alphas[len(alphas) // 2]
    else:

        def cv(alpha):
            return (
                _ridge_r2(x_fit[fold_a], y_fit[fold_a], x_fit[fold_b], y_fit[fold_b], alpha)
                + _ridge_r2(x_fit[fold_b], y_fit[fold_b], x_fit[fold_a], y_fit[fold_a], alpha)
            ) / 2

        best = max(alphas, key=cv)
    return _ridge_r2(x_fit, y_fit, x_test, y_test, best), best


# ---- data + encoders ----


@torch.no_grad()
def embed(enc, frames: torch.Tensor, device: str, batch: int = 256) -> torch.Tensor:
    from train_encoder import eval_tf

    out = []
    for i in range(0, len(frames), batch):
        x = torch.stack([eval_tf(f) for f in frames[i : i + batch]]).to(device)
        out.append(enc.features(x).float().cpu())  # backbone features, as in LeJEPA
    return torch.cat(out).double()


def load_frames(repo_id, root, episodes, cams, max_frames):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=episodes)
    # Evenly spaced over every held-out episode (taking a prefix would drop the later episodes).
    idx = torch.linspace(0, len(ds) - 1, min(max_frames, len(ds))).round().long().unique().tolist()
    rows = [ds[i] for i in idx]
    frames = {c: torch.stack([r[c] for r in rows]) for c in cams}
    state = torch.stack([r["observation.state"] for r in rows]).double()
    ep = torch.tensor([int(r["episode_index"]) for r in rows])
    return frames, state, ep


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from train_encoder import GoodViewEncoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", required=True, help="held-out episodes")
    ap.add_argument("--cams", nargs=2, default=["observation.images.wrist", "observation.images.top"])
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="align_eval.json")
    args = ap.parse_args()

    frames, state, ep = load_frames(args.repo_id, args.root, args.episodes, args.cams, args.max_frames)
    # Split by episode so fit and test frames never come from the same episode.
    eps = sorted(set(ep.tolist()))
    fit = torch.isin(ep, torch.tensor(eps[: max(1, len(eps) // 2)]))
    test = ~fit
    if test.sum() < 2 * args.k:
        raise SystemExit("Too few test frames; pass more held-out episodes.")
    print(
        f"fit: {int(fit.sum())} frames from {len(eps) // 2} episodes | test: {int(test.sum())} frames",
        file=sys.stderr,
    )

    feats = {}
    # Untrained encoder as a floor: trained encoders should beat it on every metric.
    torch.manual_seed(0)
    rand = GoodViewEncoder().eval().to(args.device)
    feats["random_init"] = {"mode": "random", **{c: embed(rand, frames[c], args.device) for c in args.cams}}
    for path in args.encoders:
        ckpt = torch.load(path, map_location="cpu")
        enc = GoodViewEncoder(ckpt["dim"])
        enc.load_state_dict(ckpt["state_dict"])
        enc.eval().to(args.device)
        name = Path(path).stem
        feats[name] = {"mode": ckpt.get("mode"), **{c: embed(enc, frames[c], args.device) for c in args.cams}}

    c1, c2 = args.cams
    results = {
        # Predicting the fit-split mean state: the floor a useful probe must beat.
        "state_mean_baseline_R2": r2(state[fit].mean(0).expand_as(state[test]), state[test], state[fit]),
        "per_encoder": {},
        "cross_seed": {},
    }
    for name, f in feats.items():
        results["per_encoder"][name] = {
            "mode": f["mode"],
            "cross_camera (favours good views by design)": alignment(
                f[c1][fit], f[c2][fit], f[c1][test], f[c2][test], args.k
            ),
            "state_probe_R2 (alpha)": {
                c: ridge_probe_r2(f[c][fit], state[fit], f[c][test], state[test], groups=ep[fit])
                for c in args.cams
            },
        }
    for a, b in itertools.combinations(feats, 2):
        if feats[a]["mode"] != feats[b]["mode"]:
            continue
        fa, fb = feats[a], feats[b]
        results["cross_seed"][f"{a}~{b}"] = {
            "same_camera": alignment(fa[c1][fit], fb[c1][fit], fa[c1][test], fb[c1][test], args.k),
            "cross_camera": alignment(fa[c1][fit], fb[c2][fit], fa[c1][test], fb[c2][test], args.k),
        }
    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
