#!/usr/bin/env python
"""Phase 1 (E1/E2): train a small ViT-S JEPA on robot frames with good or bad views.

good: the two positives are two cameras at the same timestep (wrist + external).
bad:  the two positives are two crops of one camera frame. The camera is picked at random per
      sample, so both encoders see the same image distribution and differ only in the pairing.

Loss: invariance to the view mean + SIGReg (lambda = 0.02) on a BatchNorm MLP projector, with
gradient clipping. SIGReg here is a simplified Epps-Pulley version; use the official LeJEPA code for
an exact match. SIGReg scales with batch size, so keep the batch large (256) for real runs.

Augmentations are drawn independently per sample and per view (torchvision v2 transforms called on
a [B, C, H, W] tensor would share one crop and one jitter across the whole batch).

Example:
    for s in 0 1 2; do
      python examples/goodview_jepa/goodview_jepa.py --repo_id ORG/DATA --mode good --seed $s --out good_$s.pt
      python examples/goodview_jepa/goodview_jepa.py --repo_id ORG/DATA --mode bad  --seed $s --out bad_$s.pt
    done
Hold out episodes for align_eval.py with --episodes (train) and its own --episodes (eval).
"""

import argparse

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GoodViewEncoder(nn.Module):
    """ViT-S/16 backbone (384-d CLS features) + LeJEPA-style MLP projector with BatchNorm.

    The loss uses `forward` (projector output, `dim` wide); evaluation uses `features` (backbone),
    as in LeJEPA. Without the BN projector, SIGReg at lambda = 0.02 did not stop collapse on OB15
    frames (embedding std 0.24 -> 0.008 in 30 steps at batch 16; it held at ~0.1-0.2 with it). The
    Epps-Pulley statistic has zero gradient at z = 0, so a run that collapses does not recover.
    """

    def __init__(self, dim: int = 128, hidden: int = 2048) -> None:
        super().__init__()
        self.backbone = VisionTransformer(
            image_size=224, patch_size=16, num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536
        )
        self.backbone.heads = nn.Identity()  # torchvision's zero-initialised head would start z at 0
        self.proj = nn.Sequential(
            nn.Linear(384, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


# Eval-time preprocessing shared with align_eval.py (no randomness).
eval_tf = v2.Compose([v2.Resize(224), v2.CenterCrop(224), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

aug = v2.Compose(
    [
        v2.RandomResizedCrop(224, scale=(0.3, 1.0), antialias=True),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def aug_per_sample(images: torch.Tensor) -> torch.Tensor:
    """Independent random crop and jitter for every image in a [B, C, H, W] batch."""
    return torch.stack([aug(img) for img in images])


def sigreg(z: torch.Tensor, num_dirs: int = 256, t_max: float = 3.0, n_t: int = 17) -> torch.Tensor:
    """Simplified LeJEPA SIGReg: Epps-Pulley test of random 1-D projections against N(0, 1)."""
    n, d = z.shape
    a = torch.randn(d, num_dirs, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    p = z @ a
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    x = p.unsqueeze(-1) * t
    target = torch.exp(-0.5 * t**2)
    err = (x.cos().mean(0) - target) ** 2 + x.sin().mean(0) ** 2
    return n * torch.trapezoid(err * target, t, dim=-1).mean()


def last_frame(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1] if x.ndim == 5 else x  # [B, T, C, H, W] -> [B, C, H, W]


def make_views(batch: dict, cams: list[str], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = last_frame(batch[cams[0]]), last_frame(batch[cams[1]])
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    if mode == "good":  # two cameras, same timestep
        return aug_per_sample(a), aug_per_sample(b)
    # bad: two crops of one frame; camera chosen per sample so both cameras are seen
    pick = (torch.rand(a.shape[0], device=a.device) < 0.5).view(-1, 1, 1, 1)
    frame = torch.where(pick, a, b)
    return aug_per_sample(frame), aug_per_sample(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cams", nargs=2, default=["observation.images.wrist", "observation.images.top"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="training episodes (hold out the rest)")
    ap.add_argument("--mode", choices=["good", "bad"], default="good")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=None, help="stop after this many steps (smoke runs)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="encoder.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=args.episodes)
    for cam in args.cams:
        if cam not in ds.meta.features:
            raise SystemExit(f"{cam!r} not in dataset features: {list(ds.meta.camera_keys)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    enc = GoodViewEncoder(args.dim).to(args.device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=0.05)
    step = 0
    for ep in range(args.epochs):
        if args.max_steps is not None and step >= args.max_steps:
            break
        for batch in dl:
            if args.max_steps is not None and step >= args.max_steps:
                break
            step += 1
            batch = {k: batch[k].to(args.device) for k in args.cams}
            v1, v2_ = make_views(batch, args.cams, args.mode)
            z1, z2 = enc(v1), enc(v2_)
            center = (z1 + z2) / 2
            inv = F.mse_loss(z1, center) + F.mse_loss(z2, center)
            loss = (1 - args.lam) * inv + args.lam * (sigreg(z1) + sigreg(z2)) / 2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)  # prevents collapse
            opt.step()
            if args.max_steps is not None and step % 10 == 0:
                print(f"step {step} loss {loss.item():.4f} inv {inv.item():.4f} z_std {z1.std(0).mean().item():.3f}")
        print(f"epoch {ep} loss {loss.item():.4f} inv {inv.item():.4f} z_std {z1.std(0).mean().item():.3f}")
    torch.save({"state_dict": enc.state_dict(), "dim": args.dim, "mode": args.mode, "seed": args.seed}, args.out)


if __name__ == "__main__":
    main()
