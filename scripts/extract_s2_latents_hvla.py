#!/usr/bin/env python
"""Offline S2 latent extraction for HVLA S1 training (float32 .npy per frame).

See Guide.sh "HVLA — TOOL PICKUP" for a full example command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.hvla.s2.config import S2VLMConfig
from lerobot.policies.hvla.s2.model import S2VLMModel
from lerobot.policies.hvla.s2.preprocessing import preprocess_images
from lerobot.policies.hvla.s2.tokenizer import PaligemmaTokenizer

logger = logging.getLogger(__name__)


def _sample_to_image_dict(
    sample: dict, image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Build uint8 HWC numpy images for preprocess_images."""
    out: dict[str, np.ndarray] = {}
    for key in image_keys:
        if key not in sample:
            continue
        img = sample[key]
        if hasattr(img, "detach"):
            img = img.detach().cpu()
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            scaled = img * 255.0 if img.max() <= 1.0 else img
            img = np.clip(scaled, 0, 255).astype(np.uint8)
        out[key] = np.ascontiguousarray(img)
    return out


def _state_for_tokenizer(
    sample: dict,
    q01: np.ndarray | None,
    q99: np.ndarray | None,
) -> np.ndarray | None:
    if "observation.state" not in sample or q01 is None or q99 is None:
        return None
    st = sample["observation.state"]
    if hasattr(st, "detach"):
        st = st.detach().cpu().numpy()
    st = np.asarray(st, dtype=np.float32).reshape(-1)
    n = min(len(st), len(q01), len(q99))
    st = st[:n]
    q01 = q01[:n]
    q99 = q99[:n]
    denom = np.where(np.abs(q99 - q01) < 1e-6, 1.0, q99 - q01)
    norm = (st - q01) / denom * 2.0 - 1.0
    if len(norm) < 32:
        norm = np.pad(norm, (0, 32 - len(norm)))
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HVLA S2 latents for a LeRobotDataset",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Converted Pi0.5 VLM model.safetensors path",
    )
    parser.add_argument("--dataset", required=True, help="HF dataset repo_id (e.g. Odog16/tool_pickup)")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local dataset root",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Task string (match training and deployment)",
    )
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument(
        "--image-keys",
        required=True,
        help="Comma-separated dataset image keys",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    image_keys = tuple(k.strip() for k in args.image_keys.split(",") if k.strip())
    if not image_keys:
        raise SystemExit("No --image-keys provided")

    root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    if root is not None:
        ds = LeRobotDataset(args.dataset, root=root)
    else:
        ds = LeRobotDataset(args.dataset)
    n_total = len(ds)
    n_use = min(n_total, args.max_frames) if args.max_frames else n_total
    logger.info("Dataset %s: %d frames (using %d)", args.dataset, n_total, n_use)

    q01 = q99 = None
    if ds.meta.stats and "observation.state" in ds.meta.stats:
        st = ds.meta.stats["observation.state"]
        q01 = np.asarray(st["q01"], dtype=np.float32)
        q99 = np.asarray(st["q99"], dtype=np.float32)
        logger.info("Using dataset q01/q99 for state tokenization")
    else:
        logger.warning("No observation.state stats — tokenizer uses neutral state bins")

    device = torch.device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    config = S2VLMConfig()
    model = S2VLMModel.from_pretrained(args.checkpoint, config)
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latents = np.zeros((n_use, config.latent_dim), dtype=np.float32)

    bsz = max(1, args.batch_size)
    with torch.no_grad():
        pbar = tqdm(
            total=n_use,
            desc="S2 latents",
            unit="frm",
            mininterval=0.5,
        )
        for start in range(0, n_use, bsz):
            end = min(start + bsz, n_use)
            batch_latents = []
            for i in range(start, end):
                sample = ds[i]
                img_d = _sample_to_image_dict(sample, image_keys)
                state_norm = _state_for_tokenizer(sample, q01, q99)
                token_ids, token_mask = tokenizer.tokenize_prompt(
                    args.prompt, low_prompt="", state=state_norm, subtask_only=False,
                )
                lang_tokens = torch.from_numpy(token_ids).unsqueeze(0).long().to(device)
                lang_masks = torch.from_numpy(token_mask).unsqueeze(0).bool().to(device)

                image_tensors, image_masks = preprocess_images(
                    img_d,
                    image_keys=image_keys,
                    resolution=config.image_resolution,
                    device=device,
                )
                for li in range(len(image_tensors)):
                    image_tensors[li] = image_tensors[li].to(dtype=dtype)
                use_amp = dtype != torch.float32
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=use_amp,
                ):
                    z = model.extract_prefix_latent(
                        image_tensors,
                        image_masks,
                        lang_tokens,
                        lang_masks,
                    )
                batch_latents.append(z.float().cpu().numpy()[0])

            latents[start:end] = np.stack(batch_latents, axis=0)
            pbar.update(end - start)
            log_every = max(bsz * 50, 1)
            if start == 0 or end % log_every == 0 or end == n_use:
                logger.info("Extracted %d / %d frames", end, n_use)
        pbar.close()

    np.save(str(out_path), latents)
    logger.info("Wrote %s shape=%s", out_path, latents.shape)


if __name__ == "__main__":
    main()
