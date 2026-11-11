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
"""Export a depth camera of one episode as a colour-mapped mp4 that any browser or player can show.

Depth is stored losslessly as 12-bit grayscale HEVC (``gray12le``, millimetres on a log scale), which
browsers cannot decode, so the Hugging Face dataset visualizer shows a blank panel for it. The data is fine;
this decodes it with LeRobot and writes an ordinary h264 preview.

    python -m lerobot.data_processing.depth_preview --repo-id <user/dataset> [--root <dir>] --episode 0 \\
        --out depth_preview.mp4 [--with-rgb]

Near is red-to-blue far (turbo colour map), pixels without a depth reading are black.
"""

import argparse
from pathlib import Path

import numpy as np

# The depth codec stores "no reading" (0) at its lower bound, which decodes to ~10 mm.
INVALID_BELOW_MM = 15.0


def colorize_depth(depth_mm: np.ndarray, near_mm: float = 200.0, far_mm: float = 3000.0) -> np.ndarray:
    """(H, W) or (1, H, W) depth in mm -> (H, W, 3) uint8 RGB. Near is red, far is blue, invalid is black."""
    import cv2

    d = np.asarray(depth_mm, dtype=np.float32)
    d = d.reshape(d.shape[-2], d.shape[-1]) if d.ndim == 3 else d
    # turbo runs blue (low) to red (high), so invert: close things should be warm
    norm = 1.0 - np.clip((d - near_mm) / max(far_mm - near_mm, 1.0), 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[d < INVALID_BELOW_MM] = 0
    return rgb


def export_preview(
    repo_id: str,
    out: Path,
    root: Path | None = None,
    episode: int = 0,
    camera: str = "observation.images.head_depth",
    near_mm: float = 200.0,
    far_mm: float = 3000.0,
    with_rgb: bool = False,
) -> int:
    """Write the preview and return the number of frames."""
    import av
    import cv2

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    if camera not in ds.meta.features:
        depth_cams = [k for k, v in ds.meta.features.items() if v.get("info", {}).get("is_depth_map")]
        raise SystemExit(f"{camera!r} not in dataset. Depth cameras: {depth_cams or 'none'}")
    rgb_key = camera.removesuffix("_depth") if with_rgb else None
    if rgb_key is not None and rgb_key not in ds.meta.features:
        rgb_key = None

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), "w")
    stream = None
    n = 0
    try:
        for i in range(len(ds)):
            item = ds[i]
            frame = colorize_depth(item[camera].numpy(), near_mm, far_mm)
            if rgb_key is not None:
                rgb = (item[rgb_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb = cv2.resize(rgb, (int(rgb.shape[1] * frame.shape[0] / rgb.shape[0]), frame.shape[0]))
                frame = np.concatenate([rgb, frame], axis=1)
            h, w = frame.shape[:2]
            if stream is None:
                stream = container.add_stream("libx264", ds.fps, options={"crf": "20", "preset": "veryfast"})
                stream.width, stream.height, stream.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
            vf = av.VideoFrame.from_ndarray(frame[: stream.height, : stream.width], format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
            n += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, default=None, help="local dataset folder (default: the cache)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", default="observation.images.head_depth")
    p.add_argument("--out", type=Path, default=Path("depth_preview.mp4"))
    p.add_argument("--near-mm", type=float, default=200.0)
    p.add_argument("--far-mm", type=float, default=3000.0)
    p.add_argument("--with-rgb", action="store_true", help="put the matching RGB camera beside the depth")
    a = p.parse_args()
    n = export_preview(a.repo_id, a.out, a.root, a.episode, a.camera, a.near_mm, a.far_mm, a.with_rgb)
    print(f"wrote {n} frames to {a.out}")


if __name__ == "__main__":
    main()
