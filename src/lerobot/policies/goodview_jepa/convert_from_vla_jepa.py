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
"""Turn a VLA-JEPA checkpoint into a goodview_jepa one (same weights, new policy type).

The weights and processor files are copied unchanged; only `config.json` gets
`"type": "goodview_jepa"` and the good-view fields. Then train from it with `--policy.path`:

    python -m lerobot.policies.goodview_jepa.convert_from_vla_jepa \\
        --src lerobot/VLA-JEPA-Pretrain --dst outputs/goodview_jepa_pretrain --cross_view_mode future
    lerobot-train --policy.path=outputs/goodview_jepa_pretrain --dataset.repo_id=... ...

Any field can still be overridden on the command line (e.g. `--policy.cross_view_mode=same_time`).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .configuration_goodview_jepa import CROSS_VIEW_MODES


def convert(
    src: str,
    dst: str | Path,
    cross_view_mode: str = "future",
    cross_view_source: int = 0,
    cross_view_dest: int = 1,
    video_frame_stride: int | None = None,
    camera_blank_prob: float = 0.0,
    revision: str | None = None,
) -> Path:
    if cross_view_mode not in CROSS_VIEW_MODES:
        raise ValueError(f"cross_view_mode must be one of {CROSS_VIEW_MODES}, got {cross_view_mode!r}")
    src_path = Path(src)
    if not src_path.is_dir():
        from huggingface_hub import snapshot_download

        src_path = Path(snapshot_download(src, revision=revision))
    config = json.loads((src_path / "config.json").read_text())
    if config.get("type") not in ("vla_jepa", "goodview_jepa"):
        raise ValueError(f"{src} is a {config.get('type')!r} checkpoint, not vla_jepa.")

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        if item.name == "config.json" or item.name.startswith("."):
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)  # follows the hub cache symlinks

    config.update(
        type="goodview_jepa",
        cross_view_mode=cross_view_mode,
        cross_view_source=cross_view_source,
        cross_view_dest=cross_view_dest,
        video_frame_stride=video_frame_stride,
        camera_blank_prob=camera_blank_prob,
    )
    (dst / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="hub repo id or local VLA-JEPA checkpoint dir")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--cross_view_mode", choices=CROSS_VIEW_MODES, default="future")
    ap.add_argument("--cross_view_source", type=int, default=0)
    ap.add_argument("--cross_view_dest", type=int, default=1)
    ap.add_argument("--video_frame_stride", type=int, default=None)
    ap.add_argument("--camera_blank_prob", type=float, default=0.0)
    args = ap.parse_args()
    out = convert(
        args.src,
        args.dst,
        cross_view_mode=args.cross_view_mode,
        cross_view_source=args.cross_view_source,
        cross_view_dest=args.cross_view_dest,
        video_frame_stride=args.video_frame_stride,
        camera_blank_prob=args.camera_blank_prob,
        revision=args.revision,
    )
    print(f"wrote goodview_jepa checkpoint to {out}")


if __name__ == "__main__":
    main()
