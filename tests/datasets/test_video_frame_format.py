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

"""``video_frame_format``: temporary per-frame files of a video recording, PNG (default) or raw TIFF."""

import numpy as np
import pytest

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

pytestmark = pytest.mark.dataset


def _record(tmp_path, fmt):
    features = {
        "observation.images.cam": {"dtype": "video", "shape": (64, 96, 3), "names": ["height", "width", "channels"]},
        "state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    ds = LeRobotDataset.create(f"local/fmt_{fmt}", 10, root=tmp_path / fmt, features=features, use_videos=True)
    ds.writer.video_frame_format = fmt
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 96, 3), dtype=np.uint8) for _ in range(6)]
    for f in frames:
        ds.add_frame({"observation.images.cam": f, "state": np.zeros(2, np.float32), "action": np.zeros(2, np.float32), "task": "t"})
    ds.writer.image_writer is None or ds.writer.image_writer.wait_until_done()
    tmp_files = sorted((ds.root / "images").rglob("frame-*"))
    ds.save_episode()
    ds.finalize()
    return ds, tmp_files


def test_default_frame_format_is_png():
    assert DatasetRecordConfig(repo_id="x/y", single_task="t").video_frame_format == "png"
    with pytest.raises(ValueError, match="video_frame_format"):
        DatasetRecordConfig(repo_id="x/y", single_task="t", video_frame_format="jpg")


@pytest.mark.parametrize("fmt", ["png", "tiff"])
def test_frames_are_written_in_the_requested_format_and_encode_into_a_video(tmp_path, fmt):
    ds, tmp_files = _record(tmp_path, fmt)
    assert len(tmp_files) == 6 and {f.suffix for f in tmp_files} == {f".{fmt}"}
    assert list((ds.root / "videos").rglob("*.mp4")), "the episode's video was not encoded"
    assert not (ds.root / "images").exists() or not list((ds.root / "images").rglob("frame-*"))  # temp frames deleted
    assert ds.num_frames == 6
