"""Shared-memory IPC for HVLA dual-system communication.

Uses multiprocessing.shared_memory.SharedMemory (named blocks) which works
correctly with the 'spawn' multiprocessing context. Each process attaches
by name — no pickling of large buffers.

Race protection: sequence counter incremented before and after write.
Reader retries if counters don't match (torn read).
"""

from __future__ import annotations

import logging
import multiprocessing.shared_memory
from multiprocessing import resource_tracker
import struct
import time

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Header layout (prepended to every SharedBlock):
#   [0:8]  int64  sequence_write (incremented before write)
#   [8:16] int64  sequence_done  (incremented after write, matches _write when consistent)
#   [16:24] float64 timestamp
_HEADER_SIZE = 24
_HEADER_FMT = "<qqd"  # two int64 + one float64


class SharedBlock:
    """A single named shared memory block with a consistency header.

    Supports any flat numpy array. Provides torn-read detection via
    sequence counters (writer increments before/after, reader checks match).
    """

    def __init__(self, name: str | None, shape: tuple[int, ...], dtype: np.dtype, create: bool = False):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._owner = create  # only the owner should unlink on cleanup
        self._data_size = int(np.prod(shape)) * self.dtype.itemsize
        self._total_size = _HEADER_SIZE + self._data_size

        # Suppress resource_tracker register/unregister — we manage lifecycle
        # explicitly. Without this, the tracker auto-unlinks shared memory on
        # process exit (destroys persistent S2's memory when S1 exits) and
        # causes KeyError spam on Python 3.10.
        _orig_register = resource_tracker.register
        _orig_unregister = resource_tracker.unregister
        resource_tracker.register = lambda *a, **kw: None
        resource_tracker.unregister = lambda *a, **kw: None
        try:
            if create:
                # Clean up stale shared memory from previous crashed runs
                if name is not None:
                    try:
                        old = multiprocessing.shared_memory.SharedMemory(name=name)
                        old.close()
                        old.unlink()
                    except FileNotFoundError:
                        pass
                self._shm = multiprocessing.shared_memory.SharedMemory(
                    name=name, create=True, size=self._total_size,
                )
                self._shm.buf[:_HEADER_SIZE] = struct.pack(_HEADER_FMT, 0, 0, 0.0)
                self._shm.buf[_HEADER_SIZE:] = b'\x00' * self._data_size
            else:
                self._shm = multiprocessing.shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = _orig_register
            resource_tracker.unregister = _orig_unregister

        self.shm_name = self._shm.name
        self._view: np.ndarray | None = None

    def _ensure_view(self):
        if self._view is None:
            self._view = np.ndarray(self.shape, dtype=self.dtype,
                                    buffer=self._shm.buf, offset=_HEADER_SIZE)

    def write(self, data: np.ndarray):
        """Write data with sequence counter protection."""
        self._ensure_view()
        buf = self._shm.buf

        # Read current sequence, increment write counter
        seq_write, seq_done, _ = struct.unpack_from(_HEADER_FMT, buf, 0)
        new_seq = seq_write + 1
        struct.pack_into("<q", buf, 0, new_seq)  # seq_write = new_seq

        # Write data
        np.copyto(self._view, data)

        # Update timestamp and done counter
        struct.pack_into("<qd", buf, 8, new_seq, time.time())  # seq_done = new_seq, timestamp

    def read(self, max_retries: int = 3) -> tuple[np.ndarray, float]:
        """Read data with torn-read detection. Returns (data_copy, timestamp)."""
        self._ensure_view()
        buf = self._shm.buf

        for _ in range(max_retries):
            seq_write, seq_done, ts = struct.unpack_from(_HEADER_FMT, buf, 0)
            data = self._view.copy()
            # Re-read sequence to check consistency
            seq_write2, seq_done2, ts2 = struct.unpack_from(_HEADER_FMT, buf, 0)

            if seq_done == seq_write and seq_done2 == seq_write2 and seq_done == seq_done2:
                return data, ts

        # After retries, return best-effort (may be torn)
        return self._view.copy(), ts

    @property
    def count(self) -> int:
        """Number of completed writes."""
        _, seq_done, _ = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        return seq_done

    @property
    def timestamp(self) -> float:
        _, _, ts = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        return ts

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        _orig = resource_tracker.unregister
        resource_tracker.unregister = lambda *a, **kw: None
        try:
            self._shm.unlink()
        except Exception:
            pass
        finally:
            resource_tracker.unregister = _orig

    def __getstate__(self):
        return {"shm_name": self.shm_name, "shape": self.shape, "dtype": self.dtype.str}

    def __setstate__(self, state):
        self.shape = state["shape"]
        self.dtype = np.dtype(state["dtype"])
        self._data_size = int(np.prod(self.shape)) * self.dtype.itemsize
        self._total_size = _HEADER_SIZE + self._data_size
        self.shm_name = state["shm_name"]
        self._shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
        self._view = None


class SharedLatentCache:
    """S2 → S1 latent communication via shared memory.

    S2 calls write(latent_tensor). S1 calls read() or read_with_age().
    Torn-read protected via sequence counters in SharedBlock.

    Optionally stores a subtask text string (up to 256 bytes, UTF-8) in a
    separate shared block for GUI overlay display.

    Args:
        latent_dim: size of the latent vector.
        create: True to create new shared memory (S2 side), False to attach (S1 side).
        name: well-known name for persistent S2. If None, auto-generated.
    """

    SHM_NAME = "hvla_s2_latent"
    _SUBTASK_MAX_BYTES = 256

    def __init__(self, latent_dim: int = 2048, create: bool = True, name: str | None = None):
        self.latent_dim = latent_dim
        shm_name = name or self.SHM_NAME
        self._block = SharedBlock(name=shm_name, shape=(latent_dim,), dtype=np.float32, create=create)
        self._subtask_block = SharedBlock(
            name=shm_name + "_subtask", shape=(self._SUBTASK_MAX_BYTES,), dtype=np.uint8, create=create,
        )
        self._confidence_block = SharedBlock(
            name=shm_name + "_conf", shape=(1,), dtype=np.float32, create=create,
        )

    def write(self, latent: Tensor, subtask: str | None = None, confidence: float = 0.0) -> None:
        self._block.write(latent.detach().cpu().to(torch.float32).numpy())
        if subtask is not None:
            encoded = subtask.encode("utf-8")[:self._SUBTASK_MAX_BYTES]
            buf = np.zeros(self._SUBTASK_MAX_BYTES, dtype=np.uint8)
            buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
            self._subtask_block.write(buf)
            self._confidence_block.write(np.array([confidence], dtype=np.float32))

    def read(self) -> tuple[Tensor, float]:
        data, ts = self._block.read()
        return torch.from_numpy(data), ts

    def read_subtask(self) -> tuple[str, float, float]:
        """Read the latest subtask text + confidence.

        Returns (text, timestamp, confidence). ("", 0.0, 0.0) if never written.
        """
        if self._subtask_block.count == 0:
            return "", 0.0, 0.0
        data, ts = self._subtask_block.read()
        text = bytes(data).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        conf = 0.0
        if self._confidence_block.count > 0:
            conf_data, _ = self._confidence_block.read()
            conf = float(conf_data[0])
        return text, ts, conf

    def read_with_age(self) -> tuple[Tensor, float]:
        data, ts = self._block.read()
        age = (time.time() - ts) if ts > 0 else 0.0
        return torch.from_numpy(data), age

    @property
    def age_ms(self) -> float:
        ts = self._block.timestamp
        return (time.time() - ts) * 1000 if ts > 0 else float("inf")

    @property
    def count(self) -> int:
        return self._block.count

    def wait_for_first(self, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._block.count > 0:
                return True
            time.sleep(0.1)
        return self._block.count > 0

    def cleanup(self):
        self._block.close()
        self._subtask_block.close()
        self._confidence_block.close()
        if self._block._owner:
            self._block.unlink()
            self._subtask_block.unlink()
            self._confidence_block.unlink()


class SharedImageBuffer:
    """S1 → S2 camera image sharing via shared memory.

    One SharedBlock per camera + one for joint state.
    S1 calls write_images(), S2 calls read_images().

    Args:
        create: True to create (S1/launcher side), False to attach (S2 standalone).
    """

    SHM_PREFIX = "hvla_img_"

    def __init__(self, camera_keys: tuple[str, ...], height: int = 720, width: int = 1280, channels: int = 3,
                 create: bool = True, state_dim: int = 32):
        self.camera_keys = camera_keys
        self.height = height
        self.width = width
        self.channels = channels
        self.state_dim = state_dim

        self._cam_blocks: dict[str, SharedBlock] = {}
        for key in camera_keys:
            shm_name = f"{self.SHM_PREFIX}{key}"
            self._cam_blocks[key] = SharedBlock(
                name=shm_name, shape=(height, width, channels), dtype=np.uint8, create=create,
            )

        self._state_block = SharedBlock(
            name=f"{self.SHM_PREFIX}_state", shape=(state_dim,), dtype=np.float32, create=create,
        )

    def write_images(self, robot_obs: dict, cam_key_map: dict[str, str], joint_names: list[str]):
        # Optional resize so xlerobot (e.g. 360×640) matches SHM layout without silent skips.
        try:
            import cv2
        except ImportError:
            cv2 = None

        for s1_name, s2_name in cam_key_map.items():
            if s1_name not in robot_obs or s2_name not in self._cam_blocks:
                continue
            img = np.asarray(robot_obs[s1_name])
            if img.ndim != 3 or img.shape[2] != self.channels:
                continue
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[0] != self.height or img.shape[1] != self.width:
                if cv2 is None:
                    logger.warning(
                        "HVLA SHM: image %s shape %s != (%d,%d,%d) and opencv missing — skip",
                        s1_name, img.shape, self.height, self.width, self.channels,
                    )
                    continue
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            self._cam_blocks[s2_name].write(img)

        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, name in enumerate(joint_names):
            if name in robot_obs and i < self.state_dim:
                state[i] = float(robot_obs[name])
        self._state_block.write(state)

    def read_images(self) -> dict[str, np.ndarray] | None:
        if self._state_block.count == 0:
            return None
        result = {}
        for key in self.camera_keys:
            data, _ = self._cam_blocks[key].read()
            result[key] = data
        state, _ = self._state_block.read()
        result["_state"] = state
        return result

    @property
    def is_ready(self) -> bool:
        return self._state_block.count > 0

    def cleanup(self):
        for block in self._cam_blocks.values():
            block.close()
            if block._owner:
                block.unlink()
        self._state_block.close()
        if self._state_block._owner:
            self._state_block.unlink()


# =============================================================================
# ZMQ-based IPC for cross-machine HVLA (GPU machine ↔ Jetson)
#
# Wire layout
# -----------
# Latent messages (multipart, 2 frames):
#   Frame 0: float32[latent_dim] + float64[1 timestamp] packed as bytes
#            = latent_dim*4 + 8 bytes total
#   Frame 1: JSON bytes {"subtask": "...", "confidence": 0.0}
#
# Image messages (single JSON string frame):
#   {"_state": [f32 × state_dim], "<s2_key>": "<base64-JPEG>", ...}
#
# Ports (defaults, all overridable via CLI):
#   5580 — ZMQ PUB/SUB for robot images (Jetson → GPU)
#   5581 — ZMQ PUB/SUB for S2 latent   (GPU → Jetson)
# =============================================================================


class ZMQLatentPublisher:
    """S2-side: publishes latent vectors over ZMQ PUB.

    Bound on the GPU machine. Drop-in for the write() half of SharedLatentCache.
    """

    DEFAULT_PORT = 5581

    def __init__(self, latent_dim: int = 2048, port: int = DEFAULT_PORT):
        import zmq  # lazy — not required when running same-machine mode
        self.latent_dim = latent_dim
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        # Keep only the last unsent message — avoids queue buildup during slow receivers
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.bind(f"tcp://*:{port}")
        self._count = 0
        logger.info("ZMQLatentPublisher: bound on port %d (latent_dim=%d)", port, latent_dim)

    def write(self, latent: "Tensor", subtask: str | None = None, confidence: float = 0.0) -> None:
        import json as _json
        import zmq
        lat_np = latent.detach().cpu().to(torch.float32).numpy().flatten()[:self.latent_dim]
        ts = np.array([time.time()], dtype=np.float64)
        frame0 = lat_np.tobytes() + ts.tobytes()
        frame1 = _json.dumps({"subtask": subtask or "", "confidence": float(confidence)}).encode()
        self._sock.send_multipart([frame0, frame1], flags=zmq.NOBLOCK)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def cleanup(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass


class ZMQLatentSubscriber:
    """S1-side: subscribes to S2 latent vectors over ZMQ SUB.

    Connects to the GPU machine. Drop-in for the read/age/count half of
    SharedLatentCache. CONFLATE=1 keeps only the latest message — mirrors
    shared-memory semantics where S1 always reads the most recent write.
    """

    DEFAULT_PORT = 5581

    def __init__(self, host: str, port: int = DEFAULT_PORT, latent_dim: int = 2048):
        import zmq
        self.latent_dim = latent_dim
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        # Drop queued-up messages — always serve latest (matches SHM behaviour)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._latent = torch.zeros(latent_dim, dtype=torch.float32)
        self._timestamp = 0.0
        self._subtask = ""
        self._confidence = 0.0
        self._count = 0
        logger.info("ZMQLatentSubscriber: connected to %s:%d", host, port)

    def _poll(self) -> bool:
        """Non-blocking receive. Returns True if a new latent arrived."""
        import zmq
        try:
            frames = self._sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return False
        try:
            frame0 = frames[0]
            lat_bytes = frame0[: self.latent_dim * 4]
            ts_bytes = frame0[self.latent_dim * 4: self.latent_dim * 4 + 8]
            self._latent = torch.from_numpy(np.frombuffer(lat_bytes, dtype=np.float32).copy())
            self._timestamp = float(np.frombuffer(ts_bytes, dtype=np.float64)[0])
            if len(frames) > 1:
                import json as _json
                meta = _json.loads(frames[1])
                self._subtask = meta.get("subtask", "")
                self._confidence = float(meta.get("confidence", 0.0))
            self._count += 1
            return True
        except Exception as exc:
            logger.warning("ZMQLatentSubscriber: parse error: %s", exc)
            return False

    def read(self) -> tuple["Tensor", float]:
        self._poll()
        return self._latent, self._timestamp

    def read_with_age(self) -> tuple["Tensor", float]:
        self._poll()
        age = (time.time() - self._timestamp) if self._timestamp > 0 else 0.0
        return self._latent, age

    def read_subtask(self) -> tuple[str, float, float]:
        return self._subtask, self._timestamp, self._confidence

    @property
    def age_ms(self) -> float:
        self._poll()
        return (time.time() - self._timestamp) * 1000 if self._timestamp > 0 else float("inf")

    @property
    def count(self) -> int:
        return self._count

    def wait_for_first(self, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._poll():
                return True
            time.sleep(0.1)
        return self._count > 0

    def cleanup(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass


class ZMQImagePublisher:
    """S1-side: publishes robot camera frames + joint state over ZMQ PUB.

    Bound on Jetson. Drop-in for the write_images() half of SharedImageBuffer.
    Images are JPEG-compressed to keep bandwidth ~1-3 MB/s at 10 Hz.
    """

    DEFAULT_PORT = 5580

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        height: int = 360,
        width: int = 640,
        port: int = DEFAULT_PORT,
        jpeg_quality: int = 85,
        state_dim: int = 32,
    ):
        import zmq
        self.camera_keys = camera_keys
        self.height = height
        self.width = width
        self._jpeg_quality = jpeg_quality
        self._state_dim = state_dim
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.bind(f"tcp://*:{port}")
        self._count = 0
        logger.info(
            "ZMQImagePublisher: bound on port %d (%dx%d, %d cameras, JPEG q=%d)",
            port, height, width, len(camera_keys), jpeg_quality,
        )

    def write_images(self, robot_obs: dict, cam_key_map: dict[str, str], joint_names: list[str]):
        import base64
        import json as _json
        import zmq
        try:
            import cv2
        except ImportError:
            logger.warning("ZMQImagePublisher: opencv missing, cannot encode images")
            return

        payload: dict = {}

        # Joint state (pad to state_dim)
        state = np.zeros(self._state_dim, dtype=np.float32)
        for i, name in enumerate(joint_names):
            if name in robot_obs and i < self._state_dim:
                state[i] = float(robot_obs[name])
        payload["_state"] = state.tolist()

        # Camera images → JPEG → base64
        for s1_key, s2_key in cam_key_map.items():
            if s1_key not in robot_obs:
                continue
            img = np.asarray(robot_obs[s1_key])
            if img.ndim != 3:
                continue
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if ret:
                payload[s2_key] = base64.b64encode(buf).decode("utf-8")

        try:
            self._sock.send_string(_json.dumps(payload), flags=zmq.NOBLOCK)
            self._count += 1
        except zmq.Again:
            pass

    @property
    def is_ready(self) -> bool:
        return True

    def cleanup(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass


class ZMQImageSubscriber:
    """S2-side: subscribes to robot camera frames + state from S1 (Jetson).

    Connects to Jetson. Drop-in for the read_images() half of SharedImageBuffer.
    CONFLATE=1 keeps only the latest frame — mirrors SHM semantics.
    """

    DEFAULT_PORT = 5580

    def __init__(self, host: str, port: int = DEFAULT_PORT, camera_keys: tuple[str, ...] = ()):
        import zmq
        self.camera_keys = camera_keys
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._last: dict | None = None
        self._count = 0
        logger.info("ZMQImageSubscriber: connected to %s:%d", host, port)

    def read_images(self) -> dict[str, np.ndarray] | None:
        import base64
        import json as _json
        import zmq
        try:
            import cv2
        except ImportError:
            cv2 = None

        try:
            raw = self._sock.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return self._last

        try:
            payload = _json.loads(raw)
        except Exception as exc:
            logger.warning("ZMQImageSubscriber: JSON parse error: %s", exc)
            return self._last

        result: dict = {}

        state_list = payload.get("_state")
        if state_list is not None:
            result["_state"] = np.array(state_list, dtype=np.float32)

        for key, val in payload.items():
            if key == "_state" or not val:
                continue
            try:
                jpg_data = base64.b64decode(val)
                np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                if cv2 is not None:
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        result[key] = frame
            except Exception as exc:
                logger.warning("ZMQImageSubscriber: image decode error for %s: %s", key, exc)

        self._last = result
        self._count += 1
        return result

    @property
    def is_ready(self) -> bool:
        return self._count > 0

    def cleanup(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass


# Default S1 → S2 camera key mapping for SO107 bimanual (robot obs keys → S2 shared-memory slots)
DEFAULT_S2_CAM_KEY_MAP = {
    "front": "base_0_rgb",
    "top": "base_1_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}

# xlerobot get_observation() uses short camera keys (head / left_wrist / right_wrist)
XLEROBOT_S2_CAM_KEY_MAP = {
    "head": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}

DEFAULT_S2_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb")


def parse_shm_image_size(s: str) -> tuple[int, int]:
    """Parse 'HxW' or 'H*W' into (height, width)."""
    a, b = s.lower().replace("*", "x").split("x", 1)
    return int(a.strip()), int(b.strip())


def parse_s2_camera_map(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


