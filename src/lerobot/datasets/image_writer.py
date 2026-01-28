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
import multiprocessing
import queue
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    """Convert numpy array to PIL Image, supporting RGB (3 channels) and grayscale/depth (1 channel) images."""
    if image_array.ndim != 3:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 is expected for an image.")

    # Handle channel-first format (C, H, W) - common in PyTorch
    if image_array.shape[0] in (1, 3, 4):
        # Transpose from pytorch convention (C, H, W) to (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    # Get number of channels after transpose
    num_channels = image_array.shape[-1] if image_array.ndim == 3 else 1

    # Handle single-channel images (grayscale/depth)
    if num_channels == 1:
        # Squeeze to 2D (H, W) for grayscale PIL Image
        image_array = image_array.squeeze(axis=-1)
        mode = "L"  # Grayscale mode
    elif num_channels == 3:
        mode = "RGB"  # RGB mode
    elif num_channels == 4:
        mode = "RGBA"  # RGBA mode
    else:
        raise NotImplementedError(
            f"The image has {num_channels} channels. Supported: 1 (grayscale/depth), 3 (RGB), or 4 (RGBA)."
        )

    # Convert dtype if needed
    if image_array.dtype != np.uint8:
        if range_check and np.issubdtype(image_array.dtype, np.floating):
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            # For depth images, values might be in millimeters (uint16 range)
            # Check if values are in [0, 1] float range or larger integer range
            if max_ > 1.0 or min_ < 0.0:
                # Assume values are in a larger range (e.g., depth in mm)
                # Normalize to [0, 255] for uint8
                if max_ > min_:
                    image_array = ((image_array - min_) / (max_ - min_) * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)
            else:
                # Values are in [0, 1] range, scale to [0, 255]
                image_array = (image_array * 255).astype(np.uint8)
        else:
            # For integer types, clamp to uint8 range
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array, mode=mode)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1):
    """
    Saves a NumPy array or PIL Image to a file.

    This function handles both NumPy arrays and PIL Image objects, converting
    the former to a PIL Image before saving. It includes error handling for
    the save operation.

    Args:
        image (np.ndarray | PIL.Image.Image): The image data to save.
        fpath (Path): The destination file path for the image.
        compress_level (int, optional): The compression level for the saved
            image, as used by PIL.Image.save(). Defaults to 1.
            Refer to: https://github.com/huggingface/lerobot/pull/2135
            for more details on the default value rationale.

    Raises:
        TypeError: If the input 'image' is not a NumPy array or a
            PIL.Image.Image object.

    Side Effects:
        Prints an error message to the console if the image writing process
        fails for any reason.
    """
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath, compress_level=compress_level)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_loop(queue: queue.Queue):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath, compress_level = item
        write_image(image_array, fpath, compress_level)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchronously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("Number of threads and processes must be greater than zero.")

        if self.num_processes == 0:
            # Use threading
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # Use multiprocessing
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ):
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array to minimize main process time
            image = image.cpu().numpy()
        self.queue.put((image, fpath, compress_level))

    def wait_until_done(self):
        self.queue.join()

    def stop(self):
        if self._stopped:
            return

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True
