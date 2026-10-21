"""Resolution selection matching kohya-ss/sd-scripts BucketManager.

Reference: library/dataset.py and library/model_util.py in sd-scripts.
The formulas below implement its predefined and no-upscale modes.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .config import Dataset


class BucketManager:
    def __init__(self, config: Dataset):
        self.config = config
        self.area = math.prod(config.resolution)
        self.step = config.bucket_reso_steps
        square = math.isqrt(self.area) // self.step * self.step
        if config.enable_bucket and square == 0:
            raise ValueError("resolution area must be at least bucket_reso_steps squared")
        sizes = {(square, square)}
        for width in range(config.min_bucket_reso, config.max_bucket_reso + 1, self.step):
            height = min(config.max_bucket_reso, (self.area // width) // self.step * self.step)
            if height >= config.min_bucket_reso:
                sizes.update(((width, height), (height, width)))
        self.resolutions = sorted(sizes)

    def _round(self, value: float) -> int:
        return int(value + 0.5) // self.step * self.step

    def select(self, width: int, height: int):
        if min(width, height) <= 0:
            raise ValueError("Image dimensions must be positive")
        ratio = width / height
        if not self.config.enable_bucket:
            bucket = self.config.resolution
            scale = max(bucket[0] / width, bucket[1] / height)
            if self.config.bucket_no_upscale and scale > 1:
                raise ValueError("Image is too small for fixed resolution with bucket_no_upscale")
            resized = (int(width * scale + 0.5), int(height * scale + 0.5))
        elif not self.config.bucket_no_upscale:
            bucket = (width, height)
            if bucket not in self.resolutions:
                bucket = min(self.resolutions, key=lambda size: abs(size[0] / size[1] - ratio))
            scale = max(bucket[0] / width, bucket[1] / height)
            resized = (int(width * scale + 0.5), int(height * scale + 0.5))
        else:
            if width * height > self.area:
                scaled_w = math.sqrt(self.area * ratio)
                scaled_h = self.area / scaled_w
                rounded_w = self._round(scaled_w)
                height_for_w = self._round(rounded_w / ratio)
                rounded_h = self._round(scaled_h)
                width_for_h = self._round(rounded_h * ratio)
                if min(rounded_w, height_for_w, rounded_h, width_for_h) <= 0:
                    raise ValueError("Image aspect ratio is too extreme for bucket_reso_steps")
                if abs(rounded_w / height_for_w - ratio) < abs(width_for_h / rounded_h - ratio):
                    resized = (rounded_w, int(rounded_w / ratio + 0.5))
                else:
                    resized = (int(rounded_h * ratio + 0.5), rounded_h)
            else:
                resized = (width, height)
            bucket = tuple(value // self.step * self.step for value in resized)
            if min(bucket) == 0:
                raise ValueError("Image is smaller than bucket_reso_steps; no-upscale cannot produce a valid bucket")
        # Deterministic center crop keeps cached latents and training geometry identical.
        left, top = ((resized[i] - bucket[i]) // 2 for i in (0, 1))
        crop = (left, top, left + bucket[0], top + bucket[1])
        return bucket, resized, crop


class BucketBatchSampler:
    """Shuffle within buckets and shuffle batches, preserving incomplete tail batches."""

    def __init__(self, records, batch_size: int, repeats: int, seed: int):
        if type(batch_size) is not int or batch_size < 1 or type(repeats) is not int or repeats < 1:
            raise ValueError("batch_size and repeats must be positive integers")
        self.groups = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[tuple(record["bucket"])].extend([index] * repeats)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.start_batch = 0

    @property
    def total_batches(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups.values())

    def set_epoch(self, epoch, start_batch=0):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or not 0 <= start_batch <= self.total_batches:
            raise ValueError("start_batch is outside the bucket batch list")
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return self.total_batches - self.start_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for group in self.groups.values():
            indices = group.copy()
            rng.shuffle(indices)
            batches.extend(indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size))
        rng.shuffle(batches)
        yield from batches[self.start_batch:]
