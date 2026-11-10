# -*- coding: utf-8 -*-
"""Anima 2.9B（Gazingstars123/Anima-2.9B）—— Anima 的加深版，同样用 sd-scripts 的 anima_train_network.py。

和 Anima 2B 的区别只有 DiT 层数：28 层 → 40 层（在原来的层之间插入 12 层，从相邻层复制权重、输出投影清零），
宽度、注意力头、LLM Adapter、文本编码器（Qwen3-0.6B）、VAE（Qwen-Image VAE）全都一样。
sd-scripts 把 Anima 写死成 28 层，按文件实际层数加载的补丁在 app/vendor/sitecustomize.py（「Anima 加深版」）。
2B 和 2.9B 的 LoRA 不能混用：层号对不上（2.9B 的第 2 层是新插入的层）。"""
import re

from .base import Slot
from .anima import Anima

REPO = "Gazingstars123/Anima-2.9B"


class Anima29(Anima):
    id = "anima29"
    label = "Anima 2.9B"
    subtitle = "Anima 加深版 · 40 层 2.9B · Qwen3 文本编码器 · sd-scripts"
    MAX_SWAP = 38   # sd-scripts：最多交换「层数 − 2」层
    slots = [
        Slot("dit", "主模型（Anima 2.9B DiT）", ["anima29_dit"],
             help="40 层的 Anima 2.9B。int8 ConvRot 版 sd-scripts 不能训练，要用 bf16 版。",
             downloads=[{"label": "Anima 2.9B preview v1（bf16）", "repo": REPO,
                         "file": "Anima-2.9B-preview-v1.safetensors", "size_gb": 5.84}]),
    ] + [s for s in Anima.slots if s.key != "dit"]
    # 模型说明：沿用 Anima 的提示词写法；训练数据里没有 score 标签（写了也能用）；推荐 28–50 步、CFG 3.5–5
    sample_defaults = dict(Anima.sample_defaults, steps=30, cfg=4.0)

    def is_official(self, name):
        return bool(re.match(r"anima[-_ ]?2\.9b", name))

    def score(self, slot_key, item):
        s = super().score(slot_key, item)
        if slot_key == "dit" and re.search(r"2\.9b", item["name"].lower()):
            s += 100
        return s

    def vram_presets(self):
        # 权重比 2B 大 40%（bf16 5.8 GB / 4.2 GB），同样显存下多交换一些层；批量和 2B 一样
        lr = {"learning_rate": 2e-4, "optimizer_type": "AdamW8bit", "optimizer_args": "", "lr_scheduler": "cosine",
              "lr_warmup_steps": 0, "lr_scheduler_args": "", "max_grad_norm": 1.0, "loss_type": "l2", "train_te": False}
        return [
            {"id": "8g", "label": "8 GB", "min_gb": 7.5, "note": "1024 分辨率，块交换 26，AdamW8bit 2e-4",
             "values": {"batch_size": 1, "resolution": 1024, "blocks_to_swap": 26, "gradient_checkpointing": True,
                        "num_workers": 1, "unsloth_offload": False, **lr}},
            {"id": "12g", "label": "12 GB", "min_gb": 11.5, "note": "批量 2，块交换 14，AdamW8bit 2e-4",
             "values": {"batch_size": 2, "resolution": 1024, "blocks_to_swap": 14, "gradient_checkpointing": True,
                        "num_workers": 2, "unsloth_offload": False, **lr}},
            {"id": "16g", "label": "16 GB", "min_gb": 15.5, "note": "批量 2，块交换 6",
             "values": {"batch_size": 2, "resolution": 1024, "blocks_to_swap": 6, "gradient_checkpointing": True,
                        "num_workers": 2}},
            {"id": "24g", "label": "24 GB", "min_gb": 23, "note": "批量 4",
             "values": {"batch_size": 4, "resolution": 1024, "blocks_to_swap": 0, "gradient_checkpointing": True,
                        "num_workers": 2}},
            {"id": "32g", "label": "32 GB+", "min_gb": 31, "note": "批量 6，关梯度检查点更快",
             "values": {"batch_size": 6, "resolution": 1024, "blocks_to_swap": 0, "gradient_checkpointing": False,
                        "num_workers": 4}},
        ]

    def params(self):
        f = super().params()
        for x in f:
            if x["key"] == "output_name":
                x["default"] = "mylora_anima29_v1"
        return f
