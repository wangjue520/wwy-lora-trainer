"""Trainable RMSNorm weight deltas with ComfyUI-compatible w_norm export."""
from __future__ import annotations

import torch
from lycoris.modules.base import LycorisBaseModule

from .kernels import fused_rmsnorm


class MossNormModule(LycorisBaseModule):
    name = "norm"
    weight_list = ["w_norm"]
    weight_list_det = ["w_norm"]

    def __init__(self, lora_name, org_module, multiplier=1.0,
                 rank_dropout=0.0, module_dropout=0.0, **kwargs):
        super().__init__(lora_name, org_module, multiplier,
                         rank_dropout=rank_dropout, module_dropout=module_dropout, **kwargs)
        if type(org_module).__name__ != "_RMSNorm" or org_module.weight.ndim != 1:
            raise ValueError("MOSS norm training requires the official affine RMSNorm")
        self.not_supported = False
        self.shape = tuple(org_module.weight.shape)
        self.w_norm = torch.nn.Parameter(torch.zeros(self.shape, dtype=torch.float32))
        self.use_fused = getattr(org_module, "_moss_fused_rmsnorm", False)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        delta = self.w_norm.to(device) * multiplier
        return delta.reshape(shape or self.shape), None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        delta, _ = self.get_diff_weight(multiplier, shape, device)
        return self.org_module[0].weight.to(device=device, dtype=delta.dtype).reshape(delta.shape) + delta, None

    def forward(self, value):
        if self.module_dropout and self.training and torch.rand((), device=value.device) < self.module_dropout:
            return self.org_forward(value)
        base = self.org_module[0].weight
        delta = torch.nn.functional.dropout(self.w_norm, self.rank_dropout, self.training)
        weight = (base.float() + delta * self.multiplier).to(base.dtype)
        if self.use_fused:
            return fused_rmsnorm(value, weight)
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * weight
