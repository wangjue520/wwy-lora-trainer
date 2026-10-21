"""Packed MOSS batches without padding or cross-sample attention.

Only forward methods are replaced; parameter names and exported weights stay intact.
Sequence metadata is passed explicitly so checkpoint recomputation is independent
of ambient state. The original single-sample path remains available.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch
from torch.nn.attention.varlen import varlen_attn


@dataclass(frozen=True)
class PackedPositions:
    rope: torch.Tensor
    cumulative: torch.Tensor
    maximum: int


def _positions(ropes, lengths):
    rope = torch.cat(ropes, dim=1)
    cumulative = torch.tensor([0, *lengths], device=rope.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return PackedPositions(rope, cumulative, max(lengths))


def _attention(q, k, v, positions, heads):
    # Leading dimension is one packed stream, not a sample batch dimension.
    shape = (1, q.shape[1], heads, q.shape[-1] // heads)
    sin, cos = positions.rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)

    def rotate(value):
        value = value.reshape(shape)
        a, b = value.chunk(2, -1)
        return (value * cos + torch.cat((-b, a), -1) * sin).to(v.dtype)[0]

    out = varlen_attn(
        rotate(q), rotate(k), v.reshape(shape)[0],
        positions.cumulative, positions.cumulative,
        positions.maximum, positions.maximum,
    )
    return out.flatten(1).unsqueeze(0)


def _text_forward(self, x, rope):
    if not isinstance(rope, PackedPositions):
        return self._sequential_forward(x, rope)
    q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
    q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
    x = x + self.proj(_attention(q, k, v, rope, self.heads))
    y = self.norm2(x)
    return x + self.ffn_down(self._packed_situ(self.ffn_up(y), self.ffn_gate(y)))


def _block_forward(self, x, rope, modulation):
    if not isinstance(rope, PackedPositions):
        return self._sequential_forward(x, rope, modulation)
    mod, owners = modulation
    def condition(index):
        return mod[:, index][owners].unsqueeze(0)
    y = (self.norm1(x) * (1 + condition(1)) + condition(0)).to(self.proj['q'].weight.dtype)
    q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads)) * condition(2)
    y = self.norm2(x) * (1 + condition(4)) + condition(3)
    hidden, gate = self.proj['ffn_up_gate'](y.to(self.proj['q'].weight.dtype)).chunk(2, -1)
    return x + self.proj['ffn_down'](self._packed_situ(hidden, gate)) * condition(5)


def _forward(self, hidden_states, timestep, encoder_hidden_states,
             encoder_attention_mask=None, return_dict=True, skip_layers=None):
    # Preserve the existing path (and stochastic adapter semantics) for BS=1.
    if hidden_states.shape[0] == 1:
        return self._sequential_forward(hidden_states, timestep, encoder_hidden_states,
                                        encoder_attention_mask, return_dict, skip_layers)
    module = sys.modules[type(self).__module__]
    skip_layers = module._validate_skip_layers(skip_layers, len(self.layers))
    if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
        raise ValueError('hidden_states must be [B, latent_channels, H, W].')
    batch, _, height, width = hidden_states.shape
    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
        raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
    mask = encoder_attention_mask
    if mask is None:
        mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
    mask = mask.bool()
    lengths = mask.sum(1).tolist()
    if not all(lengths):
        raise ValueError('Each sample needs at least one valid text token.')
    times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
    half = self.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
    head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
    video = hidden_states.flatten(2).transpose(1, 2)
    video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
    image_length = height * width
    ropes = [self.rope(height, width, length, hidden_states.device).to(video.dtype) for length in lengths]
    text_positions = _positions([rope[:, image_length:] for rope in ropes], lengths)
    text = self.text_embedder(encoder_hidden_states[mask].unsqueeze(0), text_positions)
    chunks = text.split(lengths, dim=1)
    joint_lengths = [image_length + length for length in lengths]
    positions = _positions(ropes, joint_lengths)
    x = torch.cat([torch.cat((video[i:i+1], chunks[i]), dim=1) for i in range(batch)], dim=1)
    # Each packed token receives its own sample's timestep modulation.
    owners = torch.repeat_interleave(torch.arange(batch, device=x.device),
                                    torch.tensor(joint_lengths, device=x.device), output_size=sum(joint_lengths))
    modulation = (mod, owners)
    for index, layer in enumerate(self.layers):
        if index not in skip_layers:
            x = layer(x, positions, modulation)
    x = torch.cat([chunk[:, :image_length] for chunk in x.split(joint_lengths, dim=1)], dim=0)
    x = self.final_norm_video(x.to(self.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
    sample = x.transpose(1, 2).reshape(batch, -1, height, width)
    return module.Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def enable_packed_batch(model):
    """Install before gradient checkpointing and adapter training; idempotent."""
    if hasattr(model, '_sequential_forward'):
        return
    module = sys.modules[type(model).__module__]
    for block in [*model.layers, *model.text_embedder.blocks]:
        if getattr(block, '_moss_checkpointed', False):
            raise ValueError('Enable packed batching before gradient checkpointing')
    for block in model.layers:
        block._sequential_forward = block.forward
        block._packed_situ = getattr(block, '_moss_situ', module._situ)
        block.forward = types.MethodType(_block_forward, block)
    for block in model.text_embedder.blocks:
        block._sequential_forward = block.forward
        block._packed_situ = getattr(block, '_moss_situ', module._situ)
        block.forward = types.MethodType(_text_forward, block)
    model._sequential_forward = model.forward
    model.forward = types.MethodType(_forward, model)
