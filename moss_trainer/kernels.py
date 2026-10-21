"""Pointwise fusion preserving eager MOSS BF16 rounding and FP32 reductions.

SITU and the pointwise parts of RMSNorm are fused; GEMMs, attention and model
weights are unchanged. Recompute intermediates instead of saving them all.
Unsupported layouts/dtypes use the original expression.
"""
from __future__ import annotations

import functools
import types

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _bf16(x):
    # These casts are intentional: ordinary fusion would change BF16 rounding.
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tanh(x):
    # Triton's libdevice lowering flushes FP32 subnormals; torch preserves them.
    return tl.where(tl.abs(x) < 1.1754943508222875e-38, x, libdevice.tanh(x))


@triton.jit
def _intermediates(h, g):
    ht = _bf16(_tanh(_bf16(h * 0.04)))
    gt = _bf16(_tanh(_bf16(g * 0.25)))
    sig = _bf16(1.0 / (1.0 + libdevice.exp(-g)))
    ga = _bf16(4.0 * gt)
    return ht, gt, sig, ga, _bf16(ga * sig), _bf16(25.0 * ht)


@triton.jit
def _situ_forward(H, G, Y, N: tl.constexpr, C: tl.constexpr,
                  HS: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = tl.load(H + i // C * HS + i % C, i < N, 0).to(tl.float32)
    g = tl.load(G + i // C * GS + i % C, i < N, 0).to(tl.float32)
    ht, gt, sig, ga, b, c = _intermediates(h, g)
    tl.store(Y + i, _bf16(b * c), i < N)


@triton.jit
def _situ_gradients(h, g, d):
    ht, gt, sig, ga, b, c = _intermediates(h, g)
    db, dc = _bf16(d * c), _bf16(d * b)
    # PyTorch's BF16 tanh/sigmoid backward also rounds scalar_t operations.
    dht = _bf16(dc * 25.0)
    dh = _bf16(_bf16(dht * _bf16(1.0 - _bf16(ht * ht))) * 0.04)
    dga, ds = _bf16(db * sig), _bf16(db * ga)
    dgt = _bf16(dga * 4.0)
    dg1 = _bf16(_bf16(dgt * _bf16(1.0 - _bf16(gt * gt))) * 0.25)
    dg2 = _bf16(_bf16(ds * _bf16(1.0 - sig)) * sig)
    return dh, _bf16(dg1 + dg2)


@triton.jit
def _situ_backward(H, G, D, DH, DG, N: tl.constexpr, C: tl.constexpr,
                   HS: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = tl.load(H + i // C * HS + i % C, i < N, 0).to(tl.float32)
    g = tl.load(G + i // C * GS + i % C, i < N, 0).to(tl.float32)
    d = tl.load(D + i, i < N, 0).to(tl.float32)
    dh, dg = _situ_gradients(h, g, d)
    tl.store(DH + i, dh, i < N)
    tl.store(DG + i, dg, i < N)


class _Situ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, gate):
        ctx.save_for_backward(hidden, gate)
        output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
        with torch.cuda.device(hidden.device):
            _situ_forward[(triton.cdiv(hidden.numel(), 512),)](
                hidden, gate, output, hidden.numel(), hidden.shape[-1],
                hidden.stride(-2), gate.stride(-2), 512, enable_fp_fusion=False,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        hidden, gate = ctx.saved_tensors
        dh = torch.empty_like(hidden, memory_format=torch.contiguous_format)
        dg = torch.empty_like(dh)
        with torch.cuda.device(hidden.device):
            _situ_backward[(triton.cdiv(hidden.numel(), 512),)](
                hidden, gate, gradient.contiguous(), dh, dg,
                hidden.numel(), hidden.shape[-1], hidden.stride(-2), gate.stride(-2),
                512, enable_fp_fusion=False,
            )
        return dh, dg


def _row_layout(value):
    if value.ndim < 2 or value.stride(-1) != 1:
        return False
    # Leading dimensions must collapse into rows without a copy. A chunk of
    # the joint FFN projection may have gaps between rows, which is supported.
    return all(value.shape[i] == 1 or value.stride(i) == value.stride(i + 1) * value.shape[i + 1]
               for i in range(value.ndim - 2))


def fused_situ(hidden, gate):
    if (hidden.is_cuda and hidden.dtype == gate.dtype == torch.bfloat16
            and hidden.device == gate.device and hidden.shape == gate.shape
            and hidden.numel() and _row_layout(hidden) and _row_layout(gate)):
        return _Situ.apply(hidden, gate)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def enable_fused_situ(model):
    """Install on this model only, before checkpointing; preserve parameter keys."""
    blocks = [*model.layers, *model.text_embedder.blocks]
    if any(getattr(block, '_moss_checkpointed', False) for block in blocks):
        raise ValueError('Enable fused SITU before gradient checkpointing')
    replacements = []
    for block in blocks:
        if getattr(block, '_moss_situ', None) is fused_situ:
            continue
        name = '_sequential_forward' if hasattr(block, '_sequential_forward') else 'forward'
        original = getattr(block, name).__func__
        if '_situ' not in original.__code__.co_names:
            raise ValueError('Expected the official MOSS block SITU expression')
        # Clone the function's globals, so another model in the same process
        # continues to use its original activation and official files stay intact.
        replacement = types.FunctionType(
            original.__code__, {**original.__globals__, '_situ': fused_situ},
            original.__name__, original.__defaults__, original.__closure__,
        )
        replacement.__kwdefaults__ = original.__kwdefaults__
        functools.update_wrapper(replacement, original)
        replacements.append((block, name, types.MethodType(replacement, block)))
    for block, name, replacement in replacements:
        setattr(block, name, replacement)
        block._moss_situ = fused_situ
        if hasattr(block, '_packed_situ'):
            block._packed_situ = fused_situ


@triton.jit
def _rms_square(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, x * x, i < N)


@torch.library.custom_op("moss_trainer::rms_statistics", mutates_args=())
def _rms_statistics(square: torch.Tensor) -> torch.Tensor:
    # An opaque boundary keeps Inductor from changing the native reduction tree.
    return torch.rsqrt(square.mean(-1, keepdim=True) + 1e-6)


@_rms_statistics.register_fake
def _rms_statistics_fake(square):
    return square.new_empty((*square.shape[:-1], 1))


@torch.library.custom_op("moss_trainer::rms_reduction", mutates_args=())
def _rms_reduction(product: torch.Tensor, inv_rms: torch.Tensor) -> torch.Tensor:
    return (-0.5 * product.sum(-1, keepdim=True) * inv_rms.pow(3)) / product.shape[-1]


@_rms_reduction.register_fake
def _rms_reduction_fake(product, inv_rms):
    return torch.empty_like(inv_rms)


@torch.library.custom_op("moss_trainer::rms_weight_reduction", mutates_args=())
def _rms_weight_reduction(product: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(product.ndim - 1))
    return product.sum(axes) if axes else product.clone()


@_rms_weight_reduction.register_fake
def _rms_weight_reduction_fake(product):
    return product.new_empty((product.shape[-1],))


@triton.jit
def _rms_weight_product(X, R, D, P, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    inv_rms = tl.load(R + i // C, i < N, 0)
    grad = tl.load(D + i, i < N, 0).to(tl.float32)
    tl.store(P + i, _bf16(grad * _bf16(x * inv_rms)), i < N)


@triton.jit
def _rms_forward(X, W, R, Y, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    weight = tl.load(W + i % C).to(tl.float32)
    inv_rms = tl.load(R + i // C, i < N, 0)
    tl.store(Y + i, _bf16(_bf16(x * inv_rms) * weight), i < N)


@triton.jit
def _rms_product(X, W, D, P, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    grad = tl.load(D + i, i < N, 0).to(tl.float32)
    weight = tl.load(W + i % C).to(tl.float32)
    tl.store(P + i, _bf16(grad * weight) * x, i < N)


@triton.jit
def _rms_backward(X, W, R, S, D, DA, DB, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    grad = tl.load(D + i, i < N, 0).to(tl.float32)
    weight = tl.load(W + i % C).to(tl.float32)
    inv_rms = tl.load(R + i // C, i < N, 0)
    grad_square = tl.load(S + i // C, i < N, 0)
    # Keep the two cast-backward contributions separate. Combining them here
    # changes BF16 accumulation order where x also feeds a residual connection.
    tl.store(DA + i, _bf16(_bf16(grad * weight) * inv_rms), i < N)
    tl.store(DB + i, _bf16(grad_square * (2.0 * x)), i < N)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, statistics_value, weight):
        grid = (triton.cdiv(value.numel(), 512),)
        with torch.cuda.device(value.device):
            square = torch.empty_like(statistics_value, dtype=torch.float32)
            _rms_square[grid](statistics_value, square, value.numel(), 512, enable_fp_fusion=False)
            # Keep native reductions and their order, rather than substituting
            # a different Triton reduction tree for mean/sum.
            inv_rms = (_rms_statistics(square) if torch.compiler.is_compiling()
                       else torch.rsqrt(square.mean(-1, keepdim=True) + 1e-6))
            output = torch.empty_like(value)
            _rms_forward[grid](value, weight, inv_rms, output, value.numel(),
                               value.shape[-1], 512, enable_fp_fusion=False)
        ctx.save_for_backward(value, statistics_value, weight, inv_rms)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        value, statistics_value, weight, inv_rms = ctx.saved_tensors
        gradient = gradient.contiguous()
        grid = (triton.cdiv(value.numel(), 512),)
        with torch.cuda.device(value.device):
            product = torch.empty_like(value, dtype=torch.float32)
            _rms_product[grid](value, weight, gradient, product, value.numel(),
                               value.shape[-1], 512, enable_fp_fusion=False)
            grad_square = (_rms_reduction(product, inv_rms) if torch.compiler.is_compiling()
                           else (-0.5 * product.sum(-1, keepdim=True) * inv_rms.pow(3)) / value.shape[-1])
            direct = torch.empty_like(value)
            statistics = torch.empty_like(statistics_value)
            _rms_backward[grid](statistics_value, weight, inv_rms, grad_square, gradient,
                                direct, statistics, value.numel(), value.shape[-1], 512,
                                enable_fp_fusion=False)
            weight_gradient = None
            if ctx.needs_input_grad[2]:
                weight_product = torch.empty_like(value)
                _rms_weight_product[grid](value, inv_rms, gradient, weight_product,
                                          value.numel(), value.shape[-1], 512, enable_fp_fusion=False)
                axes = tuple(range(value.ndim - 1))
                weight_gradient = (_rms_weight_reduction(weight_product) if torch.compiler.is_compiling()
                                   else weight_product.sum(axes) if axes else weight_product)
        return direct, statistics, weight_gradient


def fused_rmsnorm(value, weight):
    if (value.is_cuda and value.dtype == weight.dtype == torch.bfloat16
            and value.device == weight.device and value.ndim >= 1 and value.numel()
            and weight.ndim == 1 and weight.numel() == value.shape[-1]
            and value.is_contiguous() and weight.is_contiguous()):
        # Two zero-copy views retain the two independent autograd edges of the
        # official expression's separate x.float() calls without storing FP32 x.
        return _RMSNorm.apply(value.view_as(value), value.view_as(value), weight)
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    return normalized.to(value.dtype) * weight


def _rmsnorm_forward(self, value):
    return fused_rmsnorm(value, self.weight)


def enable_fused_rmsnorm(model):
    """Fuse official fixed-epsilon RMSNorm, including optional affine gradients."""
    if any(getattr(block, '_moss_checkpointed', False)
           for block in [*model.layers, *model.text_embedder.blocks]):
        raise ValueError('Enable fused RMSNorm before gradient checkpointing')
    for norm in model.modules():
        if type(norm).__name__ == '_RMSNorm':
            norm.forward = types.MethodType(_rmsnorm_forward, norm)
            norm._moss_fused_rmsnorm = True
