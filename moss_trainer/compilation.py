"""Compile complete checkpointed blocks with eager-compatible BF16 boundaries."""
from __future__ import annotations

import functools
import sys
import types

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .batching import PackedPositions, _attention
from .kernels import _bf16, _situ_gradients, _situ_forward, fused_situ
from .modeling import enable_checkpointing


@triton.jit
def _packed_situ_backward(X, D, DX, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = i // C * (2 * C) + i % C
    hidden = tl.load(X + offset, i < N, 0).to(tl.float32)
    gate = tl.load(X + offset + C, i < N, 0).to(tl.float32)
    gradient = tl.load(D + i, i < N, 0).to(tl.float32)
    dh, dg = _situ_gradients(hidden, gate, gradient)
    tl.store(DX + offset, dh, i < N)
    tl.store(DX + offset + C, dg, i < N)


class _PackedSitu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection):
        hidden, gate = projection.chunk(2, -1)
        width, count = hidden.shape[-1], hidden.numel()
        output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
        with torch.cuda.device(projection.device):
            _situ_forward[(triton.cdiv(count, 512),)](
                hidden, gate, output, count, width, 2 * width, 2 * width, 512,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(projection)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        projection, = ctx.saved_tensors
        width, count = projection.shape[-1] // 2, projection.numel() // 2
        result = torch.empty_like(projection)
        with torch.cuda.device(projection.device):
            # Write the complete projection gradient directly: no ChunkBackward cat.
            # One output argument also avoids compile functionalization copies
            # caused by passing two writable, aliased views of the result.
            _packed_situ_backward[(triton.cdiv(count, 512),)](
                projection, gradient.contiguous(), result, count, width, 512,
                enable_fp_fusion=False,
            )
        return result


@triton.jit
def _affine_kernel(X, M, O, Y, N: tl.constexpr, C: tl.constexpr,
                   INDEX: tl.constexpr, OWNERS: tl.constexpr,
                   BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    owner = tl.load(O + i // C, i < N, 0).to(tl.int32) if OWNERS else 0
    scale = tl.load(M + owner * (6 * C) + (INDEX + 1) * C + i % C, i < N, 0).to(tl.float32)
    value = tl.load(X + i, i < N, 0).to(tl.float32)
    result = _bf16(value * _bf16(1.0 + scale))
    if not BACKWARD:
        shift = tl.load(M + owner * (6 * C) + INDEX * C + i % C, i < N, 0).to(tl.float32)
        result = _bf16(result + shift)
    tl.store(Y + i, result, i < N)


@triton.jit
def _residual_kernel(X, D, M, O, Y, N: tl.constexpr, C: tl.constexpr,
                     INDEX: tl.constexpr, OWNERS: tl.constexpr,
                     BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    owner = tl.load(O + i // C, i < N, 0).to(tl.int32) if OWNERS else 0
    gate = tl.load(M + owner * (6 * C) + INDEX * C + i % C, i < N, 0).to(tl.float32)
    delta = tl.load(D + i, i < N, 0).to(tl.float32)
    result = _bf16(delta * gate)
    if not BACKWARD:
        result = _bf16(tl.load(X + i, i < N, 0).to(tl.float32) + result)
    tl.store(Y + i, result, i < N)


class _Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, modulation, owners, index):
        output = torch.empty_like(value)
        ctx.save_for_backward(modulation, owners)
        ctx.index = index
        with torch.cuda.device(value.device):
            _affine_kernel[(triton.cdiv(value.numel(), 512),)](
                value, modulation, owners, output, value.numel(), value.shape[-1],
                index, owners is not None, False, 512, enable_fp_fusion=False,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        modulation, owners = ctx.saved_tensors
        gradient = gradient.contiguous()
        result = torch.empty_like(gradient)
        with torch.cuda.device(gradient.device):
            _affine_kernel[(triton.cdiv(gradient.numel(), 512),)](
                gradient, modulation, owners, result, gradient.numel(), gradient.shape[-1],
                ctx.index, owners is not None, True, 512, enable_fp_fusion=False,
            )
        return result, None, None, None


class _Residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, delta, modulation, owners, index):
        output = torch.empty_like(value)
        ctx.save_for_backward(modulation, owners)
        ctx.index = index
        with torch.cuda.device(value.device):
            _residual_kernel[(triton.cdiv(value.numel(), 512),)](
                value, delta, modulation, owners, output, value.numel(), value.shape[-1],
                index, owners is not None, False, 512, enable_fp_fusion=False,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        modulation, owners = ctx.saved_tensors
        contiguous = gradient.contiguous()
        result = torch.empty_like(contiguous)
        with torch.cuda.device(gradient.device):
            _residual_kernel[(triton.cdiv(gradient.numel(), 512),)](
                contiguous, contiguous, modulation, owners, result,
                gradient.numel(), gradient.shape[-1], ctx.index, owners is not None,
                True, 512, enable_fp_fusion=False,
            )
        return gradient, result, None, None, None


def _modulation_layout(value, modulation, owners):
    return (value.is_cuda and value.dtype == modulation.dtype == torch.bfloat16
            and value.device == modulation.device and value.ndim == 3 and value.shape[0] == 1
            and value.numel() > 0 and value.is_contiguous() and modulation.is_contiguous()
            and modulation.shape[1:] == (6, value.shape[-1]) and not modulation.requires_grad
            and (modulation.shape[0] == 1 if owners is None else
                 owners.device == value.device and owners.is_contiguous()))


def _block_forward(self, value, rope, modulation):
    packed = isinstance(rope, PackedPositions)
    mod, owners = modulation if packed else (modulation, None)

    def condition(index):
        return mod[:, index][owners].unsqueeze(0) if packed else mod[:, index, None]

    def affine(value, index):
        if _modulation_layout(value, mod, owners):
            return _Affine.apply(value, mod, owners, index)
        return value * (1 + condition(index + 1)) + condition(index)

    def residual(value, delta, index):
        if (_modulation_layout(value, mod, owners) and delta.is_contiguous()
                and delta.shape == value.shape and delta.dtype == value.dtype):
            return _Residual.apply(value, delta, mod, owners, index)
        return value + delta * condition(index)

    dtype = self.proj['q'].weight.dtype
    y = affine(self.norm1(value), 0).to(dtype)
    q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    attention = _attention if packed else self._moss_original_attention
    value = residual(value, self.proj['o'](attention(q, k, self.proj['v'](y), rope, self.heads)), 2)
    projection = self.proj['ffn_up_gate'](affine(self.norm2(value), 3).to(dtype))
    if projection.is_cuda and projection.dtype == torch.bfloat16 and projection.is_contiguous():
        activation = _PackedSitu.apply(projection)
    else:
        activation = fused_situ(*projection.chunk(2, -1))
    return residual(value, self.proj['ffn_down'](activation), 5)


def _training_only_compilation(forward, compiled, block, cache_state=None):
    # All main blocks share the same checkpoint code object's Dynamo cache.
    cache_state = cache_state if cache_state is not None else {"exhausted": False}

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        if block.training and torch.is_grad_enabled():
            if not cache_state["exhausted"]:
                try:
                    return compiled(*args, **kwargs)
                except torch._dynamo.exc.FailOnRecompileLimitHit:
                    cache_state["exhausted"] = True
                    print("Compile cache full: reuse cached graphs; new shapes use eager checkpoint execution.",
                          file=sys.stderr, flush=True)
            # Retain guards and existing graphs; only new variants fall back.
            # This remains outside the compiled region and the offload hooks.
            with torch.compiler.set_stance("eager_on_recompile"):
                return compiled(*args, **kwargs)
        return forward(*args, **kwargs)

    return wrapped


def _offload_checkpoint_input(forward, block):
    """Move only saved aliases of this call's input, outside the compiled graph.

    A compiled backward also saves weight tensors. Applying save_on_cpu to the
    entire callable would copy those weights unnecessarily; identify the input
    storage instead. Capture scalar metadata, never the CUDA input itself.
    """
    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        value = args[0] if args else kwargs.get("value", kwargs.get("x"))
        if not (block.training and torch.is_grad_enabled() and isinstance(value, torch.Tensor)
                and value.is_cuda and value.layout == torch.strided and value.numel()):
            return forward(*args, **kwargs)
        storage = (value.device, value.untyped_storage().data_ptr())

        def pack(tensor):
            if (not tensor.is_cuda or tensor.layout != torch.strided
                    or (tensor.device, tensor.untyped_storage().data_ptr()) != storage):
                # A detached handle avoids a saved-tensor/grad_fn reference cycle.
                return tensor.detach()
            host = torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype,
                                       device="cpu", pin_memory=True)
            stream = torch.cuda.current_stream(tensor.device)
            host.copy_(tensor, non_blocking=True)
            tensor.record_stream(stream)
            ready = torch.cuda.Event()
            ready.record(stream)
            return tensor.device, host, ready

        def unpack(packed):
            if isinstance(packed, torch.Tensor):
                return packed
            device, host, ready = packed
            # Backward may execute on a different stream. The wait stays on GPU;
            # empty_strided also preserves layouts expected by the compiled graph.
            torch.cuda.current_stream(device).wait_event(ready)
            restored = torch.empty_strided(host.size(), host.stride(), dtype=host.dtype, device=device)
            restored.copy_(host, non_blocking=True)
            return restored

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return forward(*args, **kwargs)

    return wrapped


def enable_compiled_checkpointing(model, *, offload=False):
    """Install after adapters/packed execution, before any checkpoint wrapping.

    Only the main blocks are compiled. Native RMS reduction custom ops, explicit
    BF16 casts and separate residual gradient edges preserve the eager calculation.
    """
    if getattr(model, '_moss_compiled', False):
        if getattr(model, '_moss_compiled_offload', False) != offload:
            raise ValueError("Compiled checkpoints are already enabled with different offload settings")
        return
    if hasattr(model, '_moss_checkpoint_config'):
        raise ValueError("Install compiled checkpoints before ordinary checkpoint wrapping")
    if (model.config.dim, model.config.ffn_dim, model.config.num_heads, model.config.text_dim) != (3072, 8192, 24, 4096):
        raise ValueError("Compiled checkpoints are validated for the official MOSS dimensions only")
    if any(getattr(layer, '_moss_situ', None) is not fused_situ for layer in model.layers):
        raise ValueError("Compiled checkpoints require fused SITU")
    if any(not getattr(layer.norm1, '_moss_fused_rmsnorm', False) for layer in model.layers):
        raise ValueError("Compiled checkpoints require fused RMSNorm")
    # LyCORIS lazily mutates its backend registry on first use; do this outside
    # Dynamo's checkpoint subgraph, where such side effects are forbidden.
    from lycoris.kernels import available_backends, resolve_backend
    available_backends()
    if resolve_backend() != "torch":
        raise ValueError("Compiled checkpoints require LYCORIS_KERNEL_BACKEND=torch")
    module = sys.modules[type(model).__module__]
    for layer in model.layers:
        layer._moss_original_attention = module._attention
        layer.forward = types.MethodType(_block_forward, layer)
    enable_checkpointing(model)
    cache_state = {"exhausted": False}
    for layer in model.layers:
        forward = layer.forward
        compiled = torch.compile(
            forward, fullgraph=True, dynamic=True,
            # Limit GEMMs to ATen and disable size padding. Small-factor layouts
            # can still change rounding; config validation restricts tested cases.
            options={"emulate_precision_casts": True, "shape_padding": False,
                     "comprehensive_padding": False, "max_autotune_gemm_backends": "ATEN"},
        )
        layer.forward = _training_only_compilation(forward, compiled, layer, cache_state)
    if offload:
        # Keep hooks outside compile(checkpoint(...)). The uncompiled text blocks
        # also use input-only offload when text adapters are being trained.
        for block in [*model.layers, *model.text_embedder.blocks]:
            block.forward = _offload_checkpoint_input(block.forward, block)
            block._moss_checkpoint_policy = ("full", True)
        model._moss_checkpoint_config = ("full", True, len(model.layers))
    model._moss_compiled_offload = offload
    model._moss_compiled = True
