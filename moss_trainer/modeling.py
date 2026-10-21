from __future__ import annotations

import functools
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

from .weights import load_component_weights, model_settings


def pipeline_module(model_path: str | Path):
    path = Path(model_settings(model_path).path) / "pipeline_moss_image.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing local MOSS pipeline: {path}")
    name = "_moss_pipeline_" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def transformer_structure(model_path):
    model_path = model_settings(model_path).path
    cls = pipeline_module(model_path).MOSSImageTransformer
    config = json.loads((Path(model_path) / "transformer/config.json").read_text())
    with torch.device("meta"):
        return cls.from_config(config)


def load_transformer(model_path, device="cuda"):
    settings = model_settings(model_path)
    cls = pipeline_module(settings).MOSSImageTransformer
    if settings.format == "comfyui":
        from accelerate import init_empty_weights
        config = json.loads((Path(settings.path) / "transformer/config.json").read_text())
        with init_empty_weights(include_buffers=False):
            model = cls.from_config(config)
        return load_component_weights(model, settings.transformer, component="transformer",
                                      dtype=torch.bfloat16, device=device)
    model = cls.from_pretrained(
        Path(settings.path) / "transformer", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    # The official RoPE module explicitly reconstructs its fp32 frequency buffer.
    return model.to(device)


def load_text_encoder(model_path, device="cuda"):
    from transformers import AutoConfig, AutoModel
    settings = model_settings(model_path)
    path = Path(settings.path) / "text_encoder"
    if settings.format == "hf":
        return AutoModel.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        ).requires_grad_(False).eval().to(device)
    from accelerate import init_empty_weights
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModel.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    return load_component_weights(model, settings.text_encoder, component="text_encoder",
                                  dtype=torch.bfloat16, device=device).eval()


def load_vae(model_path, device="cuda"):
    from diffusers import AutoencoderKLFlux2
    settings = model_settings(model_path)
    path = Path(settings.path) / "vae"
    if settings.format == "hf":
        return AutoencoderKLFlux2.from_pretrained(
            path, torch_dtype=torch.float32, local_files_only=True,
        ).requires_grad_(False).eval().to(device)
    from accelerate import init_empty_weights
    config = json.loads((path / "config.json").read_text())
    with init_empty_weights(include_buffers=False):
        model = AutoencoderKLFlux2.from_config(config)
    return load_component_weights(model, settings.vae, component="vae",
                                  dtype=torch.float32, device=device).eval()


_SAVED_ATTENTION_OPS = frozenset({
    "aten._scaled_dot_product_flash_attention.default",
    "aten._flash_attention_forward.default",
    "torch_attn._varlen_attn.default",
})


def _attention_checkpoint_policy(ctx, op, *args, **kwargs):
    # Include the public varlen custom op: its internal aten calls are opaque
    # to the selective-checkpoint dispatch mode in packed batches.
    return CheckpointPolicy.MUST_SAVE if str(op) in _SAVED_ATTENTION_OPS else CheckpointPolicy.PREFER_RECOMPUTE


def enable_checkpointing(model, policy="full", *, offload=False, blocks=-1):
    if policy not in ("full", "attention"):
        raise ValueError("Checkpoint policy must be full or attention")
    count = len(model.layers) if blocks == -1 else blocks
    if not 1 <= count <= len(model.layers):
        raise ValueError("checkpoint_blocks exceeds this model's depth")
    settings = (policy, offload, count)
    if hasattr(model, '_moss_checkpoint_config') and model._moss_checkpoint_config != settings:
        raise ValueError("Gradient checkpointing is already enabled with different settings")
    # Preserve original state-dict/module paths for LyCORIS and saved adapters.
    for block in [*model.layers[:count], *model.text_embedder.blocks]:
        if getattr(block, "_moss_checkpointed", False):
            if block._moss_checkpoint_policy != (policy, offload):
                raise ValueError("Gradient checkpointing is already enabled with different settings")
            continue
        original = block.forward

        @functools.wraps(original)
        def wrapped(*args, _forward=original, _block=block, **kwargs):
            if _block.training and torch.is_grad_enabled():
                options = {}
                if policy == "attention":
                    options["context_fn"] = functools.partial(
                        create_selective_checkpoint_contexts, _attention_checkpoint_policy,
                    )
                # The checkpoint's own inner hooks still control recomputation;
                # this outer hook offloads its saved inputs, not model weights.
                context = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
                with context:
                    return checkpoint(_forward, *args, use_reentrant=False, **options, **kwargs)
            return _forward(*args, **kwargs)

        block.forward = wrapped
        block._moss_checkpointed = True
        block._moss_checkpoint_policy = (policy, offload)
    model._moss_checkpoint_config = settings


def pack_latents(latents, vae):
    """Inverse of the official pipeline's decode transform (no SD/SDXL scale)."""
    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("VAE latent height and width must be even")
    packed = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)
    mean = vae.bn.running_mean.float().view(1, -1, 1, 1)
    scale = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1)
    return (packed.float() - mean) / scale


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required for caching/training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires BF16 GPU support")
