"""Cache frozen, sample-independent text refinement before the training loop."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

from .data import _valid_tensor_file, _write_tensors, digest


def _refined_forward(self, context, rope):
    if context.ndim != 3 or context.shape[-1] != self._moss_refined_width:
        raise ValueError("This training model requires cached refined text features")
    return context


@torch.no_grad()
def prepare_refined_text(model, config, spec, directory):
    """Return the text cache directory; preserve the original Qwen feature files.

    Packed multi-sample GEMMs can round differently from per-caption execution,
    so that path retains live refinement. Trainable text projections also retain
    live refinement. This does not cache text states from the joint main blocks.
    """
    directory = Path(directory)
    if (config.network.train_text_projection
            or (config.training.batch_mode == "packed" and config.dataset.batch_size > 1)):
        return directory
    refiner = model.text_embedder
    if any(parameter.requires_grad for parameter in refiner.parameters()):
        raise ValueError("Text refinement caching requires frozen text parameters")
    device = next(refiner.parameters()).device
    if device.type != "cuda":
        raise ValueError("Prepare refined text before moving its weights off CUDA")
    signature = digest({
        "kind": "moss-frozen-sequential-text-v1", "source": spec["fingerprint"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "capability": torch.cuda.get_device_capability(device),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fused_situ": config.training.fused_situ,
        "fused_rmsnorm": config.training.fused_rmsnorm,
    })
    destination = directory / f"refined-{signature}"
    destination.mkdir(parents=True, exist_ok=True)
    keys = sorted({digest(record["caption"]) for record in spec["records"]} | {digest("")})
    if config.sample.every_epochs:
        keys = sorted(set(keys) | {digest(config.sample.positive), digest(config.sample.negative)})
    pending = []
    for key in keys:
        source = directory / f"text-{key}.safetensors"
        with safe_open(str(source), framework="pt", device="cpu") as stream:
            length, width = stream.get_slice("embedding").get_shape()
        if width != model.config.text_dim:
            raise ValueError("Unexpected source text embedding width")
        target = destination / source.name
        if not _valid_tensor_file(target, signature, "embedding", (length, model.config.dim)):
            pending.append((source, target, length))
    for source, target, length in tqdm(pending, desc="Cache frozen text refinement"):
        context = load_file(str(source))["embedding"].unsqueeze(0).to(device)
        # Official text coordinates depend only on text length; the dummy image
        # token is discarded. No image, timestep, noise or trainable adapter enters.
        rope = model.rope(1, 1, length, device).to(torch.bfloat16)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            refined = refiner(context, rope)[0]
        if refined.dtype != torch.bfloat16 or not torch.isfinite(refined).all():
            raise ValueError("Invalid frozen text refinement output")
        _write_tensors(target, {"embedding": refined}, {"fingerprint": signature})
        del context, rope, refined
    # Preserve module/parameter names. These frozen weights remain on CPU and
    # are never transferred or evaluated during a training step.
    refiner.to("cpu")
    refiner._moss_refined_width = model.config.dim
    refiner.forward = types.MethodType(_refined_forward, refiner)
    model._moss_refined_text = True
    gc.collect()
    torch.cuda.empty_cache()
    return destination
