from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import types

# LyCORIS 4's fused kernels are experimental. The default is its eager torch path.
os.environ.setdefault("LYCORIS_KERNEL_BACKEND", "torch")

import torch
from lycoris import LycorisNetwork
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import Config, Network, validate
from .norms import MossNormModule


def target_modules(model, config: Network):
    main = r"layers\.\d+\.proj\.(q|k|v|o"
    main += r"|ffn_up_gate|ffn_down)" if config.target == "attention-mlp" else ")"
    patterns = [main]
    if config.train_text_projection:
        patterns.append(r"text_embedder\.proj")
        suffix = r"qkv|proj|ffn_gate|ffn_up|ffn_down" if config.target == "attention-mlp" else r"qkv|proj"
        patterns.append(r"text_embedder\.blocks\.\d+\.(" + suffix + ")")
    targets = {
        name: module for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(re.fullmatch(p, name) for p in patterns)
    }
    if config.train_norm:
        for name, module in model.named_modules():
            if type(module).__name__ == "_RMSNorm" and (
                name.startswith("layers.") or name == "final_norm_video"
                or (config.train_text_projection and name.startswith("text_embedder."))
            ):
                targets[name] = module
    return targets


def build_adapter(model, config: Network, device=None):
    validate(dataclasses.replace(Config(), network=config))
    targets = target_modules(model, config)
    if not targets:
        raise ValueError("No MOSS layers matched the requested adapter targets")
    if device is None:
        device = next(model.parameters()).device
    model.requires_grad_(False)
    # Set every preset field, avoiding inherited global presets from other jobs.
    LycorisNetwork.apply_preset({
        "enable_conv": False,
        "target_module": [],
        "target_name": ["^" + re.escape(name) + "$" for name in targets],
        "module_algo_map": {},
        "name_algo_map": {},
        "lora_prefix": "lycoris",
        "use_fnmatch": False,
        "exclude_name": [],
    })
    network = LycorisNetwork(
        model, multiplier=1.0, lora_dim=config.dim, conv_lora_dim=0,
        alpha=config.alpha, conv_alpha=config.alpha,
        network_module=config.algo, use_tucker=config.use_tucker,
        dropout=config.dropout, rank_dropout=config.rank_dropout,
        module_dropout=config.module_dropout, factor=config.factor,
        full_matrix=config.full_matrix, decompose_both=config.decompose_both,
        block_size=config.block_size, constraint=config.constraint,
        weight_decompose=config.dora_wd,
        # Bypass avoids materializing a full delta matrix for the common low-rank paths.
        bypass_mode=(
            config.algo in ("lora", "locon", "loha", "lokr")
            and not config.dora_wd
            and not (config.algo == "lokr" and config.rank_dropout)
        ),
        train_norm=config.train_norm, norm_modules=MossNormModule, warn_on_unmatched=True,
    )
    for layer in network.loras:
        if layer.__class__.__name__ == "LokrModule":
            layer.get_weight = types.MethodType(_lokr_weight, layer)
        if layer.__class__.__name__ in ("LokrModule", "LohaModule"):
            layer.get_diff_weight = types.MethodType(_scaled_diff_weight, layer)
        if layer.__class__.__name__ == "DyLoraModule":
            layer.get_random_rank_weight = types.MethodType(_dylora_rank_weights, layer)
    expected = {"lycoris_" + name.replace(".", "_") for name in targets}
    actual = {layer.lora_name for layer in network.loras}
    if expected != actual:
        raise RuntimeError(f"Adapter coverage mismatch: missing={expected - actual}, unexpected={actual - expected}")
    # FullModule removes the original Linear parameters. MOSS reads weight.dtype
    # in its forward; retain frozen references so both MOSS and org_forward work.
    originals = [(module, module.weight, module.bias) for module in targets.values()] if config.algo == "full" else []
    network.apply_to()
    for module, weight, bias in originals:
        module.register_parameter("weight", weight)
        module.register_parameter("bias", bias)
    network.to(device=device, dtype=torch.float32)
    network.requires_grad_(True)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Frozen base parameters unexpectedly became trainable")
    return network


def save_adapter(network, config: Network, path, *, model_fingerprint="", step=0, epoch=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.safetensors")
    comfy = config.algo in ("lora", "locon", "lokr")
    metadata = {
        "format": "pt", "moss_architecture": "MOSSImageTransformer",
        "moss_network": json.dumps(dataclasses.asdict(config), sort_keys=True),
        "moss_model_fingerprint": model_fingerprint,
        "moss_step": str(step),
        "moss_epoch": str(epoch),
        "moss_adapter_format": "comfyui-lycoris-v1" if comfy else "lycoris-native-v1",
    }
    if comfy:
        state = {}
        for key, value in network.state_dict().items():
            if not key.startswith("lycoris_"):
                raise ValueError(f"Unexpected adapter export key: {key}")
            key = "lora_unet_" + key.removeprefix("lycoris_")
            state[key] = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        save_file(state, str(temporary), metadata)
    else:
        network.save_weights(str(temporary), torch.float32, metadata)
    temporary.replace(path)


def load_adapter(model, path, *, device=None):
    """Attach a saved native adapter to an official MOSS transformer for inference."""
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("moss_adapter_format") not in ("lycoris-native-v1", "comfyui-lycoris-v1"):
        raise ValueError("Expected a MOSS LyCORIS adapter")
    config = Network(**json.loads(metadata["moss_network"]))
    network = build_adapter(model, config, device=device)
    try:
        load_adapter_weights(network, path)
    except Exception:
        network.restore()
        raise
    network.eval()
    return network


def load_adapter_weights(network, path):
    state = load_file(str(path))
    converted = {}
    for key, value in state.items():
        name = "lycoris_" + key.removeprefix("lora_unet_") if key.startswith("lora_unet_") else key
        if name in converted:
            raise ValueError(f"Duplicate adapter key after import: {name}")
        converted[name] = value
    state = converted
    # Upstream DyLoRA exports concatenated ordinary-LoRA matrices, but its
    # trainable state uses ParameterLists and has no inverse load hook.
    for layer in network.loras:
        if layer.__class__.__name__ != "DyLoraModule":
            continue
        prefix = layer.lora_name + "."
        up = state.pop(prefix + "lora_up.weight")
        down = state.pop(prefix + "lora_down.weight")
        start = 0
        for index, parameter in enumerate(layer.up_list):
            width = parameter.shape[1]
            state[prefix + f"up_list.{index}"] = up[:, start:start + width].contiguous()
            start += width
        if start != up.shape[1]:
            raise ValueError("DyLoRA saved up rank does not match configured rank")
        start = 0
        for index, parameter in enumerate(layer.down_list):
            width = parameter.shape[0]
            state[prefix + f"down_list.{index}"] = down[start:start + width].reshape(parameter.shape).contiguous()
            start += width
        if start != down.shape[0]:
            raise ValueError("DyLoRA saved down rank does not match configured rank")
    network.load_state_dict(state, strict=True)


def _dylora_rank_weights(layer):
    # torch checkpoint preserves torch RNG, not Python's random module.
    # Deterministic full rank in eval is also required for reproducible inference.
    block = int(torch.randint(layer.block_count, (1,)).item()) if layer.training else layer.block_count - 1
    return layer.get_weight(block * layer.block_size)


def _lokr_weight(layer, shape):
    # LyCORIS 4.0's rebuild mask is allocated on CPU. Keep it on the weight
    # device so rank dropout works on CUDA and with checkpoint RNG replay.
    from lycoris.functional.lokr import kron_weight

    weight = kron_weight(
        layer.lokr_w1 if layer.use_w1 else None,
        None if layer.use_w1 else layer.lokr_w1_a,
        None if layer.use_w1 else layer.lokr_w1_b,
        layer.lokr_w2 if layer.use_w2 else None,
        None if layer.use_w2 else layer.lokr_w2_a,
        None if layer.use_w2 else layer.lokr_w2_b,
        layer.lokr_t2 if layer.tucker and not layer.use_w2 else None,
        scale=layer.scale,
    )
    if shape is not None:
        weight = weight.reshape(shape)
    if layer.training and layer.rank_dropout:
        keep = (torch.rand(weight.shape[0], device=weight.device) > layer.rank_dropout).to(weight.dtype)
        if layer.rank_dropout_scale:
            keep = keep / keep.mean().clamp_min(1 / keep.numel())
        weight = weight * keep.reshape(-1, *([1] * (weight.ndim - 1)))
    return weight


def _scaled_diff_weight(layer, multiplier=1, shape=None, device=None):
    # get_weight already applies alpha/rank; upstream applies it a second time
    # in get_diff_weight. Apply only the external multiplier and scalar here.
    diff = layer.get_weight(shape)
    diff = diff * layer.scalar.to(diff) * multiplier
    return (diff.to(device) if device is not None else diff), None
