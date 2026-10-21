"""Strict loading of separate, unquantized ComfyUI MOSS component files."""
from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
import torch

from .config import Model


def model_settings(value):
    return value if isinstance(value, Model) else Model(path=str(value))


def _weight_name(key, expected, component):
    candidates = [key]
    prefixes = {
        "transformer": ("model.diffusion_model.", "diffusion_model.", "transformer."),
        "text_encoder": ("text_encoders.qwen35_9b.transformer.", "qwen35_9b.transformer."),
        "vae": ("first_stage_model.", "vae."),
    }
    for prefix in prefixes[component]:
        if key.startswith(prefix):
            candidates.append(key.removeprefix(prefix))
    if component == "text_encoder":
        for name in tuple(candidates):
            if name.startswith(("model.language_model.", "model.visual.")):
                candidates.append(name.removeprefix("model."))
            elif name.startswith("model."):
                candidates.append("language_model." + name.removeprefix("model."))
    return next((name for name in candidates if name in expected), None)


def load_component_weights(model, path, *, component, dtype, device):
    """Validate every name/shape/dtype before allocating checkpoint weights."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise FileNotFoundError(f"Missing {component} .safetensors file: {path}")
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    mapping = {}
    unexpected = []
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        for source in stream.keys():
            name = _weight_name(source, expected, component)
            if name is None and component == "transformer" and source.endswith(".proj.qkv.weight"):
                # Comfy-Org 的单文件把 q/k/v 合并成 qkv（9216 = 3 × 3072），
                # 官方 pipeline 的配置是分开的 q/k/v：按 [q, k, v] 顺序拆开
                prefix = source[: -len("qkv.weight")]
                parts = [prefix + part + ".weight" for part in ("q", "k", "v")]
                entry = stream.get_slice(source)
                shape = tuple(entry.get_shape())
                if all(p in expected for p in parts) and len(shape) == 2 and shape[0] == 3 * shape[1]:
                    chunk = shape[0] // 3
                    for index, key in enumerate(parts):
                        if tuple(expected[key].shape) != (chunk, shape[1]):
                            raise ValueError(f"Wrong fused qkv shape for {key}: {shape}")
                        if key in mapping:
                            raise ValueError(f"Duplicate {component} tensor: {key}")
                        mapping[key] = (source, index)
                    continue
            if name is None:
                # Comfy-Org 的 Qwen3.5 单文件额外带 MTP 头和 lm_head；训练只用主体，
                # 这些权重与官方 text_encoder 配置无关，直接忽略（否则严格校验会拒绝整个文件）
                if component == "text_encoder" and source.startswith(("lm_head.", "mtp.")):
                    continue
                unexpected.append(source)
                continue
            if name in mapping:
                raise ValueError(f"Duplicate {component} tensor after prefix conversion: {name}")
            entry = stream.get_slice(source)
            if tuple(entry.get_shape()) != tuple(expected[name].shape):
                raise ValueError(f"Wrong {component} tensor shape: {source}; expected {tuple(expected[name].shape)}")
            allowed = {"F16", "BF16", "F32"} if expected[name].is_floating_point() else {"I64"}
            if entry.get_dtype() not in allowed:
                raise ValueError(
                    f"Unsupported {component} dtype {entry.get_dtype()} at {source}; "
                    "use unquantized FP32/BF16/FP16 files, not FP8/INT8/NVFP4/GGUF checkpoints"
                )
            mapping[name] = (source, None)
        missing = sorted(set(expected) - mapping.keys())
        if missing or unexpected:
            raise ValueError(
                f"Incompatible {component} checkpoint; missing={missing[:12]}, unexpected={unexpected[:12]}. "
                "Use complete MOSS component files matching model_dir's configuration."
            )
        # Assign into a meta-parameter model, avoiding a second full initialized copy.
        state = {}
        for name, (source, part) in mapping.items():
            tensor = stream.get_tensor(source)
            if part is not None:
                chunk = tensor.shape[0] // 3
                tensor = tensor[part * chunk:(part + 1) * chunk].contiguous()
            state[name] = tensor.to(
                device=device, dtype=dtype if name in parameters else expected[name].dtype,
            )
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Non-persistent rotary buffers were initialized on CPU, not meta.
    return model.requires_grad_(False).to(device)
