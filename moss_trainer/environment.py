"""Deployment readiness checks; small tensors only, never model training."""
from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import pkgutil
import platform
import shutil
import subprocess
import sys
import tempfile


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _tensorboard_check():
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    with tempfile.TemporaryDirectory(prefix="moss-doctor-") as directory:
        with SummaryWriter(directory) as writer:
            writer.add_scalar("environment/check", 1.0, 0)
            writer.add_image("environment/image", np.zeros((2, 2, 3), dtype=np.uint8), 0, dataformats="HWC")
        _require(any(Path(directory).glob("events.out.tfevents.*")), "TensorBoard did not create an event file")


def _attention_check():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.varlen import varlen_attn
    q = torch.randn(1, 2, 12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(q, q, q).float().square().mean().backward()
    _require(torch.isfinite(q.grad).all().item(), "FlashAttention gradient is not finite")
    value = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundaries = torch.tensor([0, 5, 12], dtype=torch.int32, device="cuda")
    varlen_attn(value, value, value, boundaries, boundaries, 7, 7).float().square().mean().backward()
    _require(torch.isfinite(value.grad).all().item(), "Packed attention gradient is not finite")


def _compile_check():
    import torch
    from torch.utils.checkpoint import checkpoint
    from .kernels import fused_rmsnorm, fused_situ

    def block(value, weight, norm):
        normalized = fused_rmsnorm(value, norm)
        return fused_situ(normalized @ weight, normalized)

    def forward(value, weight, norm):
        return checkpoint(block, value, weight, norm, use_reentrant=False)

    value = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(forward, fullgraph=True, dynamic=True, options={
        "emulate_precision_casts": True, "shape_padding": False,
        "comprehensive_padding": False, "max_autotune_gemm_backends": "ATEN",
    })
    # Also exercise checkpoint activation transfer without updating any parameter.
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        loss = compiled(value, weight, norm).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    _require(torch.isfinite(loss).item(), "Compiled fused forward is not finite")
    _require(all(x.grad is not None and torch.isfinite(x.grad).all().item() for x in (value, weight, norm)),
             "Compiled fused backward or CPU checkpoint offload failed")


def environment_report():
    checks = {}

    def check(name, function):
        try:
            function()
            checks[name] = {"ok": True}
        except Exception as error:
            checks[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "libc": list(platform.libc_ver()), "checks": checks}
    check("platform", lambda: _require(sys.platform == "linux" and platform.machine() == "x86_64"
                                        and sys.version_info[:2] == (3, 12), "Requires Linux x86_64 and Python 3.12"))
    check("host_compilers", lambda: _require(shutil.which("cc") and shutil.which("c++"),
                                             "C/C++ compilers missing; on Ubuntu/Debian install build-essential"))
    versions = {}
    for name in ("moss-trainer", "torch", "diffusers", "transformers", "accelerate", "lycoris-lora",
                 "safetensors", "triton", "tensorboard", "numpy", "pillow", "einops", "tqdm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    result["versions"] = versions

    def imports():
        import moss_trainer
        from diffusers import AutoencoderKLFlux2
        from transformers import AutoModel, AutoTokenizer, Qwen3_5Model
        for module in pkgutil.iter_modules(moss_trainer.__path__):
            importlib.import_module(f"moss_trainer.{module.name}")

    check("runtime_imports", imports)
    check("tensorboard_events", _tensorboard_check)
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
        check("cuda_bf16", lambda: _require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                                             "NVIDIA CUDA GPU with BF16 support is required; check the host driver"))
        if checks["cuda_bf16"]["ok"]:
            result.update(gpu=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
                          gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
            check("flash_and_packed_attention", _attention_check)
            check("triton_compile_backward_cpu_offload", _compile_check)
    except Exception as error:
        checks["cuda_runtime"] = {"ok": False, "error": str(error)}
    if shutil.which("nvidia-smi"):
        try:
            result["driver_version"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=5,
            ).strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    result["ready"] = all(item["ok"] for item in checks.values())
    result["scope"] = "Dependency and small-tensor GPU checks only; model files/data/training are not executed"
    return result
