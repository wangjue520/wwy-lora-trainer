"""Epoch previews using cached prompts, fixed noise, and restored training RNG."""
from __future__ import annotations

import gc
from pathlib import Path
import random

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from .data import digest
from .modeling import load_vae, pipeline_module


def preview_sigmas(config, device):
    # Both schedules use 50 denoising intervals.
    if config.sample.scheduler == "moss":
        scheduler = pipeline_module(config.model).MOSSImageScheduler()
        scheduler.set_timesteps(50, device=device)
        return scheduler.sigmas.to(device=device, dtype=torch.float32)
    return torch.linspace(1, 0, 51, device=device, dtype=torch.float32)


@torch.no_grad()
def generate_preview(model, network, config, text_directory, epoch):
    device = next(model.parameters()).device
    model_mode, network_mode = model.training, network.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    vae = None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            network.eval()
            features = []
            for prompt in (config.sample.positive, config.sample.negative):
                path = Path(text_directory) / f"text-{digest(prompt)}.safetensors"
                context = load_file(str(path))["embedding"].unsqueeze(0).to(device)
                features.append((context, torch.ones(context.shape[:2], dtype=torch.bool, device=device)))
            width, height = config.sample.resolution
            generator = torch.Generator(device=device).manual_seed(config.training.seed)
            latent = torch.randn((1, 128, height // 16, width // 16), device=device,
                                 dtype=torch.float32, generator=generator)
            sigmas = preview_sigmas(config, device)

            def velocity(value, sigma):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    positive = model(value, sigma * 1000, *features[0]).sample.float()
                    if config.sample.cfg == 1:
                        return positive
                    negative = model(value, sigma * 1000, *features[1]).sample.float()
                    return negative + config.sample.cfg * (positive - negative)

            for index in range(len(sigmas) - 1):
                sigma, next_sigma = sigmas[index], sigmas[index + 1]
                delta = next_sigma - sigma
                first = velocity(latent, sigma)
                proposal = latent + delta * first
                if config.sample.sampler == "heun" and index < len(sigmas) - 2:
                    second = velocity(proposal, next_sigma)
                    latent = latent + delta * (first + second) * 0.5
                else:
                    latent = proposal
            if not torch.isfinite(latent).all():
                raise ValueError("Preview produced non-finite latents")
            # Load only the VAE for decoding; Qwen features were cached beforehand.
            vae = load_vae(config.model, device=device)
            vae.enable_tiling()
            value = latent.to(vae.dtype)
            mean = vae.bn.running_mean.to(value).view(1, -1, 1, 1)
            scale = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(value).view(1, -1, 1, 1)
            value = value * scale + mean
            batch, channels, h, w = value.shape
            value = value.reshape(batch, channels // 4, 2, 2, h, w)
            value = value.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, 2 * h, 2 * w)
            decoded = vae.decode(value, return_dict=False)[0]
            if not torch.isfinite(decoded).all():
                raise ValueError("Preview VAE produced non-finite pixels")
            pixels = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            destination = Path(config.training.output_dir) / "samples" / f"epoch-{epoch:06d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".partial.png")
            Image.fromarray(pixels).save(temporary)
            temporary.replace(destination)
            return destination
    finally:
        model.train(model_mode)
        network.train(network_mode)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        del vae
        gc.collect()
