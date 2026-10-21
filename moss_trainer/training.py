from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .adapters import build_adapter, target_modules
from .buckets import BucketBatchSampler
from .checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint, validate_output_directory
from .config import Config
from .data import CachedDataset, collate_cached, digest, verify_cache
from .modeling import enable_checkpointing, load_transformer, require_cuda, transformer_structure
from .scheduling import build_optimizer, build_scheduler, training_plan
from .telemetry import TensorBoardLogger


def sample_flow_batch(latents, config, generator=None):
    """Rectified-flow velocity target: z_t=(1-sigma)*z+sigma*eps, v=eps-z."""
    batch = latents.shape[0]
    device = latents.device
    if config.timestep_sampling == "uniform":
        sigma = torch.rand(batch, device=device, dtype=torch.float32, generator=generator)
    else:
        normal = torch.randn(batch, device=device, dtype=torch.float32, generator=generator)
        sigma = torch.sigmoid(normal * config.logit_std + config.logit_mean)
    sigma = config.timestep_shift * sigma / (1 + (config.timestep_shift - 1) * sigma)
    # Match the official model's 1000-point training time grid.
    timestep = (sigma * 1000).round().clamp(1, 1000)
    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    noise = torch.randn(latents.shape, device=device, dtype=torch.float32, generator=generator)
    clean = latents.float()
    return (1 - sigma) * clean + sigma * noise, timestep, noise - clean


def apply_caption_dropout(context, mask, empty, probability):
    if probability == 0:
        return context, mask
    selected = torch.rand(context.shape[0]) < probability
    if not selected.any():
        return context, mask
    length = max(context.shape[1], empty.shape[0])
    padded = context.new_zeros(context.shape[0], length, context.shape[2])
    padded[:, :context.shape[1]] = context
    output_mask = mask.new_zeros(mask.shape[0], length)
    output_mask[:, :mask.shape[1]] = mask
    for index in selected.nonzero().flatten().tolist():
        padded[index].zero_()
        padded[index, :empty.shape[0]] = empty
        output_mask[index].zero_()
        output_mask[index, :empty.shape[0]] = True
    return padded, output_mask


def train(config: Config, resume=None, *, new_stage=False):
    """Explicit training entry point. Never called by doctor, inspect or bucket planning."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This initial trainer supports one GPU; launch with uv run, not distributed launch")
    if new_stage and not resume:
        raise ValueError("--new-stage requires --resume")
    if new_stage:
        output = Path(config.training.output_dir)
        if output.resolve() == Path(resume).resolve().parent:
            raise ValueError("A new stage requires a separate empty output directory")
        if output.exists() and any(output.iterdir()):
            raise ValueError("A new stage requires an empty output directory")
    require_cuda()
    spec, cache_directory = verify_cache(config)
    records = spec["records"]
    sampler = BucketBatchSampler(records, config.dataset.batch_size, config.dataset.num_repeats, config.training.seed)
    batch_count = sampler.total_batches
    plan = training_plan(config.training, batch_count)
    state = load_checkpoint(resume, config, spec, batch_count=batch_count, new_stage=new_stage) if resume else None
    step, epoch, start_batch = (state["step"], state["epoch"], state["next_batch"]) if state and not new_stage else (0, 0, 0)
    output = Path(config.training.output_dir)
    validate_output_directory(output, resume, step, plan.max_steps, epoch=epoch, max_epochs=plan.max_epochs)
    # Check an unavoidable lower bound before loading a full-sized model.
    if config.network.algo == "full":
        structure = transformer_structure(config.model)
        target_parameters = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                                for m in target_modules(structure, config.network).values())
        minimum_bytes = sum(p.numel() * 2 for p in structure.parameters()) + target_parameters * 16
        free_bytes, _ = torch.cuda.mem_get_info()
        del structure
        if minimum_bytes > free_bytes:
            raise RuntimeError("Full fine-tuning exceeds available VRAM even before activations; choose a low-rank algorithm")
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    model = load_transformer(config.model)
    if config.training.fused_situ:
        from .kernels import enable_fused_situ
        enable_fused_situ(model)
    if config.training.fused_rmsnorm:
        from .kernels import enable_fused_rmsnorm
        enable_fused_rmsnorm(model)
    if config.training.batch_mode == "packed":
        from .batching import enable_packed_batch
        enable_packed_batch(model)
    network = build_adapter(model, config.network, device="cuda")
    from .refinement import prepare_refined_text
    text_directory = prepare_refined_text(model, config, spec, cache_directory)
    model.train()
    network.train()
    if config.training.compile_blocks:
        from .compilation import enable_compiled_checkpointing
        enable_compiled_checkpointing(model, offload=config.training.checkpoint_offload)
    elif config.training.gradient_checkpointing:
        enable_checkpointing(model, config.training.checkpoint_policy,
                             offload=config.training.checkpoint_offload,
                             blocks=config.training.checkpoint_blocks)
    parameters = [p for p in network.parameters() if p.requires_grad]
    optimizer = build_optimizer(parameters, config.training)
    scheduler = build_scheduler(optimizer, config.training, plan)
    dataset = CachedDataset(records, cache_directory, text_directory=text_directory)
    # A private DataLoader RNG leaves model/dropout RNG unchanged when resuming.
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_cached,
        num_workers=config.training.num_workers, pin_memory=True,
        generator=loader_generator,
    )
    empty = load_file(str(text_directory / f"text-{digest('')}.safetensors"))["embedding"]
    if state is not None:
        restore_checkpoint(resume, state, network, optimizer, scheduler, new_stage=new_stage)
        if new_stage:
            print(json.dumps({"new_stage": True, "source_step": state["step"],
                              "source_epoch": state["epoch"], "stage_max_epochs": plan.max_epochs,
                              "stage_max_steps": plan.max_steps, "initial_lr": optimizer.param_groups[0]["lr"],
                              "warmup_steps": plan.warmup_steps}), flush=True)
        del state
    output.mkdir(parents=True, exist_ok=True)
    accumulation = config.training.gradient_accumulation_steps
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with TensorBoardLogger(output / "tensorboard", enabled=config.training.tensorboard, start_step=step) as telemetry:
        update_started = time.monotonic()
        last_checkpoint = None
        while step < plan.max_steps:
            sampler.set_epoch(epoch, start_batch)
            loader_generator.manual_seed(config.training.seed + epoch)
            loss_sum, sample_count = 0.0, 0
            for batch_index, (latent, context, mask) in enumerate(loader, start=start_batch):
                context, mask = apply_caption_dropout(context, mask, empty, config.training.caption_dropout)
                latent, context, mask = (value.to("cuda", non_blocking=True) for value in (latent, context, mask))
                noisy, times, target = sample_flow_batch(latent, config.training)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(noisy, times, context, mask).sample
                # Sum per-sample means; normalize accumulated gradients by actual sample count,
                # including incomplete bucket batches and the last accumulation group.
                per_sample = (prediction.float() - target).square().flatten(1).mean(1)
                loss = per_sample.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite flow loss; no optimizer step was performed")
                loss.backward()
                loss_sum += loss.detach().item()
                sample_count += latent.shape[0]
                boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count
                if not boundary:
                    continue
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(sample_count)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm, error_if_nonfinite=True)
                update_learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                statistics = {
                    "epoch": epoch + 1, "max_epochs": plan.max_epochs,
                    "batch": batch_index + 1, "batches_per_epoch": batch_count,
                    "step": step, "loss": loss_sum / sample_count,
                    "grad_norm": float(grad_norm), "lr": update_learning_rate,
                    "bucket": [latent.shape[-1] * 16, latent.shape[-2] * 16],
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                }
                print(json.dumps(statistics), flush=True)
                update_seconds = time.monotonic() - update_started
                telemetry.scalars(step, {
                    "train/loss": statistics["loss"], "train/learning_rate": statistics["lr"],
                    "train/grad_norm": statistics["grad_norm"], "train/epoch": epoch + (batch_index + 1) / batch_count,
                    "performance/update_seconds": update_seconds,
                    "performance/images_per_second": sample_count / max(update_seconds, 1e-9),
                })
                loss_sum, sample_count = 0.0, 0
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch == batch_count:
                    next_epoch, next_batch = epoch + 1, 0
                if plan.should_save(step, next_epoch, next_batch):
                    checkpoint_path = save_checkpoint(
                        output, network, optimizer, scheduler, config, spec, step, next_epoch, next_batch,
                    )
                    print(f"Saved {checkpoint_path}", flush=True)
                    last_checkpoint = checkpoint_path
                if next_batch == 0 and config.sample.every_epochs and next_epoch % config.sample.every_epochs == 0:
                    try:
                        from .sampling import generate_preview
                        preview = generate_preview(model, network, config, text_directory, next_epoch)
                        telemetry.image(step, preview)
                        print(f"Saved preview {preview}", flush=True)
                    except Exception as error:
                        print(f"Preview failed at epoch {next_epoch}: {error}; training continues.", file=sys.stderr, flush=True)
                update_started = time.monotonic()
                if step >= plan.max_steps:
                    break
            epoch += 1
            start_batch = 0
        return last_checkpoint
