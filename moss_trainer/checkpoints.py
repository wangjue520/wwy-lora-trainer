"""Atomic checkpoint publication and validation before loading model weights."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

import torch

from .adapters import load_adapter_weights, save_adapter
from .data import digest
from .scheduling import checkpoint_name, training_plan

CHECKPOINT_VERSION = 2
PAYLOAD_FILES = ("adapter.safetensors", "trainer-state.pt", "config.json")


def _resume_signature(config, cache_fingerprint):
    values = config.as_dict()
    # Decaying schedules depend on the original horizon; reject duration changes.
    schedule_horizon = (
        {"max_steps": config.training.max_steps, "max_epochs": config.training.max_epochs}
        if config.training.scheduler != "constant" else None
    )
    if config.training.scheduler != "cosine_with_restarts":
        values["training"].pop("cosine_restarts")
    # Preserve the exact signature of legacy AdamW + step-warmup checkpoints.
    if config.training.warmup_steps >= 0:
        values["training"].pop("warmup_ratio")
        if config.training.optimizer == "adamw" and config.training.scheduler == "constant":
            values["training"].pop("optimizer")
            values["training"].pop("scheduler")
    values.pop("sample")
    for field in ("output_dir", "save_every", "num_workers", "max_steps", "max_epochs", "save_every_epochs", "tensorboard"):
        values["training"].pop(field)
    # Retain the original HF signature when no component override is used.
    if values["model"]["format"] == "hf":
        for field in ("format", "transformer", "text_encoder", "vae"):
            values["model"].pop(field)
    if not values["network"]["train_norm"]:
        values["network"].pop("train_norm")
    # Sequential retains the pre-packed execution signature for legacy resumes.
    if values["training"]["batch_mode"] == "sequential":
        values["training"].pop("batch_mode")
    # Default execution settings retain the signature of existing checkpoints.
    if not values["training"]["fused_situ"]:
        values["training"].pop("fused_situ")
    if values["training"]["checkpoint_policy"] == "full":
        values["training"].pop("checkpoint_policy")
    for field in ("fused_rmsnorm", "checkpoint_offload", "compile_blocks"):
        if not values["training"][field]:
            values["training"].pop(field)
    if values["training"]["checkpoint_blocks"] == -1:
        values["training"].pop("checkpoint_blocks")
    if schedule_horizon is not None:
        values["training"].update(schedule_horizon)
    return digest({"config": values, "cache": cache_fingerprint})


def parameter_manifest(network, optimizer):
    named = [(name, p) for name, p in network.named_parameters() if p.requires_grad]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if [id(p) for _, p in named] != [id(p) for p in actual]:
        raise ValueError("Optimizer parameter order differs from the adapter")
    return [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in named]


def _file_info(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"size": Path(path).stat().st_size, "sha256": checksum}


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(directory, network, optimizer, scheduler, config, spec, step, epoch, next_batch):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / checkpoint_name(config, step, epoch, next_batch)
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        save_adapter(network, config.network, staging / "adapter.safetensors",
                     model_fingerprint=spec["model_fingerprint"], step=step, epoch=epoch)
        state = {
            "format_version": CHECKPOINT_VERSION,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "step": step, "epoch": epoch, "next_batch": next_batch,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "signature": _resume_signature(config, spec["fingerprint"]),
            "cache_fingerprint": spec["fingerprint"],
            "parameter_manifest": parameter_manifest(network, optimizer),
        }
        torch.save(state, staging / "trainer-state.pt")
        (staging / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
        manifest = {
            "version": CHECKPOINT_VERSION, "step": step, "epoch": epoch, "next_batch": next_batch,
            "files": {name: _file_info(staging / name) for name in PAYLOAD_FILES},
        }
        (staging / "checkpoint.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name in (*PAYLOAD_FILES, "checkpoint.json"):
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(staging)
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        staging.rename(destination)
        _sync_directory(root)
    except BaseException:
        # Only delete the private temporary directory created by this call.
        if staging.exists() and staging.resolve().parent == root.resolve():
            shutil.rmtree(staging)
        raise
    return destination


def _validate_cursor(state, config, batch_count, *, allow_complete=False):
    for key in ("step", "epoch", "next_batch"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f"Checkpoint {key} must be a nonnegative integer")
    step, epoch, start = state["step"], state["epoch"], state["next_batch"]
    accumulation = config.training.gradient_accumulation_steps
    if batch_count <= 0 or start >= batch_count or start % accumulation:
        raise ValueError("Checkpoint cursor is not at an optimizer-update boundary")
    updates_per_epoch = (batch_count + accumulation - 1) // accumulation
    if step != epoch * updates_per_epoch + start // accumulation:
        raise ValueError("Checkpoint step and bucket cursor are inconsistent")
    limit = training_plan(config.training, batch_count).max_steps
    if step > limit or (step == limit and not allow_complete):
        unit = "max_epochs" if config.training.max_epochs else "max_steps"
        raise ValueError(f"Increase {unit} beyond the saved checkpoint to continue")
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("Checkpoint scheduler position differs from optimizer step")
    for key in ("torch_rng", "cuda_rng"):
        value = state[key]
        if key == "cuda_rng" and value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"Invalid {key} state")
    random.Random().setstate(state["python_rng"])


def load_checkpoint(directory, config, spec, *, batch_count, new_stage=False):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        if manifest["version"] != CHECKPOINT_VERSION or set(manifest["files"]) != set(PAYLOAD_FILES):
            raise ValueError("Unsupported or incomplete checkpoint format")
        for name in PAYLOAD_FILES:
            if _file_info(directory / name) != manifest["files"][name]:
                raise ValueError(f"Checkpoint payload failed integrity check: {name}")
        state = torch.load(directory / "trainer-state.pt", map_location="cpu", weights_only=True)
        if state["format_version"] != CHECKPOINT_VERSION or state["step"] != manifest["step"]:
            raise ValueError("Checkpoint manifest and training state disagree")
        if state["cache_fingerprint"] != spec["fingerprint"]:
            raise ValueError("Checkpoint feature cache differs from current model/data")
        validation_config = config
        if new_stage:
            # Only the learning-rate plan may change; optimizer and data remain strict.
            saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))["training"]
            schedule_fields = ("learning_rate", "scheduler", "cosine_restarts",
                               "warmup_steps", "warmup_ratio", "max_steps", "max_epochs")
            validation_config = dataclasses.replace(
                config, training=dataclasses.replace(
                    config.training, **{key: saved[key] for key in schedule_fields}))
        if state["signature"] != _resume_signature(validation_config, spec["fingerprint"]):
            raise ValueError("Resume model, data or training settings differ from the checkpoint")
        _validate_cursor(state, validation_config, batch_count, allow_complete=new_stage)
        if new_stage and (state["next_batch"] != 0 or
                          state["step"] != training_plan(validation_config.training, batch_count).max_steps):
            raise ValueError("A new stage requires a completed run at an epoch boundary")
        from safetensors import safe_open
        with safe_open(str(directory / "adapter.safetensors"), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
        saved_network = json.loads(metadata.get("moss_network", "{}"))
        saved_network.setdefault("train_norm", False)
        if (metadata.get("moss_adapter_format") not in ("lycoris-native-v1", "comfyui-lycoris-v1")
                or metadata.get("moss_step") != str(state["step"])
                or metadata.get("moss_model_fingerprint") != spec["model_fingerprint"]
                or saved_network != config.as_dict()["network"]):
            raise ValueError("Adapter metadata does not match this training checkpoint")
        if not isinstance(state["parameter_manifest"], list):
            raise ValueError("Missing optimizer parameter manifest")
        return state
    except Exception as error:
        raise ValueError(f"Cannot resume checkpoint {directory}: {error}") from error


def validate_output_directory(output, resume, step, max_steps, *, epoch=0, max_epochs=0):
    output = Path(output)
    if not output.exists():
        return
    if not output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output}")
    entries = list(output.iterdir())
    if not resume:
        if entries:
            raise FileExistsError("Output directory is not empty; use a new directory or --resume")
        return
    if output.resolve() != Path(resume).resolve().parent and entries:
        raise FileExistsError("Resuming into another directory requires an empty output directory")
    conflicts = [
        path.name for path in entries
        if (match := re.fullmatch(r"step-(\d+)", path.name)) and step < int(match[1]) <= max_steps
    ]
    conflicts.extend(
        path.name for path in entries
        if (match := re.fullmatch(r"epoch-(\d+)", path.name)) and epoch < int(match[1]) <= max_epochs
    )
    if conflicts:
        raise FileExistsError(
            f"Later checkpoints already exist ({', '.join(sorted(conflicts))}); "
            "resume the latest checkpoint or use an empty output directory"
        )


def restore_checkpoint(directory, state, network, optimizer, scheduler, *, new_stage=False):
    if parameter_manifest(network, optimizer) != state["parameter_manifest"]:
        raise ValueError("Optimizer parameter names, shapes or ordering changed")
    load_adapter_weights(network, Path(directory) / "adapter.safetensors")
    fresh_lrs = scheduler.get_last_lr()
    fresh_base_lrs = list(scheduler.base_lrs)
    optimizer.load_state_dict(state["optimizer"])
    if new_stage:
        # Preserve Adam moments and per-parameter steps, but use the fresh schedule.
        for group, lr, base_lr in zip(optimizer.param_groups, fresh_lrs, fresh_base_lrs):
            group["lr"], group["initial_lr"] = lr, base_lr
    else:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        if state["cuda_rng"] is None:
            raise ValueError("CUDA training requires a checkpoint with CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
