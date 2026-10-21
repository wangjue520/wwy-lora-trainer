from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import tomllib
from typing import get_type_hints


@dataclasses.dataclass(frozen=True)
class Model:
    path: str = "models/MOSS-Image-Base"
    format: str = "hf"
    transformer: str = ""
    text_encoder: str = ""
    vae: str = ""


@dataclasses.dataclass(frozen=True)
class Dataset:
    image_dir: str = "datasets/train"
    caption_extension: str = ".txt"
    resolution: tuple[int, int] = (1024, 1024)
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = True
    batch_size: int = 1
    num_repeats: int = 1


@dataclasses.dataclass(frozen=True)
class Cache:
    directory: str = "cache/train"
    max_sequence_length: int = 4096


@dataclasses.dataclass(frozen=True)
class Network:
    algo: str = "lokr"
    dim: int = 16
    alpha: float = 16.0
    factor: int = 8
    full_matrix: bool = False
    decompose_both: bool = False
    train_text_projection: bool = False
    train_norm: bool = False
    target: str = "attention-mlp"
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    block_size: int = 4
    use_tucker: bool = False
    dora_wd: bool = False
    constraint: float = 0.0


@dataclasses.dataclass(frozen=True)
class Training:
    output_dir: str = "outputs/lokr"
    max_steps: int = 1000
    max_epochs: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    batch_mode: str = "packed"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    checkpoint_policy: str = "full"
    fused_situ: bool = False
    fused_rmsnorm: bool = False
    compile_blocks: bool = False
    checkpoint_offload: bool = False
    checkpoint_blocks: int = -1
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    cosine_restarts: int = 0
    warmup_ratio: float = 0.3
    warmup_steps: int = -1  # Legacy explicit step-based warmup; -1 selects ratio.
    save_every: int = 100
    save_every_epochs: int = 0
    seed: int = 42
    timestep_sampling: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    timestep_shift: float = 1.0
    caption_dropout: float = 0.1
    num_workers: int = 0
    tensorboard: bool = False


@dataclasses.dataclass(frozen=True)
class Sample:
    positive: str = "A serene mountain landscape, soft natural light."
    negative: str = ""
    cfg: float = 3.0
    resolution: tuple[int, int] = (1024, 1024)
    scheduler: str = "moss"
    sampler: str = "euler"
    every_epochs: int = 0


@dataclasses.dataclass(frozen=True)
class Config:
    model: Model = dataclasses.field(default_factory=Model)
    dataset: Dataset = dataclasses.field(default_factory=Dataset)
    cache: Cache = dataclasses.field(default_factory=Cache)
    network: Network = dataclasses.field(default_factory=Network)
    training: Training = dataclasses.field(default_factory=Training)
    sample: Sample = dataclasses.field(default_factory=Sample)

    def as_dict(self):
        return dataclasses.asdict(self)


ALGORITHMS = ("lora", "locon", "loha", "lokr", "dylora", "glora", "ia3", "diag-oft", "boft", "full")


def _expand_option_groups(raw: dict) -> None:
    """Accept grouped TOML options without changing runtime/checkpoint fields."""
    groups = {
        "paths": {
            "image_dir": ("dataset", "image_dir"),
            "output_dir": ("training", "output_dir"),
            "model_dir": ("model", "path"),
            "model_format": ("model", "format"),
            "model": ("model", "transformer"),
            "te": ("model", "text_encoder"),
            "vae": ("model", "vae"),
            "cache_dir": ("cache", "directory"),
        },
        "optimization": {
            name: ("training", name)
            for name in ("fused_situ", "fused_rmsnorm", "compile_blocks", "checkpoint_offload")
        },
    }
    for group, aliases in groups.items():
        if group not in raw:
            continue
        values = raw.pop(group)
        if not isinstance(values, dict):
            raise ValueError(f"[{group}] must be a TOML table")
        if unknown := set(values) - aliases.keys():
            raise ValueError(f"Unknown [{group}] options: {sorted(unknown)}")
        for key, value in values.items():
            section_name, option = aliases[key]
            section = raw.setdefault(section_name, {})
            if not isinstance(section, dict):
                raise ValueError(f"[{section_name}] must be a TOML table")
            if option in section:
                raise ValueError(
                    f"Conflicting options: [{group}].{key} and [{section_name}].{option}; keep only one"
                )
            section[option] = value


def load_config(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _expand_option_groups(raw)
    training = raw.get("training", {})
    if isinstance(training, dict):
        if "warmup_steps" in training:
            if "warmup_ratio" in training:
                raise ValueError("Set either warmup_ratio or legacy warmup_steps, not both")
            training.setdefault("scheduler", "constant")
        # Keep legacy step-based files readable, but never silently mix units.
        for epoch_key, step_key in (("max_epochs", "max_steps"), ("save_every_epochs", "save_every")):
            if epoch_key in training:
                if step_key in training:
                    raise ValueError(f"Set either {epoch_key} or {step_key}, not both")
                training[step_key] = 0
    classes = {"model": Model, "dataset": Dataset, "cache": Cache, "network": Network, "training": Training, "sample": Sample}
    if unknown := set(raw) - classes.keys():
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in classes.items():
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"[{name}] must be a TOML table")
        values = dict(section)
        if unknown := set(values) - {f.name for f in dataclasses.fields(cls)}:
            raise ValueError(f"Unknown [{name}] options: {sorted(unknown)}")
        if name in ("dataset", "sample") and "resolution" in values:
            reso = values["resolution"]
            if type(reso) is int:
                values["resolution"] = (reso, reso)
            elif isinstance(reso, list):
                values["resolution"] = tuple(reso)
            else:
                raise ValueError("resolution must be an integer or an integer pair")
        sections[name] = cls(**values)
    cfg = Config(**sections)
    validate(cfg)
    return cfg


def validate(c: Config) -> None:
    d, n, t = c.dataset, c.network, c.training
    for section in (c.model, d, c.cache, n, t, c.sample):
        hints = get_type_hints(type(section))
        for field in dataclasses.fields(section):
            value = getattr(section, field.name)
            expected = hints[field.name]
            if expected in (int, bool, str) and type(value) is not expected:
                raise ValueError(f"{field.name} must be {expected.__name__}")
            if expected is float and type(value) not in (int, float):
                raise ValueError(f"{field.name} must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
    if c.model.format not in ("hf", "comfyui"):
        raise ValueError("model_format must be hf or comfyui")
    if not c.model.path.strip():
        raise ValueError("model_dir must identify the local model configuration/tokenizer directory")
    components = (c.model.transformer, c.model.text_encoder, c.model.vae)
    if c.model.format == "comfyui":
        if any(not path.strip() or Path(path).suffix.lower() != ".safetensors" for path in components):
            raise ValueError("ComfyUI format requires separate model, te and vae .safetensors files")
    elif any(components):
        raise ValueError("Separate model/te/vae files require model_format='comfyui'; use empty strings for hf")
    if not isinstance(d.resolution, (tuple, list)) or len(d.resolution) != 2 or any(type(x) is not int or x <= 0 or x % 16 for x in d.resolution):
        raise ValueError("resolution must contain two positive multiples of 16")
    if d.bucket_reso_steps < 16 or d.bucket_reso_steps % 16:
        raise ValueError("bucket_reso_steps must be a positive multiple of 16")
    if d.enable_bucket and math.prod(d.resolution) < d.bucket_reso_steps ** 2:
        raise ValueError("resolution area must be at least bucket_reso_steps squared")
    if d.min_bucket_reso <= 0 or d.max_bucket_reso < max(*d.resolution, d.min_bucket_reso):
        raise ValueError("Invalid bucket minimum/maximum resolution")
    if any(x % d.bucket_reso_steps for x in (d.min_bucket_reso, d.max_bucket_reso)):
        raise ValueError("min/max_bucket_reso must be multiples of bucket_reso_steps")
    if d.batch_size < 1 or d.num_repeats < 1:
        raise ValueError("batch_size and num_repeats must be positive")
    if not d.caption_extension.startswith("."):
        raise ValueError("caption_extension must start with a dot")
    if not 1 <= c.cache.max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be in [1, 4096]")
    if n.algo not in ALGORITHMS:
        raise ValueError(f"Supported algorithms: {', '.join(ALGORITHMS)}")
    if n.train_norm and n.algo != "lokr":
        raise ValueError("train_norm is currently supported for LoKr only")
    if n.dim <= 0 or n.alpha <= 0 or n.factor == 0 or n.factor < -1:
        raise ValueError("dim/alpha must be positive; factor must be -1 or positive")
    if n.target not in ("attention", "attention-mlp"):
        raise ValueError("target must be attention or attention-mlp")
    if n.algo == "dylora" and (n.block_size <= 0 or n.dim % n.block_size):
        raise ValueError("DyLoRA dim must be divisible by block_size")
    if n.algo == "boft" and (n.dim < 2 or n.dim % 2):
        raise ValueError("BOFT dim must be an even block size")
    if n.dora_wd and n.algo not in ("lora", "locon", "loha", "lokr"):
        raise ValueError("dora_wd is supported for LoRA/LoCon, LoHa and LoKr only")
    if n.dropout and n.algo not in ("lora", "locon"):
        raise ValueError("Ordinary dropout is only enabled for LoRA/LoCon; use module_dropout for other algorithms")
    for value in (n.dropout, n.rank_dropout, n.module_dropout, t.caption_dropout):
        if not 0 <= value < 1:
            raise ValueError("Dropout probabilities must be in [0, 1)")
    if n.constraint < 0:
        raise ValueError("constraint must be nonnegative")
    if t.max_epochs < 0 or t.max_steps < 0:
        raise ValueError("max_epochs and max_steps cannot be negative")
    if t.max_epochs:
        if t.max_steps or t.save_every or t.save_every_epochs < 1:
            raise ValueError("Epoch scheduling requires max_steps=save_every=0 and positive save_every_epochs")
    elif t.max_steps < 1 or t.save_every < 1 or t.save_every_epochs:
        raise ValueError("Legacy step scheduling requires positive max_steps/save_every and save_every_epochs=0")
    if t.learning_rate <= 0 or t.gradient_accumulation_steps < 1:
        raise ValueError("learning_rate and gradient_accumulation_steps must be positive")
    if t.batch_mode not in ("packed", "sequential"):
        raise ValueError("batch_mode must be packed or sequential")
    if t.checkpoint_policy not in ("full", "attention"):
        raise ValueError("checkpoint_policy must be full or attention")
    if t.compile_blocks:
        if not (t.gradient_checkpointing and t.fused_situ and t.fused_rmsnorm):
            raise ValueError("compile_blocks requires gradient_checkpointing, fused_situ and fused_rmsnorm")
        if t.checkpoint_policy != "full" or t.checkpoint_blocks != -1:
            raise ValueError("compile_blocks requires all full checkpoints")
        if (n.algo not in ("lora", "locon", "lokr") or n.dora_wd
                or n.dropout or n.rank_dropout or n.module_dropout):
            raise ValueError("compile_blocks currently supports LoRA/LoKr without dropout or weight decomposition")
        if n.algo == "lokr":
            if n.factor != 8 or n.decompose_both or n.use_tucker:
                raise ValueError("Compiled LoKr requires factor=8 without decompose_both/use_tucker")
            # Official targets' largest factor has width 16384 / 8 = 2048.
            # LyCORIS uses complete factors once dim reaches half that width.
            complete_factors = n.full_matrix or n.dim >= 1024
            if not complete_factors and (n.dim != 16 or n.alpha != 16):
                raise ValueError("Compiled low-rank LoKr requires dim=alpha=16; use full_matrix=true for full factors")
        elif n.dim != 16 or n.alpha != 16:
            raise ValueError("Compiled LoRA/LoCon requires dim=alpha=16")
    if not t.gradient_checkpointing and t.checkpoint_policy != "full":
        raise ValueError("checkpoint_policy='attention' requires gradient_checkpointing=true")
    if t.checkpoint_offload and not t.gradient_checkpointing:
        raise ValueError("checkpoint_offload requires gradient_checkpointing=true")
    if t.checkpoint_blocks != -1 and not 1 <= t.checkpoint_blocks <= 42:
        raise ValueError("checkpoint_blocks must be -1 (all) or between 1 and 42")
    if t.checkpoint_blocks != -1 and not t.gradient_checkpointing:
        raise ValueError("checkpoint_blocks requires gradient_checkpointing=true")
    if t.batch_mode == "packed" and d.batch_size > 1 and (
        n.algo == "dylora" or n.dropout or n.rank_dropout or n.module_dropout
    ):
        raise ValueError("Packed BS>1 requires deterministic adapters: disable adapter dropout and DyLoRA, or use batch_mode='sequential'")
    if t.optimizer not in ("adamw", "adam", "sgd", "rmsprop"):
        raise ValueError("optimizer must be adamw, adam, sgd or rmsprop")
    if t.scheduler not in ("constant", "linear", "cosine", "cosine_with_restarts"):
        raise ValueError("scheduler must be constant, linear, cosine or cosine_with_restarts")
    if not 0 <= t.warmup_ratio <= 1 or t.cosine_restarts < 0:
        raise ValueError("warmup_ratio must be in [0, 1] and cosine_restarts must be nonnegative")
    if t.warmup_steps < -1 or t.num_workers < 0 or t.weight_decay < 0 or t.max_grad_norm <= 0:
        raise ValueError("Invalid training, optimizer or worker setting")
    if t.timestep_sampling not in ("uniform", "logit_normal"):
        raise ValueError("timestep_sampling must be uniform or logit_normal")
    if t.timestep_shift <= 0 or t.logit_std <= 0:
        raise ValueError("timestep_shift and logit_std must be positive")
    if c.sample.every_epochs < 0 or c.sample.cfg < 0:
        raise ValueError("sample.every_epochs and sample.cfg must be nonnegative")
    if (not isinstance(c.sample.resolution, (tuple, list)) or len(c.sample.resolution) != 2
            or any(type(x) is not int or x <= 0 or x % 16 for x in c.sample.resolution)):
        raise ValueError("sample.resolution must contain two positive multiples of 16")
    if c.sample.scheduler not in ("moss", "linear") or c.sample.sampler not in ("euler", "heun"):
        raise ValueError("sample supports scheduler=moss/linear and sampler=euler/heun")
