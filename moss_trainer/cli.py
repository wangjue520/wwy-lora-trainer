from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

from .config import ALGORITHMS, load_config


def doctor():
    from .environment import environment_report
    result = environment_report()
    result["adapter_algorithms"] = list(ALGORITHMS)
    return result


def inspect_model(config):
    from .adapters import target_modules
    from .modeling import transformer_structure
    root = Path(config.model.path)
    for index in (root / "transformer").glob("*.index.json") if config.model.format == "hf" else ():
        data = json.loads(index.read_text())
        for shard in set(data["weight_map"].values()):
            if not (index.parent / shard).is_file():
                raise FileNotFoundError(f"Missing weight shard: {shard}")
    model = transformer_structure(config.model)
    targets = target_modules(model, config.network)
    return {
        "model": config.model.path, "model_format": config.model.format,
        "component_files": {k: getattr(config.model, k) for k in ("transformer", "text_encoder", "vae")},
        "algorithm": config.network.algo,
        "base_parameters": sum(p.numel() for p in model.parameters()),
        "target_count": len(targets),
        "targets": {name: list(module.weight.shape) for name, module in targets.items()},
        "text_encoder_frozen": True, "vae_frozen": True,
        "bucket_settings": {
            key: value for key, value in config.as_dict()["dataset"].items()
            if key in ("resolution", "enable_bucket") or "bucket" in key
        },
        "dataset_present": Path(config.dataset.image_dir).is_dir(),
        "mode": "structure inspection only; no weight loading or training",
    }


def bucket_summary(config):
    from .data import scan_dataset
    from .buckets import BucketBatchSampler
    from .scheduling import training_plan
    records = scan_dataset(config)
    counts = Counter(tuple(record["bucket"]) for record in records)
    batches = BucketBatchSampler(records, config.dataset.batch_size, config.dataset.num_repeats, config.training.seed).total_batches
    plan = training_plan(config.training, batches)
    return {
        "images": len(records), "repeats": config.dataset.num_repeats,
        "batches_per_epoch": batches, "optimizer_updates_per_epoch": plan.updates_per_epoch,
        "max_epochs": plan.max_epochs, "total_optimizer_updates": plan.max_steps,
        "optimizer": config.training.optimizer, "scheduler": config.training.scheduler,
        "warmup_optimizer_updates": plan.warmup_steps,
        "effective_cosine_restarts": (
            config.training.cosine_restarts if config.training.scheduler == "cosine_with_restarts" else 0
        ),
        "save_every_epochs": plan.save_every_epochs,
        "buckets": [{"width": w, "height": h, "images": n} for (w, h), n in sorted(counts.items())],
        "mean_aspect_ratio_error": sum(
            abs(r["original_size"][0] / r["original_size"][1] - r["bucket"][0] / r["bucket"][1])
            for r in records
        ) / len(records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="MOSS-Image LoRA/LyCORIS trainer (explicit subcommands)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Verify imports, TensorBoard, CUDA, attention, Triton compilation and tiny backward; no training")
    for name, help_text in (
        ("inspect", "Inspect model structure and adapter targets without loading weights"),
        ("buckets", "Plan kohya-style buckets without loading a model"),
        ("cache", "Cache frozen text and VAE features; does not train"),
        ("check-cache", "Verify the feature cache against current data and settings"),
        ("train", "Start actual adapter training; requires an existing feature cache"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="Training TOML; paths are relative to working directory")
        if name == "train":
            command.add_argument("--resume", help="Completed epoch-* or legacy step-* checkpoint directory")
            command.add_argument("--new-stage", action="store_true",
                                 help="Keep adapter/optimizer/RNG from a completed run; restart epoch/step and LR plan in a new empty output")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        else:
            config = load_config(args.config)
            if args.command == "inspect":
                result = inspect_model(config)
            elif args.command == "buckets":
                result = bucket_summary(config)
            elif args.command == "cache":
                from .data import cache_dataset
                result = {"cache_directory": str(cache_dataset(config))}
            elif args.command == "check-cache":
                from .data import verify_cache
                spec, directory = verify_cache(config)
                result = {"cache_directory": str(directory), "images": len(spec["records"]), "status": "valid"}
            else:
                from .training import train
                result = {"checkpoint": str(train(config, args.resume, new_stage=args.new_stage))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "doctor" and not result["ready"]:
            return 2
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
