from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from .buckets import BucketManager
from .config import Config
from .modeling import load_text_encoder, load_vae, pack_latents, pipeline_module, require_cuda
from .weights import model_settings

CACHE_VERSION = 2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_fingerprint(model_path):
    settings = model_settings(model_path)
    root = Path(settings.path)
    if settings.format == "comfyui":
        assets = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".py", ".jinja"))
        required = ("pipeline_moss_image.py", "transformer/config.json", "text_encoder/config.json", "vae/config.json", "tokenizer/tokenizer_config.json")
        if any(not (root / name).is_file() for name in required):
            raise FileNotFoundError("ComfyUI mode requires local MOSS configs, tokenizer and pipeline in model_dir; HF weights are not required")
        files = [(p.relative_to(root).as_posix(), p.stat().st_size, file_digest(p)) for p in assets]
        for name in ("transformer", "text_encoder", "vae"):
            path = Path(getattr(settings, name))
            if not path.is_file():
                raise FileNotFoundError(f"Missing ComfyUI {name}: {path}")
            files.append((name, path.resolve().as_posix(), path.stat().st_size, path.stat().st_mtime_ns))
        return digest({"format": "comfyui-unquantized-v1", "files": files})
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".py", ".jinja", ".safetensors"))
    if not paths or not (root / "model_index.json").is_file():
        raise FileNotFoundError(f"Missing HF model directory: {root}")
    # Large weights were verified on import. Stat changes invalidate derived caches.
    files = [
        (p.relative_to(root).as_posix(), p.stat().st_size,
         p.stat().st_mtime_ns if p.suffix == ".safetensors" else file_digest(p))
        for p in paths
    ]
    return digest(files)


def scan_dataset(config: Config):
    root = Path(config.dataset.image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Create the image/caption dataset directory first: {root}")
    buckets = BucketManager(config.dataset)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = path.with_suffix(config.dataset.caption_extension)
        if not caption_path.is_file():
            raise FileNotFoundError(f"Missing sidecar caption: {caption_path}")
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        with Image.open(path) as image:
            size = ImageOps.exif_transpose(image).size
        bucket, resized, crop = buckets.select(*size)
        record = {
            "image": path.relative_to(root).as_posix(),
            "image_sha256": file_digest(path), "caption": caption,
            "original_size": list(size), "bucket": list(bucket),
            "resized_size": list(resized), "crop": list(crop),
        }
        record["key"] = digest(record)
        records.append(record)
    if not records:
        raise ValueError(f"No supported images found in {root}")
    return records


def cache_spec(config, records):
    model_id = model_fingerprint(config.model)
    spec = {
        "version": CACHE_VERSION, "model_fingerprint": model_id,
        "dataset": dataclasses.asdict(config.dataset),
        "cache": dataclasses.asdict(config.cache),
        "vae_encoding": "posterior_mean",
        "records": records,
    }
    # Runtime batching and file locations do not change feature values.
    for key in ("batch_size", "num_repeats", "image_dir"):
        spec["dataset"].pop(key)
    spec["cache"].pop("directory")
    spec = json.loads(json.dumps(spec))
    return {"fingerprint": digest(spec), **spec}


def cache_path(config, spec):
    return Path(config.cache.directory) / spec["fingerprint"]


def _write_tensors(path, tensors, metadata):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary), metadata)
    temporary.replace(path)


def _valid_tensor_file(path, signature, key, shape=None):
    if not Path(path).is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            if (stream.metadata() or {}).get("fingerprint") != signature or key not in stream.keys():
                return False
            if shape is not None and list(stream.get_slice(key).get_shape()) != list(shape):
                return False
        return True
    except (ValueError, OSError, RuntimeError):
        return False


@torch.no_grad()
def cache_dataset(config: Config):
    require_cuda()
    records = scan_dataset(config)
    spec = cache_spec(config, records)
    directory = cache_path(config, spec)
    directory.mkdir(parents=True, exist_ok=True)
    signature = spec["fingerprint"]
    model_root = Path(config.model.path)
    captions = {digest(record["caption"]): record["caption"] for record in records}
    if config.sample.every_epochs:
        captions.update({digest(prompt): prompt for prompt in (config.sample.positive, config.sample.negative)})
    captions[digest("")] = ""  # CFG caption dropout uses the actual empty-prompt embedding.
    pending_text = [
        (key, caption) for key, caption in captions.items()
        if not _valid_tensor_file(directory / f"text-{key}.safetensors", signature, "embedding")
    ]
    if pending_text:
        from transformers import AutoTokenizer
        module = pipeline_module(model_root)
        tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
        encoder = load_text_encoder(config.model)
        pipe = module.MOSSImagePipeline(
            transformer=None, scheduler=module.MOSSImageScheduler(),
            text_encoder=encoder, tokenizer=tokenizer, vae=None,
        )
        for key, caption in tqdm(pending_text, desc="Cache frozen text encoder"):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            token_count = len(tokenizer(formatted, add_special_tokens=True)["input_ids"])
            if token_count > config.cache.max_sequence_length:
                raise ValueError(
                    f"Caption has {token_count} tokens, exceeding max_sequence_length; "
                    "shorten it or raise the limit explicitly (maximum 4096)."
                )
            # Uses the official chat template, final hidden layer and unpadded forward.
            embeds, mask = pipe.encode_prompt(caption, config.cache.max_sequence_length)
            value = embeds[0, mask[0]].to(torch.bfloat16)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite text encoder output")
            _write_tensors(directory / f"text-{key}.safetensors", {"embedding": value},
                           {"fingerprint": signature})
        del pipe, encoder, tokenizer, embeds, mask, value
        gc.collect()
        torch.cuda.empty_cache()
    pending_images = [
        record for record in records
        if not _valid_tensor_file(
            directory / f"latent-{record['key']}.safetensors", signature, "latent",
            (128, record["bucket"][1] // 16, record["bucket"][0] // 16),
        )
    ]
    if pending_images:
        vae = load_vae(config.model)
        vae.enable_tiling()
        for record in tqdm(pending_images, desc="Cache frozen VAE"):
            with Image.open(Path(config.dataset.image_dir) / record["image"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize(tuple(record["resized_size"]), Image.Resampling.LANCZOS)
                image = image.crop(tuple(record["crop"]))
                pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
            pixels = (pixels.unsqueeze(0).to("cuda") / 127.5 - 1.0)
            # Flux2/MOSS uses posterior means; sampled posterior noise is not image content.
            raw = vae.encode(pixels).latent_dist.mode()
            latent = pack_latents(raw, vae)[0]
            expected = (128, record["bucket"][1] // 16, record["bucket"][0] // 16)
            if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
                raise ValueError(f"Invalid VAE latent for {record['image']}")
            _write_tensors(directory / f"latent-{record['key']}.safetensors", {"latent": latent},
                           {"fingerprint": signature})
        del vae, pixels, raw, latent
        gc.collect()
        torch.cuda.empty_cache()
    temporary = directory / "manifest.partial.json"
    temporary.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    verify_cache(config, records=records)
    return directory


def verify_cache(config, *, records=None):
    records = scan_dataset(config) if records is None else records
    spec = cache_spec(config, records)
    directory = cache_path(config, spec)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Matching feature cache not found. Run the cache command first.")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved != spec:
        raise ValueError("Cache manifest differs from current model/dataset/settings")
    text_keys = {digest(record["caption"]) for record in records} | {digest("")}
    if config.sample.every_epochs:
        text_keys.update(digest(prompt) for prompt in (config.sample.positive, config.sample.negative))
    for key in text_keys:
        path = directory / f"text-{key}.safetensors"
        if not _valid_tensor_file(path, spec["fingerprint"], "embedding"):
            raise ValueError(f"Invalid or incomplete text cache: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            shape = stream.get_slice("embedding").get_shape()
            if len(shape) != 2 or not 0 < shape[0] <= config.cache.max_sequence_length or shape[1] != 4096:
                raise ValueError(f"Invalid text embedding dimensions: {path}")
    for record in records:
        path = directory / f"latent-{record['key']}.safetensors"
        expected = (128, record["bucket"][1] // 16, record["bucket"][0] // 16)
        if not _valid_tensor_file(path, spec["fingerprint"], "latent", expected):
            raise ValueError(f"Invalid or incomplete latent cache: {path}")
    return spec, directory


class CachedDataset(TorchDataset):
    def __init__(self, records, directory, *, text_directory=None):
        self.records, self.directory = records, Path(directory)
        self.text_directory = Path(text_directory) if text_directory is not None else self.directory

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        latent = load_file(str(self.directory / f"latent-{record['key']}.safetensors"))["latent"]
        text_key = digest(record["caption"])
        embedding = load_file(str(self.text_directory / f"text-{text_key}.safetensors"))["embedding"]
        return latent, embedding


def collate_cached(rows):
    latents, embeddings = zip(*rows)
    length = max(value.shape[0] for value in embeddings)
    context = embeddings[0].new_zeros(len(rows), length, embeddings[0].shape[1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for index, value in enumerate(embeddings):
        context[index, :value.shape[0]] = value
        mask[index, :value.shape[0]] = True
    return torch.stack(latents), context, mask
