# -*- coding: utf-8 -*-
"""下载 MOSS 官方仓库里训练需要的配置 / pipeline / 分词器（不下权重）。

用法：python fetch_moss_config.py <仓库> <保存目录>
专用引擎（moss_trainer）在 comfyui 单文件模式下也需要这些文件：
pipeline_moss_image.py、transformer/config.json、text_encoder/config.json、vae/config.json、tokenizer/*

用 huggingface_hub 的 snapshot_download：它带正确的 UA，能从 HF 官方和 hf-mirror 下
（训练器自己的多连接下载器是给大文件用的，对 hf-mirror 的 API 会被 403）。"""
import os
import sys

PATTERNS = ["*.json", "*.py", "*.jinja", "*.txt", "*.model", "tokenizer/*", "scheduler/*"]


def download(dest):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=dest, allow_patterns=PATTERNS,
                      ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"],
                      max_workers=4)


if __name__ == "__main__":
    repo, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    print(f"下载 {repo} 的配置到 {dest}（HF_ENDPOINT={os.environ.get('HF_ENDPOINT') or '默认'}）", flush=True)
    try:
        download(dest)
    except Exception as e:
        if os.environ.get("HF_ENDPOINT") != "https://huggingface.co":
            print(f"当前线路失败（{e}），改用 HF 官方重试", flush=True)
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), repo, dest])
        raise
    need = ("pipeline_moss_image.py", "transformer/config.json",
            "text_encoder/config.json", "vae/config.json", "tokenizer/tokenizer_config.json")
    missing = [p for p in need if not os.path.isfile(os.path.join(dest, p))]
    if missing:
        print("缺少：" + ", ".join(missing))
        sys.exit(1)
    print("MOSS 官方配置已就绪：" + dest)
