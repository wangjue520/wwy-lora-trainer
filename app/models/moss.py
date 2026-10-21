# -*- coding: utf-8 -*-
"""MOSS-Image（OpenMOSS，5B 单流 DiT + Qwen3.5-9B 文本编码器 + FLUX.2 VAE）—— kohya-ss sd-scripts 的 moss_image_train_network.py。

和 Anima 一样是 sd-scripts 的流匹配 DiT，数据集 / 标签处理 / 测试出图 / 续训 / 冻结全部沿用 Anima 的实现，
这里只换模型槽位、参数默认值和 MOSS 特有的几个参数（ConvRot INT8 量化、文本长度、出图时跳过的层）。
官方说明见 engines/sd-scripts/docs/moss_image_train_network.md。"""
import glob
import os
import re

import paths
from .base import F, Slot
from .anima import Anima, OPTIMIZERS
from .common import override, drop, insert_after

CO = "Comfy-Org/MOSS-Image"   # ComfyUI 官方单文件（魔搭上有同名仓库，国内裸连也能下）
TURBO_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"   # 8 步加速 LoRA（在 SFT 上蒸馏，50 步 CFG 3 → 8 步 CFG 1）
TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-Base"

# Anima 特有、MOSS 没有的参数
_ANIMA_ONLY = ("qwen_image_vae_2d", "vae_chunk_size", "vae_disable_cache", "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr",
               "llm_adapter_lr", "qwen3_max_token_length", "t5_max_token_length", "llm_adapter_path", "t5_tokenizer_path",
               "export_comfy", "train_llm_adapter")


class Moss(Anima):
    id = "moss"
    label = "MOSS-Image"
    subtitle = "5B DiT · Qwen3.5-9B 文本编码器 · 自然语言"
    engine = "sd-scripts"
    family = "moss"
    engine_label = "sd-scripts"
    engine_note = "支持 int8 量化、冻结 / 续训、训练中换主模型和加速出图"
    caption_style = "natural"
    resume = True
    SCRIPT = "moss_image_train_network.py"
    TE_ARG = "--text_encoder"
    LORA_MODULE = "networks.lora_generic"
    MAX_SWAP = 40
    slots = [
        Slot("dit", "主模型（MOSS-Image DiT）", ["moss_dit"], allow_quant=True,
             help="Base（基础版）或 SFT（精调版）都能练。官方格式选第 1 个分片（…-00001-of-00003），其余分片自动加载；"
                  "ComfyUI 单文件（bf16 / int8 ConvRot）直接选。",
             downloads=[{"label": "MOSS-Image Base bf16（ComfyUI 单文件）", "repo": CO,
                         "file": "diffusion_models/moss_image_base_bf16.safetensors", "size_gb": 10.16},
                        {"label": "MOSS-Image SFT bf16", "repo": CO, "file": "diffusion_models/moss_image_sft_bf16.safetensors", "size_gb": 10.16},
                        {"label": "MOSS-Image SFT int8 ConvRot（显存小）", "repo": CO,
                         "file": "diffusion_models/moss_image_sft_int8_convrot.safetensors", "size_gb": 5.41}]),
        Slot("te", "文本编码器（Qwen3.5-9B）", ["qwen35_9b"], allow_quant=True,
             help="只用来缓存标签编码。int8 ConvRot 版（9.8 GB）比 bf16（19 GB）省一半显存和内存，效果差别很小。",
             downloads=[{"label": "Qwen3.5-9B int8 ConvRot（推荐）", "repo": CO,
                         "file": "text_encoders/qwen3.5_9b_int8_convrot.safetensors", "size_gb": 9.75},
                        {"label": "Qwen3.5-9B bf16", "repo": CO, "file": "text_encoders/qwen3.5_9b_bf16.safetensors", "size_gb": 19.31}]),
        Slot("vae", "VAE（FLUX.2 VAE）", ["flux2_vae"],
             help="ComfyUI 的 flux2-vae 或官方仓库 vae 文件夹里的都行（潜空间一样，缓存通用）。",
             downloads=[{"label": "FLUX.2 VAE", "repo": CO, "file": "vae/flux2-vae.safetensors", "size_gb": 0.34}]),
        Slot("sample_dit", "测试出图用的主模型（可选）", ["moss_dit"], required=False,
             help="训练中测试换成这个主模型出图（比如用 Base 练、用 SFT 出图），不选就用训练的主模型。只支持 bf16，训练的主模型也要是 bf16、不开 int8。"),
        Slot("turbo", "8 步加速 LoRA（测试出图用，可选）", ["lora"], required=False,
             help="训练中测试勾选「加速出图」时用。不选会自动在模型库的 loras 文件夹里找 MOSS-SFT-8step-DMD。",
             downloads=[{"label": "MOSS-Image SFT 8 步加速 LoRA（lodestones DMD）", "repo": TURBO_REPO, "file": "lora.safetensors",
                         "size_gb": 1.19}]),
    ]
    turbo_sample_defaults = {"steps": 8, "cfg": 1}
    # 官方用户指南：质量词 best quality, highres；推荐的负面 tag；出图 50 步、CFG 3.0（测试图用 40 步，28 步细节不够）
    sample_defaults = {"prompt": "best quality, highres, An anime illustration of a girl with long hair standing on a city street at dusk, "
                                 "looking at the viewer, soft warm light, detailed background.",
                       "negative": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
                       "width": 1024, "height": 1024, "steps": 40, "cfg": 3.0, "seed": 42}
    variant_groups = [
        ["official", "官方 Base / SFT（推荐）", "官方底模。"],
        ["finetune", "微调 / 合并版", "用微调版训练：LoRA 在这个微调版上效果最好，换到官方模型上可能会变样。"],
        ["turbo", "加速 / 蒸馏版（不建议训练）", "⚠ 这是加速蒸馏版，拿它训练 LoRA 效果通常不好，建议换成官方 Base / SFT。"],
    ]

    def accept(self, slot, item):
        """int8 ConvRot 能直接练；int4 / W4A8 GPTQ / GGUF 这些只能推理，训练引擎不认。"""
        n = item["name"].lower()
        if re.search(r"int4|w4a8|gptq|nf4|\.gguf$|q[2-8]_[0-9k]", n):
            return False
        return item["kind"] in slot.kinds

    def is_official(self, name):
        n = name.lower().replace("\\", "/")
        return bool(re.search(r"moss[-_ ]?image[-_ ]?(base|sft)|(^|/)(base|sft)/transformer/", n))

    def score(self, slot_key, item):
        n = item["name"].lower().replace("\\", "/")
        s = 0
        if slot_key == "dit":
            s += 100 if self.is_official(n) else 0
            s += 20 if "base" in n else (15 if "sft" in n else 0)
            s += -15 if item.get("quant") else 0          # bf16 能选更多量化方式；int8 单文件也能练
            s += -40 if "int4" in n or "gguf" in n else 0
        elif slot_key == "te":
            s += 10 if item.get("quant") else 0           # 只用来缓存，int8 省显存
            s += 5 if "moss" in n or "qwen3.5" in n or "qwen35" in n else 0
        elif slot_key == "vae":
            s += 10 if "flux2" in n or "moss" in n else 0
        elif slot_key == "sample_dit":   # 不自动选（分数不大于 0）；列表里 SFT 排前面
            s += 0 if self.is_official(n) and "sft" in n and not item.get("quant") else -1
        return s

    def base_label(self, task):
        return "MOSS"   # 共享模型库 / WebUI 里 MOSS 的 LoRA 一直放在 loras/MOSS（sd-forge-moss-image 插件的约定）

    # ---------------- 参数 ----------------
    def params(self):
        f = drop(super().params(), *_ANIMA_ONLY)
        for k, v in {"learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "lr_scheduler": "cosine", "lr_warmup_steps": 0,
                     "output_name": "mylora_moss_v1", "network_dim": 32, "network_alpha": 32, "batch_size": 1,
                     "epochs": 10, "repeats": 4, "save_every_n_epochs": 1,
                     "timestep_sampling": "sigma", "discrete_flow_shift": 1.0, "weighting_scheme": "logit_normal",
                     "logit_mean": -1.09, "logit_std": 0.95, "num_workers": 2}.items():
            override(f, k, default=v)
        override(f, "optimizer_type", options=OPTIMIZERS)
        # 2026-10-21 实测（Mika 画风 37 张，学习率 1e-4）：alpha 16 第 2–4 轮就学到画风、第 6 轮后开始跑偏；
        # 官方示例的 alpha 1 要第 6–8 轮才到同样程度。默认用 alpha = dim、每轮保存挑最好的。
        # 同一天 WebUI 实测：dim 16 练出的画风 LoRA 在 Base + CFG 3 下明显，但配 8 步加速（CFG 1）几乎看不出来；
        # 社区 rank 32、1500 步的画风 LoRA 改动约是它的 2.5 倍，加速下照样有画风。改成 dim 32。
        # kieed 画风实测：dim 32、学习率 1e-4、重复 4 × 第 8 轮，和「2e-4、余弦重启、20 轮」练到的强度一样，Base 练、SFT + 8 步加速出图都有画风。
        # 社区配方也多是 rank 16–32、学习率 1e-4。默认学习率 1e-4、余弦下降（后面几轮学习率小，结果更稳）、重复 4 × 10 轮、每轮保存。
        override(f, "timestep_sampling", options=[["sigma", "sigma（复现官方预训练分布，推荐）"], ["shift", "shift（配 flow shift 3.0 分布相近）"],
                                                   ["sigmoid", "sigmoid"], ["uniform", "uniform"], ["flux_shift", "flux_shift"]])
        override(f, "discrete_flow_shift", show_if={"timestep_sampling": ["sigma", "shift"]},
                 help="sigma 采样用 1.0（官方训练网格）；shift 采样用 3.0 近似官方分布。")
        override(f, "blocks_to_swap", max=40, help="把部分 DiT 块放到内存里轮换（最多 40），显存不够时加大，越大越慢。")
        override(f, "attn_mode", options=[["torch", "torch SDPA（默认，最稳）"], ["xformers", "xformers（会自动开拆分注意力）"],
                                          ["flash", "flash-attn"]])
        override(f, "train_te", help="官方建议冻结文本编码器。开启后不能缓存文本编码，int8 也要约 17 GB 显存（bf16 要 32 GB）。")
        override(f, "shuffle_caption", help="MOSS 用自然语言长句描述训练，一般不要开。")
        f = insert_after(f, "learning_rate", [
            F("convrot_int8", "DiT 用 int8 量化（ConvRot）", "bool", False, "lr",
              help="训练时把 DiT 主干量化成 int8：显存减半（10 GB → 5 GB），LoRA 效果和 bf16 基本一样；30 系显卡上还更快。"
                   "选的是 int8 单文件时自动生效，不用开。"),
            F("convrot_int8_text_encoder", "文本编码器用 int8 量化", "bool", True, "lr",
              help="Qwen3.5-9B 只用来缓存标签编码，int8 版显存约 8 GB（bf16 要 16 GB）。选的是 int8 单文件时自动生效。"),
        ])
        f = insert_after(f, "cache_te_to_disk", [
            F("te_offload", "文本编码器分层卸载（8 GB 显卡用）", "bool", False, "adv", "显存 / 速度",
              help="缓存标签编码时 int8 的 Qwen3.5-9B 也要 8.3 GB，8 GB 显卡放不下。打开后解码层留在内存，"
                   "一层一层搬进显存计算，只占约 2 GB，缓存会慢一些。"),
        ])
        f = insert_after(f, "network_weights", [
            F("train_text_embedder", "同时训练文本嵌入层", "bool", False, "adv", "LoRA 结构",
              help="对 DiT 里把文本编码接进来的那几层（text_embedder）也加 LoRA，不训练文本编码器本身也能调整文字理解；"
                   "可以和缓存文本编码一起用。学习率建议低一些。"),
        ])
        f += [
            F("moss_max_token_length", "标签最大长度（token）", "int", 4096, "adv", "文本编码", min=256, max=4096, step=256,
              help="含对话模板的最大 token 数，超出部分截掉。官方上限 4096；只按实际长度计算，写长不会变慢。"),
            F("tokenizer_dir", "分词器目录", "dir", "", "adv", "文本编码",
              help="留空 = 自动：先找模型旁边官方仓库的 tokenizer 文件夹，找不到就自动下载（约 20 MB）。"),
            F("convrot_int8_bwd", "int8 反向传播方式", "select", "bf16", "adv", "显存 / 速度",
              options=[["bf16", "bf16（默认，最准）"], ["int8", "int8（需要 triton，开梯度检查点时基本不提速）"]]),
            F("sample_uncond_skip_layers", "出图时 CFG 跳过的层", "text", "22", "adv", "其他",
              help="官方出图在反向分支里跳过第 22 层，留空 = 不跳过。一般不用改。"),
        ]
        return f

    def vram_presets(self):
        return [
            {"id": "8g", "label": "8 GB", "min_gb": 7.5, "note": "int8 + 块交换 36 + 文本编码器分层卸载；内存要 32 GB 以上，速度较慢",
             "values": {"convrot_int8": True, "convrot_int8_text_encoder": True, "blocks_to_swap": 36, "te_offload": True,
                        "batch_size": 1, "resolution": 1024, "gradient_checkpointing": True, "num_workers": 1}},
            {"id": "12g", "label": "12 GB", "min_gb": 11.5, "note": "DiT / 文本编码器都 int8 + 块交换 24；内存要 24 GB 以上",
             "values": {"convrot_int8": True, "convrot_int8_text_encoder": True, "blocks_to_swap": 24, "batch_size": 1,
                        "resolution": 1024, "gradient_checkpointing": True, "num_workers": 1}},
            {"id": "16g", "label": "16 GB", "min_gb": 15.5, "note": "DiT / 文本编码器都 int8（约 9 GB）",
             "values": {"convrot_int8": True, "convrot_int8_text_encoder": True, "blocks_to_swap": 0, "batch_size": 1,
                        "resolution": 1024, "gradient_checkpointing": True, "num_workers": 2}},
            {"id": "24g", "label": "24 GB", "min_gb": 23, "note": "DiT bf16（约 13 GB），文本编码器 int8",
             "values": {"convrot_int8": False, "convrot_int8_text_encoder": True, "blocks_to_swap": 0, "batch_size": 2,
                        "resolution": 1024, "gradient_checkpointing": True, "num_workers": 2}},
            {"id": "32g", "label": "32 GB+", "min_gb": 31, "note": "DiT bf16，批量 4",
             "values": {"convrot_int8": False, "convrot_int8_text_encoder": False, "blocks_to_swap": 0, "batch_size": 4,
                        "resolution": 1024, "gradient_checkpointing": True, "num_workers": 2}},
        ]

    # ---------------- 命令 ----------------
    def _vae_args(self, p):
        return []

    def _subset_extra(self, p):
        # 自然语言标签里常有换行：整段当一条标签（不开的话 sd-scripts 会把每行当成一条候选随机挑）
        return super()._subset_extra(p) + ["  enable_multiline_caption = true"]

    def _model_args(self, p, m):
        a = []
        if p.get("convrot_int8"):
            a.append("--convrot_int8")
            if (p.get("convrot_int8_bwd") or "bf16") != "bf16":
                a += ["--convrot_int8_bwd", p["convrot_int8_bwd"]]
        if p.get("convrot_int8_text_encoder", True):
            a.append("--convrot_int8_text_encoder")
        if int(p.get("moss_max_token_length") or 4096) != 4096:
            a += ["--moss_max_token_length", str(int(p["moss_max_token_length"]))]
        skip = str(p.get("sample_uncond_skip_layers") if p.get("sample_uncond_skip_layers") is not None else "22").strip()
        if skip != "22":
            a += ["--sample_uncond_skip_layers", skip]
        a += ["--tokenizer_cache_dir", self.tokenizer_dir(p, m)]
        if p.get("train_text_embedder"):
            # 和 network_args 合并（Anima 部分已经加过 --network_args 的话追加到后面）
            a += ["--network_args", "train_text_embedder=True"]
        return a

    def build_steps(self, task, task_dir):
        steps = super().build_steps(task, task_dir)
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        m = task.get("models") or {}
        for s in steps:   # 多个 --network_args 合成一个（argparse 只认最后一个）
            if s["kind"] == "train":
                s["cmd"] = _merge_network_args(s["cmd"])
                if p.get("te_offload"):   # 见 app/vendor/sitecustomize.py：MOSS 文本编码器分层卸载
                    s["env"] = dict(s.get("env") or {}, LORA_TE_OFFLOAD="1")
                s["env"] = dict(s.get("env") or {}, **self._sample_env(task, p, m, s.setdefault("notes", [])))
        tok = self.tokenizer_dir(p, m)
        if not os.path.isfile(os.path.join(tok, "tokenizer_config.json")):
            steps.insert(0, {"name": "下载 MOSS 分词器", "kind": "cache", "cwd": paths.APP,
                             "cmd": [os.path.join(paths.APP, "workers", "fetch_repo.py"), TOKENIZER_REPO, os.path.dirname(tok),
                                     "tokenizer/*"]})
        return steps

    def _sample_env(self, task, p, m, notes):
        """训练中测试换主模型 / 加速出图：传给训练进程，由 app/vendor/sitecustomize.py 在出图时换权重、挂加速 LoRA。"""
        smp = task.get("sample") or {}
        if not smp.get("enabled"):
            return {}
        env = {}
        alt = (m.get("sample_dit") or "").strip()
        if alt and os.path.normcase(os.path.abspath(alt)) != os.path.normcase(os.path.abspath(m.get("dit") or "")):
            quant = lambda path: bool(re.search(r"int8|convrot", os.path.basename(path).lower()))
            if p.get("convrot_int8") or quant(m.get("dit") or ""):
                notes.append("训练的主模型是 int8，测试出图不能换主模型，本次用训练的主模型出图（要换主模型请用 bf16、不开 int8 的显存档）。")
            elif quant(alt):
                notes.append("测试出图用的主模型是 int8 单文件，不能用来换，本次用训练的主模型出图（请选 bf16 版）。")
            else:
                env["LORA_SAMPLE_DIT"] = alt
        if smp.get("use_turbo"):
            lora = (m.get("turbo") or "").strip() or find_turbo_lora()
            if lora and os.path.isfile(lora):
                env["LORA_SAMPLE_LORA"] = lora
                # lodestones 的说明：lora.safetensors 用 1，lora_latest.safetensors 用 0.7
                env["LORA_SAMPLE_LORA_WEIGHT"] = "0.7" if "latest" in os.path.basename(lora).lower() else "1"
                if "LORA_SAMPLE_DIT" not in env and not re.search(r"sft", (m.get("dit") or "").lower()):
                    notes.append("8 步加速 LoRA 是在 SFT 上蒸馏的，用 Base 出图效果会差；可以在「测试出图用的主模型」里选 SFT。")
            else:
                notes.append("没有找到 8 步加速 LoRA（模型设置 → 8 步加速 LoRA 可以选择或下载），本次测试图不加速。")
        return env

    @staticmethod
    def tokenizer_dir(p, m):
        """分词器：设置里指定的 > 模型旁边官方仓库的 tokenizer 文件夹 > 训练器自己下载的。"""
        if (p.get("tokenizer_dir") or "").strip():
            return p["tokenizer_dir"].strip()
        roots = set()
        for path in (m.get("dit"), m.get("te")):
            if path:
                d = os.path.dirname(path)
                roots |= {os.path.dirname(d), os.path.dirname(os.path.dirname(d)), d}
        for r in sorted(roots):
            for cand in [os.path.join(r, "tokenizer")] + glob.glob(os.path.join(r, "*", "tokenizer")) + glob.glob(os.path.join(r, "MOSS-Image", "*", "tokenizer")):
                if os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                    return cand
        return os.path.join(paths.ROOT, "models", "moss_tokenizer", "tokenizer")


def find_turbo_lora():
    """在 WebUI / ComfyUI / 共享模型库的 LoRA 文件夹和训练器下载目录里找 8 步加速 LoRA（MOSS-SFT-8step-DMD）。
    原版（lora.safetensors，用 1）优先于 latest（用 0.7）。找不到返回 ""。"""
    import scanner
    roots = [paths.DOWNLOADS]
    for it in scanner._read_json(scanner.INSTALLS_FILE, {}).get("items", []):
        r = it["path"]
        roots += {"webui": [os.path.join(r, "models", "Lora")], "comfyui": [os.path.join(r, "models", "loras")],
                  "library": [os.path.join(r, "loras")]}.get(it["type"], [])
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("\\/").count(os.sep)
        for dirpath, dirnames, files in os.walk(root):
            if dirpath.count(os.sep) - base >= 3:
                dirnames[:] = []
            for fn in files:
                if fn.lower().endswith(".safetensors") and re.search(r"8.?step.*dmd|dmd.*8.?step", fn.lower()):
                    found.append(os.path.join(dirpath, fn))
    found.sort(key=lambda f: ("latest" in os.path.basename(f).lower(), len(f)))
    return found[0] if found else ""


def _merge_network_args(cmd):
    out, extra, i = [], [], 0
    while i < len(cmd):
        if cmd[i] == "--network_args":
            j = i + 1
            while j < len(cmd) and not cmd[j].startswith("--"):
                extra.append(cmd[j])
                j += 1
            i = j
            continue
        out.append(cmd[i])
        i += 1
    return out + (["--network_args"] + extra if extra else [])
