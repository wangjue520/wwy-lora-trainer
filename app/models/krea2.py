# -*- coding: utf-8 -*-
"""Krea 2（12B single-stream MMDiT + Qwen3-VL-4B + Qwen-Image VAE）—— kohya-ss musubi-tuner。

官方建议：在 RAW 上训练，在 Turbo 上使用 / 出测试图。"""
import json
import os
import re

import paths
from .base import ModelSpec, Slot, F, dataset_subsets, q, split_args, write_atomic
from .common import common_fields, insert_after, override, drop, common_args, common_network_args, optimizer_name, SCHEDULERS, \
    scheduler_name, resolution_wh

# musubi-tuner 内置 AdamW / AdamW8bit / AdaFactor，其余按「模块.类名」加载（训练环境里都装好了）
OPTIMIZERS = [
    ["AdamW8bit", "AdamW8bit · 推荐，省显存"], ["AdamW", "AdamW · 标准"], ["AdaFactor", "Adafactor · 最省显存"],
    ["bitsandbytes.optim.PagedAdamW8bit", "PagedAdamW8bit · 显存吃紧时自动换页"],
    ["bitsandbytes.optim.Lion8bit", "Lion8bit · 学习率要比 AdamW 小 3–10 倍"], ["lion_pytorch.Lion", "Lion"],
    ["prodigyopt.Prodigy", "Prodigy · 自适应学习率，学习率填 1"],
    ["prodigyplus.ProdigyPlusScheduleFree", "Prodigy Plus ScheduleFree · 自适应，学习率填 1，调度选 constant"],
    ["pytorch_optimizer.CAME", "CAME · 省显存、收敛稳"],
    ["lora_optim.Automagic", "Automagic · 每个参数自动调学习率（学习率填起始值 1e-6～5e-5）"],
    ["lora_optim.EmoSens", "EmoSens · 按 loss 变化自动调学习率（学习率填 1）"],
    ["torch.optim.RAdam", "RAdam"],
    ["custom", "自定义（填写 模块.类名）"],
]

CF = "Comfy-Org/Krea-2"
MUSUBI_SRC = os.path.join(paths.MUSUBI, "src", "musubi_tuner")


class Krea2(ModelSpec):
    id = "krea2"
    label = "Krea 2"
    subtitle = "写实 / 通用 12B · Qwen3-VL 文本编码器 · musubi-tuner"
    engine = "musubi-tuner"
    caption_style = "natural"
    slots = [
        Slot("dit", "主模型（RAW）", ["krea2_dit"], allow_quant=True,
             help="训练要用 RAW（未蒸馏）版本，Turbo 版不适合训练。bf16 原版最好；30 系显卡也可直接用 int8_convrot 版。",
             downloads=[{"label": "Krea 2 RAW bf16", "repo": CF, "file": "diffusion_models/krea2_raw_bf16.safetensors", "size_gb": 26.28},
                        {"label": "Krea 2 RAW int8 ConvRot（显存小推荐）", "repo": CF,
                         "file": "diffusion_models/krea2_raw_int8_convrot.safetensors", "size_gb": 13.49}]),
        Slot("te", "文本编码器（Qwen3-VL-4B）", ["qwen3vl_4b"],
             help="必须是 bf16 单文件版本。",
             downloads=[{"label": "Qwen3-VL-4B bf16", "repo": CF, "file": "text_encoders/qwen3vl_4b_bf16.safetensors", "size_gb": 8.88}]),
        Slot("vae", "VAE（Qwen-Image VAE）", ["qwen_image_vae", "wan_vae"],
             downloads=[{"label": "Qwen-Image VAE", "repo": CF, "file": "vae/qwen_image_vae.safetensors", "size_gb": 0.25}]),
        Slot("turbo", "Turbo 模型（测试出图用，可选）", ["krea2_dit"], required=False,
             help="训练中测试用 Turbo 出图：8 步、不用 CFG，又快又准。不填就用 RAW 出图（28 步，较慢）。",
             downloads=[{"label": "Krea 2 Turbo bf16", "repo": CF, "file": "diffusion_models/krea2_turbo_bf16.safetensors", "size_gb": 26.28}]),
    ]
    sample_defaults = {"prompt": "a photo of a young woman standing on a city street at dusk, natural light",
                       "negative": "low quality, worst quality, blurry, distorted, deformed, bad anatomy, watermark, text",
                       "width": 1024, "height": 1024, "steps": 28, "cfg": 5.5, "seed": 42}
    turbo_sample_defaults = {"steps": 8, "cfg": 1}

    def accept(self, slot, item):
        if item["kind"] not in slot.kinds:
            return False
        if item.get("quant"):
            # 训练只能用 bf16 原版现场量化，或 ComfyUI 的 int8 ConvRot 预量化版；Turbo 测试出图只支持原版
            return slot.key == "dit" and item.get("qtype") == "int8" and "convrot" in item["name"].lower()
        if slot.key == "dit" and "turbo" in item["name"].lower():
            return False
        if slot.key == "turbo" and "raw" in item["name"].lower():
            return False
        return True

    def base_label(self, task):
        return "Krea 2"

    def is_official(self, name):
        return bool(re.match(r"krea[-_ ]?2[-_ ]?raw", name))

    def score(self, slot_key, item):
        n = item["name"].lower()
        s = 0
        if slot_key in ("dit", "turbo"):
            want, other = ("raw", "turbo") if slot_key == "dit" else ("turbo", "raw")
            if want in n:
                s += 100
            if other in n:
                s -= 200
            if item.get("quant"):
                # 训练只支持从 bf16 现场量化，或者 ComfyUI 的 int8 ConvRot 预量化版
                s += -40 if (item.get("qtype") == "int8" and "convrot" in n) else -500
        elif slot_key == "te":
            if item.get("quant"):
                s -= 500
            if "qwen3vl_4b" in n or "qwen3-vl-4b" in n:
                s += 20
        elif slot_key == "vae":
            if item.get("kind") == "qwen_image_vae":
                s += 30
        return s

    def params(self):
        f = common_fields(
            {"output_name": "mylora_krea2_v1", "learning_rate": 1e-4, "network_dim": 32, "network_alpha": 32, "batch_size": 1, "epochs": 16,
             "repeats": 5, "timestep_sampling": "krea2_shift", "discrete_flow_shift": 2.5, "lr_scheduler": "constant",
             "lr_warmup_steps": 0, "save_precision": "bf16", "cache_latents_to_disk": True, "export_comfy": True},
            optimizers=OPTIMIZERS, schedulers=SCHEDULERS[:-1] + [["rex", "rex · REX 衰减"], SCHEDULERS[-1]],
            timesteps=[["krea2_shift", "krea2_shift（按分辨率自动 shift，推荐）"], ["shift", "shift（固定 shift）"],
                       ["flux_shift", "flux_shift"], ["sigmoid", "sigmoid"], ["uniform", "uniform"]])
        override(f, "discrete_flow_shift", help="官方 1024×1024 推理用 2.5。")
        override(f, "resolution", options=[[512, "512"], [1024, "1024（推荐）"], [1280, "1280"], [1536, "1536"], ["custom", "自定义宽×高"]],
                 help="Krea 2 预训练只用了 256 / 512 / 1024，用这几个尺寸效果最好；768 这类中间尺寸会变差（已从选项里去掉）。按面积分桶，非方图自动按比例裁到接近这个面积。")
        override(f, "attn_mode", options=[["sdpa", "torch SDPA（默认，最稳）"], ["xformers", "xformers"],
                                          ["flash_attn", "flash-attn"], ["sage_attn", "SageAttention"]], default="sdpa")
        f = drop(f, "cache_latents_to_disk", "split_attn")
        f = insert_after(f, "blocks_to_swap", [
            F("quant", "主模型量化", "select", "auto", "adv", "显存 / 速度",
              options=[["auto", "自动（按显卡选择）"], ["none", "不量化 bf16（需约 40 GB 显存）"],
                       ["fp8", "fp8 scaled（40 系及以上推荐）"], ["convrot", "int8 ConvRot（30 系推荐，更快）"]],
              help="12B 模型 bf16 就要 26 GB，24 GB 显卡必须量化。量化只作用于主干，LoRA 本身仍是 bf16。"),
            F("gc_cpu_offload", "梯度检查点卸载到内存", "bool", False, "adv", "显存 / 速度",
              help="进一步省显存，速度略慢。"),
            F("te_cpu", "文本编码器分层卸载（8 GB 显卡用）", "bool", False, "adv", "显存 / 速度",
              help="Qwen3-VL-4B 文本编码器整个放进显存要约 9 GB，8 GB 显卡放不下。打开后权重留在内存，"
                   "一层一层搬进显存计算，只占 2–4 GB 显存，每条标签一两秒。缓存文本编码和训练中出测试图都会用。"),
            F("attention_only", "只训练注意力层", "bool", False, "adv", "LoRA 结构",
              help="只训练每个块的 wq/wk/wv/wo/gate，文件更小，适合画风类。"),
        ])
        f = insert_after(f, "sigmoid_scale", [
            F("num_timestep_buckets", "时间步分桶数", "int", 0, "adv", "时间步 / 噪声", min=0,
              help="把时间步均匀分成几段轮流采样，小训练集上 loss 更稳。0 = 关。"),
            F("preserve_distribution_shape", "限制时间步时保持分布形状", "bool", False, "adv", "时间步 / 噪声",
              help="设了最小 / 最大时间步时，用拒绝采样保持原分布形状，而不是整体压缩。"),
        ])
        f = insert_after(f, "attn_mode", [
            F("compile", "torch.compile 编译加速", "bool", False, "adv", "显存 / 速度",
              help="编译 28 个主块，开始训练前要多花几分钟编译，之后每步更快。需要 triton（训练环境已装）。"),
            F("compile_mode", "编译模式", "select", "default", "adv", "显存 / 速度", show_if={"compile": True},
              options=[["default", "default（推荐）"], ["reduce-overhead", "reduce-overhead"], ["max-autotune", "max-autotune（编译最久）"],
                       ["max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs"]]),
            F("cuda_allow_tf32", "允许 TF32 计算", "bool", False, "adv", "显存 / 速度", help="30 系及以上显卡开启后部分计算更快，精度影响很小。"),
            F("cuda_cudnn_benchmark", "cuDNN 自动选最快算法", "bool", False, "adv", "显存 / 速度"),
            F("disable_numpy_memmap", "缓存不用内存映射", "bool", False, "adv", "显存 / 速度",
              help="缓存放在机械硬盘或网络盘上读取很慢时可以试试。"),
            F("split_attn", "拆分注意力计算", "bool", False, "adv", "显存 / 速度",
              help="进一步省显存但变慢；用 xformers / SageAttention 时会自动开启。"),
            F("turbo_cache", "Turbo 测试模型常驻内存", "bool", False, "adv", "显存 / 速度",
              help="测试出图更快，但多占一份模型大小的内存（约 26 GB）。内存小于 64 GB 不要开。"),
        ])
        return f

    def vram_presets(self):
        return [
            {"id": "8g", "label": "8 GB", "min_gb": 7.5, "note": "量化 + 块交换 26（最多）+ 512 分辨率 + 文本编码器分层卸载；内存要 32 GB 以上，速度较慢",
             "values": {"quant": "auto", "blocks_to_swap": 26, "resolution": 512, "batch_size": 1, "gradient_checkpointing": True,
                        "gc_cpu_offload": True, "num_workers": 1, "te_cpu": True}},
            {"id": "12g", "label": "12 GB", "min_gb": 11.5, "note": "量化 + 块交换 24 + 1024 分辨率",
             "values": {"quant": "auto", "blocks_to_swap": 24, "resolution": 1024, "batch_size": 1, "gradient_checkpointing": True,
                        "gc_cpu_offload": True, "num_workers": 1}},
            {"id": "16g", "label": "16 GB", "min_gb": 15.5, "note": "量化 + 块交换 12",
             "values": {"quant": "auto", "blocks_to_swap": 12, "resolution": 1024, "batch_size": 1, "gradient_checkpointing": True,
                        "gc_cpu_offload": False, "num_workers": 2}},
            {"id": "24g", "label": "24 GB", "min_gb": 23, "note": "量化，不交换",
             "values": {"quant": "auto", "blocks_to_swap": 0, "resolution": 1024, "batch_size": 1, "gradient_checkpointing": True,
                        "gc_cpu_offload": False, "num_workers": 2}},
            {"id": "32g", "label": "32 GB", "min_gb": 31, "note": "fp8 量化，批量 2",
             "values": {"quant": "fp8", "blocks_to_swap": 0, "resolution": 1024, "batch_size": 2, "gradient_checkpointing": True,
                        "gc_cpu_offload": False, "num_workers": 2}},
            {"id": "48g", "label": "48 GB+", "min_gb": 46, "note": "不量化 bf16（质量最好）",
             "values": {"quant": "none", "blocks_to_swap": 0, "resolution": 1024, "batch_size": 2, "gradient_checkpointing": True,
                        "gc_cpu_offload": False, "num_workers": 4}},
        ]

    # ---------------- 命令 ----------------
    def resolve_quant(self, p, dit_path, gpu):
        """auto → 按显卡决定；预量化的 ConvRot 文件只能走 convrot。"""
        n = os.path.basename(dit_path or "").lower()
        if "convrot" in n or "int8" in n:
            return "convrot"
        qm = p.get("quant") or "auto"
        if qm != "auto":
            return qm
        vram = (gpu or {}).get("total_gb") or 24
        cc = (gpu or {}).get("compute_cap") or 8.6
        if vram >= 46:
            return "none"
        return "fp8" if cc >= 8.9 else "convrot"

    def build_steps(self, task, task_dir):
        import gpu as gpu_mod
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        m = task.get("models") or {}
        out_dir = task["output_dir"]
        rw, rh = resolution_wh(p)
        cache_dir = os.path.join(task_dir, "cache")
        g = gpu_mod.info()

        subsets = dataset_subsets(p.get("dataset_dir"), p.get("repeats") or 1, p.get("use_folder_repeats", True))
        lines = ["[general]", f"resolution = [{rw}, {rh}]", f"caption_extension = {q(p.get('caption_extension') or '.txt')}",
                 f"batch_size = {int(p.get('batch_size') or 1)}",
                 f"enable_bucket = {str(bool(p.get('enable_bucket', True))).lower()}",
                 f"bucket_no_upscale = {str(bool(p.get('bucket_no_upscale', True))).lower()}"]
        for i, s in enumerate(subsets):
            lines += ["", "[[datasets]]", f"image_directory = {q(s['dir'])}",
                      f"cache_directory = {q(os.path.join(cache_dir, str(i)))}", f"num_repeats = {s['repeats']}"]
        ds = os.path.join(task_dir, "dataset.toml")
        with open(ds, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        steps = [
            {"name": "缓存潜变量", "kind": "cache", "cwd": paths.MUSUBI,
             "cmd": [os.path.join(MUSUBI_SRC, "krea2_cache_latents.py"), "--dataset_config", ds, "--vae", m.get("vae", "")]},
            {"name": "缓存文本编码", "kind": "cache", "cwd": paths.MUSUBI,
             "cmd": [os.path.join(MUSUBI_SRC, "krea2_cache_text_encoder_outputs.py"), "--dataset_config", ds,
                     "--text_encoder", m.get("te", ""), "--batch_size", "4" if p.get("te_cpu") else "1"],
             "env": {"LORA_TE_OFFLOAD": "1"} if p.get("te_cpu") else {}},   # 分层卸载时一次编码几条，每层只搬一次
        ]

        quant = self.resolve_quant(p, m.get("dit"), g)
        attn = p.get("attn_mode") or "sdpa"
        a = [os.path.join(MUSUBI_SRC, "krea2_train_network.py"),
             "--dit", m.get("dit", ""), "--vae", m.get("vae", ""), "--dataset_config", ds,
             "--" + (attn if attn != "torch" else "sdpa"),
             "--mixed_precision", p.get("mixed_precision") or "bf16",
             "--save_precision", p.get("save_precision") or "bf16",
             "--timestep_sampling", p.get("timestep_sampling") or "krea2_shift",
             "--discrete_flow_shift", str(p.get("discrete_flow_shift") or 2.5),
             "--sigmoid_scale", str(p.get("sigmoid_scale") or 1.0),
             "--optimizer_type", optimizer_name(p),
             "--learning_rate", str(p.get("learning_rate")),
             "--lr_scheduler", scheduler_name(p),
             "--max_grad_norm", str(p.get("max_grad_norm") if p.get("max_grad_norm") is not None else 1.0),
             "--max_data_loader_n_workers", str(int(p.get("num_workers") or 0)),
             "--network_module", "networks.lora_krea2",
             "--network_dim", str(int(p.get("network_dim") or 32)),
             "--network_alpha", str(p.get("network_alpha") or 32),
             "--seed", str(int(p.get("seed") or 42)),
             "--gradient_accumulation_steps", str(int(p.get("grad_accum") or 1)),
             "--output_dir", out_dir, "--output_name", p.get("output_name") or "lora"]
        if attn == "sage_attn" and not p.get("split_attn"):   # xformers 的情况 common_args 已处理
            a.append("--split_attn")
        if (p.get("weighting_scheme") or "none") == "none":
            a += ["--weighting_scheme", "none"]
        if int(p.get("num_workers") or 0) > 0:
            a.append("--persistent_data_loader_workers")
        if quant == "fp8":
            a += ["--fp8_base", "--fp8_scaled"]
        elif quant == "convrot":
            a.append("--convrot_int8")
        if p.get("gradient_checkpointing"):
            a.append("--gradient_checkpointing")
            if p.get("gc_cpu_offload"):
                a.append("--gradient_checkpointing_cpu_offload")
        swap = int(p.get("blocks_to_swap") or 0)
        if swap > 0:
            a += ["--blocks_to_swap", str(min(swap, 26))]
        a += common_args(p, task_dir)
        if p.get("compile"):
            a += ["--compile", "--compile_mode", p.get("compile_mode") or "default"]
        for k in ("cuda_allow_tf32", "cuda_cudnn_benchmark", "disable_numpy_memmap", "preserve_distribution_shape"):
            if p.get(k):
                a.append("--" + k)
        if int(p.get("num_timestep_buckets") or 0) > 0:
            a += ["--num_timestep_buckets", str(int(p["num_timestep_buckets"]))]
        if (p.get("optimizer_args") or "").strip():
            a += ["--optimizer_args"] + split_args(p["optimizer_args"])
        if float(p.get("network_dropout") or 0) > 0:
            a += ["--network_dropout", str(p["network_dropout"])]
        net_args = common_network_args(p)
        if p.get("attention_only"):
            net_args.append(r"exclude_patterns=['.*\.mlp\..*','first','last\.linear','tmlp\..*','txtmlp\..*','tproj\.1','txtfusion\..*']")
        if net_args:
            a += ["--network_args"] + net_args
        if p.get("network_weights"):
            a += ["--network_weights", p["network_weights"]]
        if p.get("steps_mode") == "steps":
            a += ["--max_train_steps", str(int(p.get("max_train_steps") or 1000))]
        else:
            a += ["--max_train_epochs", str(int(p.get("epochs") or 1))]
        if int(p.get("save_every_n_epochs") or 0) > 0:
            a += ["--save_every_n_epochs", str(int(p["save_every_n_epochs"]))]
            if int(p.get("save_last_n") or 0) > 0:
                a += ["--save_last_n_epochs", str(int(p["save_last_n"]))]
        if p.get("save_state"):
            a.append("--save_state")

        notes = []
        a += self._sample_args(task, task_dir, m, swap, quant, notes)
        a += split_args(p.get("extra_args") or "")
        steps.append({"name": "训练", "kind": "train", "cmd": a, "cwd": paths.MUSUBI, "notes": notes,
                      "info": {"quant": quant}, "env": {"LORA_TE_OFFLOAD": "1"} if p.get("te_cpu") else {}})
        if p.get("export_comfy"):
            name = p.get("output_name") or "lora"
            steps.append({"name": "导出 ComfyUI 格式", "kind": "post", "cwd": paths.MUSUBI,
                          "cmd": [os.path.join(MUSUBI_SRC, "convert_lora.py"), "--input", os.path.join(out_dir, name + ".safetensors"),
                                  "--output", os.path.join(out_dir, name + "_comfy.safetensors"), "--target", "other"]})
        return steps

    def write_sample_prompts(self, task, task_dir):
        smp = task.get("sample") or {}
        prompts = [x for x in smp.get("prompts") or [] if (x.get("prompt") or "").strip()]
        if not smp.get("enabled") or not prompts:
            return None
        # 用 JSON 写测试提示词：musubi 只有「有反向提示词」时才开 CFG，而 txt 格式（--n 文字）写不出空的反向提示词，
        # 结果 CFG 被悄悄关掉，RAW 模型出的测试图发糊、像没练好（WebUI 里空反向提示词照样开 CFG）。
        # JSON 里反向提示词写成 ""，和 WebUI 的行为一致。
        f = os.path.join(task_dir, "sample_prompts.json")
        items = []
        for pr in prompts:
            d = {"prompt": (pr.get("prompt") or "").replace("\n", " ").strip()}
            for k, key, conv in (("width", "width", int), ("height", "height", int), ("steps", "sample_steps", int),
                                 ("cfg", "cfg_scale", float), ("seed", "seed", int)):
                if pr.get(k) not in (None, ""):
                    d[key] = conv(pr[k])
            if float(pr.get("cfg") or 5.5) > 1.0:
                d["negative_prompt"] = (pr.get("negative") or "").replace("\n", " ").strip()
            items.append(d)
        return write_atomic(f, json.dumps(items, ensure_ascii=False, indent=1))

    def _sample_args(self, task, task_dir, m, swap, quant, notes):
        f = self.write_sample_prompts(task, task_dir)
        if not f:
            return []
        smp = task.get("sample") or {}
        a = ["--sample_prompts", f, "--text_encoder", m.get("te", "")]
        if smp.get("use_turbo") and m.get("turbo"):
            if swap > 0:
                notes.append("开了块交换，Turbo 测试出图和块交换不能同时用，本次改用 RAW 出图。")
            elif quant == "convrot":
                notes.append("int8 ConvRot 量化暂不支持 Turbo 测试出图，本次改用 RAW 出图。")
            else:
                a += ["--turbo_dit", m["turbo"]]
                if (task.get("params") or {}).get("turbo_cache"):
                    a.append("--turbo_dit_cache")
        a += self.sample_interval_args(task)
        if smp.get("at_first"):
            a.append("--sample_at_first")
        return a
