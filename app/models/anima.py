# -*- coding: utf-8 -*-
"""Anima（circlestone-labs，Cosmos-Predict2 2B + Qwen3-0.6B）—— kohya-ss sd-scripts 的 anima_train_network.py。"""
import os
import re

import paths
from .base import ModelSpec, Slot, F, dataset_subsets, q, split_args, write_atomic
from .common import common_fields, insert_after, override, drop, common_args, common_network_args, optimizer_name, \
    scheduler_name, resolution_wh

# sd-scripts 内置的优化器（library/optimizer.py）+ 训练环境里装好的第三方优化器
OPTIMIZERS = [
    ["AdamW8bit", "AdamW8bit · 推荐，省显存"], ["AdamW", "AdamW · 标准"],
    ["PagedAdamW8bit", "PagedAdamW8bit · 显存吃紧时自动换页"], ["PagedAdamW", "PagedAdamW"], ["PagedAdamW32bit", "PagedAdamW32bit"],
    ["Adafactor", "Adafactor · 最省显存"],
    ["Lion", "Lion · 学习率要比 AdamW 小 3–10 倍"], ["Lion8bit", "Lion8bit"], ["PagedLion8bit", "PagedLion8bit"],
    ["SGDNesterov", "SGDNesterov"], ["SGDNesterov8bit", "SGDNesterov8bit"],
    ["Prodigy", "Prodigy · 自适应学习率，学习率填 1"],
    ["prodigyplus.ProdigyPlusScheduleFree", "Prodigy Plus ScheduleFree · 自适应，学习率填 1，调度选 constant"],
    ["DAdaptAdam", "DAdaptAdam · 自适应，学习率填 1"], ["DAdaptLion", "DAdaptLion"], ["DAdaptAdaGrad", "DAdaptAdaGrad"],
    ["DAdaptAdan", "DAdaptAdan"], ["DAdaptAdanIP", "DAdaptAdanIP"], ["DAdaptSGD", "DAdaptSGD"], ["DAdaptation", "DAdaptation"],
    ["AdamWScheduleFree", "AdamW ScheduleFree · 调度选 constant"], ["RAdamScheduleFree", "RAdam ScheduleFree"],
    ["SGDScheduleFree", "SGD ScheduleFree"],
    ["pytorch_optimizer.CAME", "CAME · 省显存、收敛稳"],
    ["lora_optim.Automagic", "Automagic · 每个参数自动调学习率（学习率填起始值 1e-6～5e-5）"],
    ["lora_optim.EmoSens", "EmoSens · 按 loss 变化自动调学习率（学习率填 1）"],
    ["custom", "自定义（填写 模块.类名）"],
]

HF = "circlestone-labs/Anima"


class Anima(ModelSpec):
    id = "anima"
    label = "Anima"
    subtitle = "二次元 2B · Qwen3 文本编码器 · sd-scripts"
    engine = "sd-scripts"
    caption_style = "tags"    # 实测 Anima LoRA 用 WD14 标签就够，不用自然语言
    slots = [
        Slot("dit", "主模型（DiT）", ["anima_dit"], help="优先选官方 base 模型：微调模型 / turbo 模型训练出的 LoRA 泛化较差。",
             downloads=[{"label": "Anima base v1.0（官方）", "repo": HF,
                         "file": "split_files/diffusion_models/anima-base-v1.0.safetensors", "size_gb": 4.18}]),
        Slot("te", "文本编码器（Qwen3-0.6B）", ["qwen3_06b"],
             downloads=[{"label": "Qwen3-0.6B base（官方）", "repo": HF,
                         "file": "split_files/text_encoders/qwen_3_06b_base.safetensors", "size_gb": 1.19}]),
        Slot("vae", "VAE（Qwen-Image VAE）", ["qwen_image_vae", "wan_vae"],
             downloads=[{"label": "Qwen-Image VAE（官方）", "repo": HF,
                         "file": "split_files/vae/qwen_image_vae.safetensors", "size_gb": 0.25}]),
    ]
    # 官方推荐：正向前缀 masterpiece, best quality, score_7, safe；负向见模型说明（circlestone-labs/Anima）
    sample_defaults = {"prompt": "masterpiece, best quality, score_7, safe, 1girl, solo, looking at viewer, upper body",
                       "negative": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration",
                       "width": 1024, "height": 1024, "steps": 30, "cfg": 4.0, "seed": 42}

    def is_official(self, name):
        return bool(re.match(r"anima[-_ ]?(base|preview)", name))   # 官方发布：anima-base / anima-preview 系列

    def score(self, slot_key, item):
        n = item["name"].lower()
        s = super().score(slot_key, item)
        if slot_key == "dit":
            if "base" in n:
                s += 100
            if re.match(r"anima[-_ ]?base", n):
                s += 40     # 官方 base 文件名
            if "turbo" in n:
                s -= 60
            if "aesthetic" in n or "preview" in n:
                s -= 10
            if n.startswith("anima"):
                s += 10
            if re.search(r"-\d{6}\.safetensors$", n):
                s -= 30     # 训练中间产物
        elif slot_key == "te":
            if "anima" in n or "qwen_3_06b" in n or "qwen3" in n:
                s += 20
            if "base" in n:
                s += 10
        elif slot_key == "vae":
            if item.get("kind") == "qwen_image_vae":
                s += 30
        return s

    resume = True   # sd-scripts：--resume 训练状态 + 跳过已练的步数；没有状态时从某一轮的 LoRA 接着练
    # 子类（MOSS-Image 等同为 sd-scripts 流匹配 DiT 的底模）改这几项就能复用整套命令
    SCRIPT = "anima_train_network.py"
    TE_ARG = "--qwen3"
    LORA_MODULE = "networks.lora_anima"
    MAX_SWAP = 26

    def resume_args(self, r):
        if r.get("mode") == "state":
            return ["--resume", r["state"], "--skip_until_initial_step"]
        if r.get("mode") == "weights":
            return ["--network_weights", r["weights"], "--initial_epoch", str(int(r["epoch"]) + 1)]
        return []

    def params(self):
        f = common_fields(
            {"learning_rate": 5e-5, "optimizer_type": "lora_optim.Automagic", "lr_scheduler": "constant", "lr_warmup_steps": 0,
             "output_name": "mylora_anima_v1", "network_dim": 16, "network_alpha": 16, "batch_size": 2, "epochs": 10,
             "repeats": 10, "export_comfy": False, "timestep_sampling": "sigmoid", "discrete_flow_shift": 3.0},
            optimizers=OPTIMIZERS,
            timesteps=[["sigmoid", "sigmoid（官方默认）"], ["shift", "shift"], ["uniform", "uniform"],
                       ["sigma", "sigma"], ["flux_shift", "flux_shift"]])
        override(f, "attn_mode", options=[["torch", "torch SDPA（默认，最稳）"], ["xformers", "xformers"],
                                          ["flash", "flash-attn"]])
        f = insert_after(f, "learning_rate", [
            F("train_te", "同时训练文本编码器", "bool", False, "lr",
              help="一般不需要。开启后不能缓存文本编码，也需要转换后才能在 ComfyUI 里用。"),
            F("text_encoder_lr", "文本编码器学习率", "float", 1e-5, "lr", step=1e-6, sci=True, show_if={"train_te": True}),
            F("train_te_only", "只训练文本编码器", "bool", False, "lr", show_if={"train_te": True},
              help="不训练 DiT，只训练 Qwen3 文本编码器（特殊用途）。"),
        ])
        f = insert_after(f, "max_grad_norm", [
            F("loss_type", "损失函数", "select", "l2", "lr",
              options=[["l2", "L2 · 默认"], ["huber", "Huber · 对异常样本更稳"], ["smooth_l1", "Smooth L1"], ["l1", "L1"]]),
            F("huber_c", "Huber 参数 c", "float", 0.1, "lr", step=0.01, show_if={"loss_type": ["huber", "smooth_l1"]}),
            F("huber_schedule", "Huber 调度", "select", "snr", "lr", show_if={"loss_type": ["huber", "smooth_l1"]},
              options=[["snr", "snr（默认）"], ["exponential", "exponential"], ["constant", "constant"]]),
            F("huber_scale", "Huber 缩放", "float", 1.0, "lr", step=0.1, show_if={"loss_type": ["huber", "smooth_l1"]}),
        ])
        f = insert_after(f, "caption_extension", [
            F("shuffle_caption", "打乱标签顺序", "bool", False, "adv", "训练细节",
              help="标签式打标时可以开，自然语言不要开。开启后不能缓存文本编码，训练会慢一些。"),
            F("keep_tokens", "固定前几个标签不打乱", "int", 1, "adv", "训练细节", min=0, show_if={"shuffle_caption": True},
              help="一般把触发词放第一个，这里填 1。"),
            F("keep_tokens_separator", "固定部分分隔符", "text", "", "adv", "训练细节", show_if={"shuffle_caption": True},
              help="例如 |||：分隔符前面的标签固定不动，后面的才打乱。"),
            F("caption_dropout_rate", "整条标签丢弃率", "float", 0.0, "adv", "训练细节", min=0, max=1, step=0.05,
              help="按比例让部分图片不带标签训练，可以提高不写触发词时的效果。"),
            F("caption_dropout_every_n_epochs", "每几轮整条丢弃一次", "int", 0, "adv", "训练细节", min=0, help="0 = 关。"),
            F("caption_tag_dropout_rate", "单个标签丢弃率", "float", 0.0, "adv", "训练细节", min=0, max=1, step=0.05,
              help="每个逗号分隔的标签按这个概率随机去掉，防止模型只认个别标签。"),
            F("flip_aug", "水平翻转增强", "bool", False, "adv", "训练细节",
              help="图片左右翻转当作新样本。不对称的角色（发饰、伤疤在一侧）不要开。"),
            F("caption_prefix", "标签统一前缀", "text", "", "adv", "训练细节",
              help="训练时在每条标签前面加上这段文字（不改标签文件），例如触发词。"),
            F("caption_suffix", "标签统一后缀", "text", "", "adv", "训练细节"),
            F("secondary_separator", "次级分隔符", "text", "", "adv", "训练细节",
              help="例如 ;;; ，用它连起来的几个标签打乱时会被当成一个整体。"),
            F("enable_wildcard", "标签通配符", "bool", False, "adv", "训练细节",
              help="标签里写 {红发|蓝发} 时每次随机取一个。"),
            F("color_aug", "颜色增强", "bool", False, "adv", "训练细节",
              help="随机微调色调。开启后不能缓存潜变量，会自动改成每步现场编码，训练明显变慢。"),
            F("random_crop", "随机裁剪", "bool", False, "adv", "训练细节",
              help="每次随机裁一块。开启后不能缓存潜变量，训练明显变慢。"),
            F("alpha_mask", "用透明通道做遮罩", "bool", False, "adv", "训练细节",
              help="PNG 透明的部分不参与训练（只学主体）。需要图片本身带透明通道。"),
        ])
        f = insert_after(f, "bucket_no_upscale", [
            F("resize_interpolation", "缩放算法", "select", "", "adv", "训练细节",
              options=[["", "默认（area）"], ["lanczos", "lanczos · 最清晰"], ["bicubic", "bicubic"], ["bilinear", "bilinear"],
                       ["area", "area"], ["nearest", "nearest"]]),
        ])
        f = insert_after(f, "max_bucket_reso", [
            F("bucket_reso_steps", "桶边长步进", "int", 64, "adv", "训练细节", min=16, step=16, show_if={"enable_bucket": True}),
        ])
        f = insert_after(f, "sigmoid_scale", [
            F("ip_noise_gamma", "输入扰动噪声 ip_noise_gamma", "float", 0.0, "adv", "时间步 / 噪声", step=0.01, min=0,
              help="给输入加一点额外噪声，能减轻过拟合。推荐 0.05–0.1，0 = 关。"),
            F("ip_noise_gamma_random_strength", "扰动强度随机化", "bool", False, "adv", "时间步 / 噪声",
              help="每步在 0 到上面的值之间随机取强度。"),
        ])
        f = insert_after(f, "resume", [
            F("reg_data_dir", "正则化图片文件夹", "dir", "", "adv", "正则化",
              help="放一些不含训练概念的同类图（带标签），减少 LoRA 污染底模的其他内容。留空 = 不用。"),
            F("reg_repeats", "正则化图片重复次数", "int", 1, "adv", "正则化", min=1),
            F("prior_loss_weight", "正则化损失权重", "float", 1.0, "adv", "正则化", step=0.1),
        ])
        f = insert_after(f, "cache_latents_to_disk", [
            F("cache_te_to_disk", "文本编码缓存写入硬盘", "bool", True, "adv", "显存 / 速度"),
            F("qwen_image_vae_2d", "VAE 用 2D 加速版", "bool", True, "adv", "显存 / 速度",
              help="官方权重转换的 2D VAE，编码结果一致，速度约 2 倍、显存约 1/3。"),
            F("vae_chunk_size", "VAE 分块大小", "int", 64, "adv", "显存 / 速度", min=2, step=2, show_if={"qwen_image_vae_2d": False},
              help="编码时分块，越小越省显存。必须是偶数。"),
            F("vae_disable_cache", "VAE 不用内部缓存", "bool", True, "adv", "显存 / 速度", show_if={"qwen_image_vae_2d": False}),
            F("vae_batch_size", "VAE 编码批量", "int", 0, "adv", "显存 / 速度", min=0, help="缓存潜变量时一次编码几张。0 = 默认。"),
            F("text_encoder_batch_size", "文本编码批量", "int", 0, "adv", "显存 / 速度", min=0, help="缓存文本编码时一次处理几条。0 = 默认。"),
            F("cpu_offload_checkpointing", "梯度检查点卸载到内存", "bool", False, "adv", "显存 / 速度",
              help="进一步省显存但变慢。不能和块交换同时用。"),
            F("disable_mmap_load_safetensors", "不用内存映射加载模型", "bool", False, "adv", "显存 / 速度",
              help="模型放在机械硬盘或网络盘上加载很慢时可以试试。"),
            F("lowram", "低内存模式", "bool", False, "adv", "显存 / 速度", help="电脑内存小（16 GB 以下）时开。"),
            F("highvram", "高显存模式", "bool", False, "adv", "显存 / 速度", help="显存充足时开，缓存时不来回清显存，稍快。"),
            F("skip_cache_check", "跳过缓存检查", "bool", False, "adv", "显存 / 速度",
              help="训练集很大、确定缓存没问题时开，启动更快。"),
            F("unsloth_offload", "激活值异步卸载到内存", "bool", False, "adv", "显存 / 速度",
              help="另一种省显存方式，不能和块交换同时用。"),
        ])
        f = insert_after(f, "network_weights", [
            F("train_llm_adapter", "同时训练 LLM Adapter", "bool", False, "adv", "LoRA 结构",
              help="让 LoRA 也作用于文本适配层，学新概念 / 新词时可能有帮助。"),
        ])
        # 网络类型放在 LoRA 结构最前面
        idx = next(i for i, x in enumerate(f) if x["key"] == "network_dropout")
        f[idx:idx] = [
            F("network_type", "网络类型", "select", "lora", "adv", "LoRA 结构",
              options=[["lora", "LoRA · 标准"], ["lokr", "LoKr · 文件小、适合画风"], ["loha", "LoHa"]],
              help="LoKr / LoHa 用同样的 dim / alpha，学习率一般要比 LoRA 高一些。"),
            F("lokr_factor", "LoKr 分解因子", "int", -1, "adv", "LoRA 结构", min=-1, show_if={"network_type": "lokr"},
              help="-1 = 自动。越小参数越多。"),
            F("network_reg_dims", "按层设置 dim", "text", "", "adv", "LoRA 结构",
              help="正则匹配层名，分号分隔，例如 .*self_attn.*=32;.*mlp.*=8"),
            F("network_reg_lrs", "按层设置学习率", "text", "", "adv", "LoRA 结构",
              help="正则匹配层名，例如 .*cross_attn.*=5e-5;.*mlp.*=1e-5"),
        ]
        f += [
            F("unet_lr", "DiT 学习率", "text", "", "adv", "分层学习率", help="留空 = 和基础学习率相同。"),
            F("self_attn_lr", "自注意力层学习率", "text", "", "adv", "分层学习率", help="留空 = 和基础学习率相同；0 = 冻结这部分。"),
            F("cross_attn_lr", "交叉注意力层学习率", "text", "", "adv", "分层学习率"),
            F("mlp_lr", "MLP 层学习率", "text", "", "adv", "分层学习率"),
            F("mod_lr", "调制层学习率", "text", "", "adv", "分层学习率"),
            F("llm_adapter_lr", "LLM Adapter 学习率", "text", "", "adv", "分层学习率", help="需要开启「同时训练 LLM Adapter」。"),
            F("qwen3_max_token_length", "Qwen3 最大 token 长度", "int", 512, "adv", "文本编码", min=64, step=64,
              help="自然语言标签很长时可以调大。"),
            F("t5_max_token_length", "T5 最大 token 长度", "int", 512, "adv", "文本编码", min=64, step=64),
            F("llm_adapter_path", "单独的 LLM Adapter 权重", "file", "", "adv", "文本编码", help="留空 = 用主模型里自带的。"),
            F("t5_tokenizer_path", "T5 分词器目录", "dir", "", "adv", "文本编码", help="留空 = 用 sd-scripts 自带的。"),
            F("validation_split", "验证集比例", "float", 0.0, "adv", "验证集", min=0, max=0.5, step=0.05,
              help="从训练集里拿出这个比例的图不参与训练，只用来算验证 loss，看有没有过拟合。0 = 关。"),
            F("validation_seed", "验证集抽取种子", "int", 0, "adv", "验证集"),
            F("validate_every_n_epochs", "每几轮验证一次", "int", 1, "adv", "验证集", min=0),
            F("validate_every_n_steps", "每几步验证一次", "int", 0, "adv", "验证集", min=0, help="0 = 不按步数。"),
            F("max_validation_steps", "每次最多验证几批", "int", 0, "adv", "验证集", min=0, help="0 = 全部。"),
            F("save_model_as", "保存格式", "select", "safetensors", "adv", "训练细节",
              options=[["safetensors", "safetensors（推荐）"], ["ckpt", "ckpt"], ["pt", "pt"]]),
        ]
        return f

    def vram_presets(self):
        return [
            {"id": "6g", "label": "6 GB", "min_gb": 5.5, "note": "分辨率降到 768，块交换 20",
             "values": {"batch_size": 1, "resolution": 768, "blocks_to_swap": 20, "gradient_checkpointing": True,
                        "num_workers": 0, "unsloth_offload": False}},
            {"id": "8g", "label": "8 GB", "min_gb": 7.5, "note": "1024 分辨率，块交换 14",
             "values": {"batch_size": 1, "resolution": 1024, "blocks_to_swap": 14, "gradient_checkpointing": True,
                        "num_workers": 1, "unsloth_offload": False}},
            {"id": "12g", "label": "12 GB", "min_gb": 11.5, "note": "批量 2，块交换 4",
             "values": {"batch_size": 2, "resolution": 1024, "blocks_to_swap": 4, "gradient_checkpointing": True,
                        "num_workers": 2, "unsloth_offload": False}},
            {"id": "16g", "label": "16 GB", "min_gb": 15.5, "note": "批量 2，不交换",
             "values": {"batch_size": 2, "resolution": 1024, "blocks_to_swap": 0, "gradient_checkpointing": True,
                        "num_workers": 2}},
            {"id": "24g", "label": "24 GB", "min_gb": 23, "note": "批量 4",
             "values": {"batch_size": 4, "resolution": 1024, "blocks_to_swap": 0, "gradient_checkpointing": True,
                        "num_workers": 2}},
            {"id": "32g", "label": "32 GB+", "min_gb": 31, "note": "批量 6，关梯度检查点更快",
             "values": {"batch_size": 6, "resolution": 1024, "blocks_to_swap": 0, "gradient_checkpointing": False,
                        "num_workers": 4}},
        ]

    # ---------------- 命令 ----------------
    def build_steps(self, task, task_dir):
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        m = task.get("models") or {}
        out_dir = task["output_dir"]
        cache_te = self._can_cache_te(p)
        cache_lat = not p.get("color_aug") and not p.get("random_crop")   # 这两种增强每步图都不同，不能缓存
        ds = self._write_dataset(p, task_dir)
        a = [self.SCRIPT,
             "--pretrained_model_name_or_path", m.get("dit", ""), self.TE_ARG, m.get("te", ""), "--vae", m.get("vae", ""),
             "--dataset_config", ds, "--output_dir", out_dir, "--output_name", p.get("output_name") or "lora",
             "--save_model_as", p.get("save_model_as") or "safetensors",
             "--network_module", {"lokr": "networks.lokr", "loha": "networks.loha"}.get(p.get("network_type"), self.LORA_MODULE),
             "--network_dim", str(int(p.get("network_dim") or 16)),
             "--network_alpha", str(p.get("network_alpha") or p.get("network_dim") or 16),
             "--learning_rate", str(p.get("learning_rate")),
             "--optimizer_type", optimizer_name(p),
             "--lr_scheduler", scheduler_name(p),
             "--max_grad_norm", str(p.get("max_grad_norm") if p.get("max_grad_norm") is not None else 1.0),
             "--timestep_sampling", p.get("timestep_sampling") or "sigmoid",
             "--discrete_flow_shift", str(p.get("discrete_flow_shift") or 1.0),
             "--sigmoid_scale", str(p.get("sigmoid_scale") or 1.0),
             "--mixed_precision", p.get("mixed_precision") or "bf16",
             "--save_precision", p.get("save_precision") or "bf16",
             "--seed", str(int(p.get("seed") or 42)),
             "--gradient_accumulation_steps", str(int(p.get("grad_accum") or 1)),
             "--attn_mode", p.get("attn_mode") or "torch",
             "--max_data_loader_n_workers", str(int(p.get("num_workers") or 0))]
        if cache_lat:
            a.append("--cache_latents")
        if cache_lat and p.get("cache_latents_to_disk"):
            a.append("--cache_latents_to_disk")
        if cache_te:
            a.append("--cache_text_encoder_outputs")
            if p.get("cache_te_to_disk"):
                a.append("--cache_text_encoder_outputs_to_disk")
        a += self._vae_args(p)
        if int(p.get("num_workers") or 0) > 0:
            a.append("--persistent_data_loader_workers")
        if p.get("gradient_checkpointing"):
            a.append("--gradient_checkpointing")
        swap = int(p.get("blocks_to_swap") or 0)
        if swap > 0:
            a += ["--blocks_to_swap", str(min(swap, self.MAX_SWAP))]
        elif p.get("unsloth_offload") and p.get("gradient_checkpointing"):
            a.append("--unsloth_offload_checkpointing")
        elif p.get("cpu_offload_checkpointing") and p.get("gradient_checkpointing"):
            a.append("--cpu_offload_checkpointing")
        for k in ("disable_mmap_load_safetensors", "lowram", "highvram", "skip_cache_check"):
            if p.get(k):
                a.append("--" + k)
        for k in ("vae_batch_size", "text_encoder_batch_size"):
            if int(p.get(k) or 0) > 0:
                a += ["--" + k, str(int(p[k]))]
        if float(p.get("ip_noise_gamma") or 0) > 0:
            a += ["--ip_noise_gamma", str(p["ip_noise_gamma"])]
            if p.get("ip_noise_gamma_random_strength"):
                a.append("--ip_noise_gamma_random_strength")
        if p.get("alpha_mask"):
            a.append("--alpha_mask")
        if float(p.get("validation_split") or 0) > 0:
            if int(p.get("validate_every_n_epochs") or 0) > 0:
                a += ["--validate_every_n_epochs", str(int(p["validate_every_n_epochs"]))]
            if int(p.get("validate_every_n_steps") or 0) > 0:
                a += ["--validate_every_n_steps", str(int(p["validate_every_n_steps"]))]
            if int(p.get("max_validation_steps") or 0) > 0:
                a += ["--max_validation_steps", str(int(p["max_validation_steps"]))]
        if p.get("t5_tokenizer_path"):
            a += ["--t5_tokenizer_path", p["t5_tokenizer_path"]]
        if p.get("train_te"):
            a += ["--text_encoder_lr", str(p.get("text_encoder_lr") or 1e-5)]
            if p.get("train_te_only"):
                a.append("--network_train_text_encoder_only")
        else:
            a.append("--network_train_unet_only")
        a += common_args(p, task_dir)
        if (p.get("optimizer_args") or "").strip():
            a += ["--optimizer_args"] + split_args(p["optimizer_args"])
        if (p.get("loss_type") or "l2") != "l2":
            a += ["--loss_type", p["loss_type"], "--huber_c", str(p.get("huber_c") or 0.1),
                  "--huber_schedule", p.get("huber_schedule") or "snr", "--huber_scale", str(p.get("huber_scale") or 1.0)]
        for k in ("unet_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr"):
            v = str(p.get(k) or "").strip()
            if v:
                float(v)   # 不是数字时直接报错，比训练到一半才失败好
                a += ["--" + k, v]
        if int(p.get("qwen3_max_token_length") or 512) != 512:
            a += ["--qwen3_max_token_length", str(int(p["qwen3_max_token_length"]))]
        if int(p.get("t5_max_token_length") or 512) != 512:
            a += ["--t5_max_token_length", str(int(p["t5_max_token_length"]))]
        if p.get("llm_adapter_path"):
            a += ["--llm_adapter_path", p["llm_adapter_path"]]
        if p.get("reg_data_dir"):
            a += ["--prior_loss_weight", str(p.get("prior_loss_weight") or 1.0)]
        net_args = common_network_args(p)
        if p.get("network_type") == "lokr" and int(p.get("lokr_factor") or -1) != -1:
            net_args.append(f"factor={int(p['lokr_factor'])}")
        if p.get("train_llm_adapter"):
            net_args.append("train_llm_adapter=true")
        for k in ("network_reg_dims", "network_reg_lrs"):
            if (p.get(k) or "").strip():
                net_args.append(f"{k}={p[k].strip()}")
        if net_args:
            a += ["--network_args"] + net_args
        if float(p.get("network_dropout") or 0) > 0:
            a += ["--network_dropout", str(p["network_dropout"])]
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

        a += self._model_args(p, m)
        a += self._sample_args(task, task_dir)
        a += split_args(p.get("extra_args") or "")
        steps = [{"name": "训练", "kind": "train", "cmd": a, "cwd": paths.SD_SCRIPTS}]
        if p.get("export_comfy"):
            name = p.get("output_name") or "lora"
            steps.append({"name": "导出 ComfyUI 格式", "kind": "post", "cwd": paths.SD_SCRIPTS,
                          "cmd": [os.path.join("networks", "convert_anima_lora_to_comfy.py"),
                                  os.path.join(out_dir, name + ".safetensors"), os.path.join(out_dir, name + "_comfy.safetensors")]})
        return steps

    def _vae_args(self, p):
        if p.get("qwen_image_vae_2d"):
            return ["--qwen_image_vae_2d"]
        a = ["--vae_chunk_size", str(int(p.get("vae_chunk_size") or 64))]
        if p.get("vae_disable_cache", True):
            a.append("--vae_disable_cache")
        return a

    def _model_args(self, p, m):
        """底模特有的参数（加在训练命令最后）。"""
        return []

    @staticmethod
    def _can_cache_te(p):
        """sd-scripts 缓存文本编码时不允许：训练文本编码器、打乱标签、丢弃标签、通配符（每步标签都可能不同）。"""
        return not (p.get("train_te") or p.get("shuffle_caption") or float(p.get("caption_dropout_rate") or 0) > 0
                    or int(p.get("caption_dropout_every_n_epochs") or 0) > 0 or float(p.get("caption_tag_dropout_rate") or 0) > 0
                    or p.get("enable_wildcard"))

    def _write_dataset(self, p, task_dir):
        """写 sd-scripts 的 dataset.toml，返回路径。"""
        rw, rh = resolution_wh(p)
        subsets = dataset_subsets(p.get("dataset_dir"), p.get("repeats") or 1, p.get("use_folder_repeats", True))
        lines = ["[general]", f"caption_extension = {q(p.get('caption_extension') or '.txt')}",
                 f"shuffle_caption = {str(bool(p.get('shuffle_caption'))).lower()}",
                 f"keep_tokens = {int(p.get('keep_tokens') or 0) if p.get('shuffle_caption') else 0}"]
        if p.get("shuffle_caption") and (p.get("keep_tokens_separator") or "").strip():
            lines.append(f"keep_tokens_separator = {q(p['keep_tokens_separator'].strip())}")
        lines += ["",
                 "[[datasets]]", f"resolution = [{rw}, {rh}]", f"batch_size = {int(p.get('batch_size') or 1)}",
                 f"enable_bucket = {str(bool(p.get('enable_bucket', True))).lower()}",
                 f"min_bucket_reso = {int(p.get('min_bucket_reso') or 256)}",
                 f"max_bucket_reso = {int(p.get('max_bucket_reso') or 2048)}",
                 f"bucket_reso_steps = {int(p.get('bucket_reso_steps') or 64)}",
                 f"bucket_no_upscale = {str(bool(p.get('bucket_no_upscale', True))).lower()}"]
        if p.get("resize_interpolation"):
            lines.append(f"resize_interpolation = {q(p['resize_interpolation'])}")
        if float(p.get("validation_split") or 0) > 0:
            lines += [f"validation_split = {float(p['validation_split'])}", f"validation_seed = {int(p.get('validation_seed') or 0)}"]
        for s in subsets:
            lines += ["", "  [[datasets.subsets]]", f"  image_dir = {q(s['dir'])}", f"  num_repeats = {s['repeats']}"]
            lines += self._subset_extra(p)
        if p.get("reg_data_dir") and os.path.isdir(p["reg_data_dir"]):
            lines += ["", "  [[datasets.subsets]]", f"  image_dir = {q(p['reg_data_dir'])}", "  is_reg = true",
                      f"  num_repeats = {int(p.get('reg_repeats') or 1)}"]
        ds = os.path.join(task_dir, "dataset.toml")
        with open(ds, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return ds

    def _subset_extra(self, p):
        out = []
        if float(p.get("caption_dropout_rate") or 0) > 0:
            out.append(f"  caption_dropout_rate = {float(p['caption_dropout_rate'])}")
        if int(p.get("caption_dropout_every_n_epochs") or 0) > 0:
            out.append(f"  caption_dropout_every_n_epochs = {int(p['caption_dropout_every_n_epochs'])}")
        if float(p.get("caption_tag_dropout_rate") or 0) > 0:
            out.append(f"  caption_tag_dropout_rate = {float(p['caption_tag_dropout_rate'])}")
        for k in ("flip_aug", "color_aug", "random_crop", "enable_wildcard", "alpha_mask"):
            if p.get(k):
                out.append(f"  {k} = true")
        # sd-scripts 把前缀 / 后缀和标签直接拼接，这里自动补上「, 」分隔
        pre, suf = (p.get("caption_prefix") or "").strip(), (p.get("caption_suffix") or "").strip()
        if pre:
            out.append(f"  caption_prefix = {q(pre.rstrip(', ') + ', ')}")
        if suf:
            out.append(f"  caption_suffix = {q(', ' + suf.lstrip(', '))}")
        if (p.get("secondary_separator") or "").strip():
            out.append(f"  secondary_separator = {q(p['secondary_separator'].strip())}")
        return out

    def write_sample_prompts(self, task, task_dir):
        smp = task.get("sample") or {}
        prompts = [x for x in smp.get("prompts") or [] if (x.get("prompt") or "").strip()]
        if not smp.get("enabled") or not prompts:
            return None
        return write_atomic(os.path.join(task_dir, "sample_prompts.txt"), "".join(self.sample_line(pr) + "\n" for pr in prompts))

    def _sample_args(self, task, task_dir):
        f = self.write_sample_prompts(task, task_dir)
        if not f:
            return []
        smp = task.get("sample") or {}
        a = ["--sample_prompts", f]
        a += self.sample_interval_args(task)
        if smp.get("at_first"):
            a.append("--sample_at_first")
        return a
