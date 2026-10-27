# -*- coding: utf-8 -*-
"""SDXL（含 Illustrious / NoobAI / Pony / Animagine 等微调）—— kohya-ss sd-scripts 的 sdxl_train_network.py。

和 Anima 共用 sd-scripts 的数据集配置、标签处理、测试出图、续训，所以直接继承 Anima，只换模型槽位、参数和训练命令。"""
import os
import re

import paths
from .base import F, Slot, remove_legacy_latent_cache, split_args
from .anima import Anima, OPTIMIZERS
from .common import common_args, common_network_args, optimizer_name, scheduler_name, override, drop, insert_after

# 各家官方底模的文件名（下载站 / HuggingFace 原名）
OFFICIAL = re.compile(r"illustrious[-_ ]?xl[-_ ]?v\d|illustriousxl[-_ ]?v?\d|noobai[-_ ]?xl[-_ ]?(vpred[-_ ]?)?v?\d|"
                      r"ponydiffusionv6xl|sd_xl_base_1\.0|animagine[-_ ]?xl[-_ ]?\d")
TURBO = re.compile(r"turbo|lightning|hyper|dmd|lcm|[-_]\d+steps?\b")

# Anima 特有、SDXL 没有的参数
_ANIMA_ONLY = ("timestep_sampling", "discrete_flow_shift", "sigmoid_scale", "weighting_scheme", "logit_mean", "logit_std",
               "mode_scale", "blocks_to_swap", "attn_mode", "split_attn", "qwen_image_vae_2d", "vae_chunk_size", "vae_disable_cache",
               "text_encoder_batch_size", "unsloth_offload", "train_llm_adapter", "network_reg_dims", "network_reg_lrs",
               "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr", "qwen3_max_token_length",
               "t5_max_token_length", "llm_adapter_path", "t5_tokenizer_path", "export_comfy")


def family(item):
    """主模型属于哪一类：official / illustrious / noobai / vpred / pony / finetune / turbo。
    item 里有文件名，可能还有 .civitai.info 的 base_model 和文件头里的 vpred 标记。"""
    n = (item.get("name") or os.path.basename(item.get("path") or "")).lower()
    b = (item.get("base_model") or "").lower()
    if TURBO.search(n) or re.search(r"turbo|lightning|hyper|lcm", b):
        return "turbo"
    if OFFICIAL.search(n):
        return "official"
    if item.get("vpred"):
        return "vpred"
    if "noob" in n or "noobai" in b:
        return "noobai"
    if "illustrious" in n or "illustrious" in b or re.search(r"(^|[-_ ])ilxl|(^|[-_ ])il[-_ ]?v\d", n):
        return "illustrious"
    if "pony" in n or "pony" in b or re.search(r"autismmix|(^|[-_ ])pdxl", n):
        return "pony"
    return "finetune"


class SDXL(Anima):
    id = "sdxl"
    label = "SDXL"
    subtitle = "Illustrious / NoobAI / Pony 等 · 单文件大模型 · sd-scripts"
    engine = "sd-scripts"
    caption_style = "tags"
    resume = True
    slots = [
        Slot("dit", "大模型（SDXL checkpoint）", ["sdxl_checkpoint"],
              help="二次元推荐 Illustrious / NoobAI 系；用官方底模练出的 LoRA 在同系列各个微调版上都通用。",
              downloads=[
                  {"label": "Illustrious-XL v0.1（官方，二次元最通用）", "repo": "OnomaAIResearch/Illustrious-xl-early-release-v0",
                   "file": "Illustrious-XL-v0.1.safetensors", "size_gb": 6.94},
                  {"label": "NoobAI-XL v1.1（官方 eps 版）", "repo": "Laxhar/noobai-XL-1.1",
                   "file": "NoobAI-XL-v1.1.safetensors", "size_gb": 7.11},
                  {"label": "NoobAI-XL v-pred 1.0（官方）", "repo": "Laxhar/noobai-XL-Vpred-1.0",
                   "file": "NoobAI-XL-Vpred-v1.0.safetensors", "size_gb": 7.11},
                  {"label": "SDXL 1.0 base（官方原版，写实 / 通用）", "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                   "file": "sd_xl_base_1.0.safetensors", "size_gb": 6.94}]),
        Slot("vae", "VAE（可选）", ["sdxl_vae", "sd_vae"], required=False,
              help="留空 = 用大模型自带的 VAE（一般就这样）。混合精度选 fp16 时建议换成 fp16-fix 版，否则可能出黑图 / NaN。",
              downloads=[{"label": "SDXL VAE fp16-fix", "repo": "madebyollin/sdxl-vae-fp16-fix", "file": "sdxl.vae.safetensors",
                          "size_gb": 0.33}]),
    ]
    # 各系列官方推荐的标准正向 / 负向（Illustrious-XL、NoobAI-XL 模型说明；Pony 的 score 标签）
    PROMPTS = {
        "illustrious": ("masterpiece, best quality, amazing quality, very aesthetic, absurdres, 1girl, solo, looking at viewer, upper body",
                        "worst quality, bad quality, low quality, lowres, displeasing, very displeasing, bad anatomy, bad hands, "
                        "scan artifacts, jpeg artifacts, signature, watermark, multiple views, extra digits, fewer digits"),
        "noobai": ("masterpiece, best quality, newest, absurdres, highres, safe, 1girl, solo, looking at viewer, upper body",
                   "nsfw, worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, "
                   "mammal, anthro, furry, ambiguous form, feral, semi-anthro"),
        "pony": ("score_9, score_8_up, score_7_up, source_anime, 1girl, solo, looking at viewer, upper body",
                 "score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, watermark"),
        "sdxl": ("high quality, highly detailed, a young woman standing on a city street at dusk, natural light",
                 "lowres, worst quality, low quality, blurry, bad anatomy, bad hands, watermark, text"),
    }
    sample_defaults = {"prompt": PROMPTS["illustrious"][0], "negative": PROMPTS["illustrious"][1],
                       "width": 1024, "height": 1024, "steps": 28, "cfg": 6.0, "seed": 42}

    def sample_defaults_for(self, models):
        path = (models or {}).get("dit") or ""
        n = os.path.basename(path).lower()
        f = family(self._item(path)) if path else "illustrious"
        if f in ("noobai", "vpred") or (f == "official" and "noob" in n):
            key = "noobai"
        elif f == "pony" or (f == "official" and "pony" in n):
            key = "pony"
        elif f == "illustrious" or (f == "official" and ("illustrious" in n or "animagine" in n)):
            key = "illustrious"
        else:
            key = "sdxl"   # 原版 SDXL / 写实微调：不认 D 站质量标签
        pos, neg = self.PROMPTS[key]
        return dict(self.sample_defaults, prompt=pos, negative=neg)
    variant_groups = [
        ["official", "官方底模（推荐）", "官方底模练出来的 LoRA 在同系列的各种微调版上都通用。"],
        ["illustrious", "Illustrious 系微调", "LoRA 在这个微调版上效果最好，用在其他 Illustrious / NoobAI 模型上基本通用；想更通用就选 Illustrious-XL 官方底模。"],
        ["noobai", "NoobAI 系微调", "LoRA 在这个微调版上效果最好，用在其他 NoobAI / Illustrious 模型上基本通用。"],
        ["vpred", "v-pred 模型", "v-pred 模型：训练时会自动开启 v-parameterization 和 zero terminal SNR；练出的 LoRA 只适合用在 v-pred 模型上。"],
        ["pony", "Pony 系微调", "Pony 系的 LoRA 一般只在 Pony 模型上好用；测试提示词建议在前面加 score_9, score_8_up, score_7_up。"],
        ["finetune", "其他 SDXL 微调 / 写实", "用微调版训练：LoRA 在这个微调版上效果最好，换到别的模型上可能会变样。"],
        ["turbo", "Turbo / Lightning 加速版（不建议训练）", "⚠ 这是加速蒸馏版，拿它训练 LoRA 效果通常不好，建议换成官方底模或普通微调版。"],
    ]

    def is_official(self, name):
        return bool(OFFICIAL.search(name))

    def variant(self, slot_key, item):
        return family(item) if slot_key == "dit" else ""

    def score(self, slot_key, item):
        s = 0
        n = item["name"].lower()
        if slot_key == "dit":
            s += {"official": 100, "illustrious": 40, "noobai": 35, "vpred": 20, "pony": 10, "finetune": 0, "turbo": -60}[family(item)]
            if "illustrious" in n:
                s += 5
            if re.search(r"-\d{6}\.safetensors$", n):
                s -= 30     # 训练中间产物
        elif slot_key == "vae":
            s = -10 if re.search(r"sd1\.?5|ft-mse|840000|kl-f8-anime", n) else 0   # 不自动选（留空 = 用大模型自带的）
        return s

    def base_label(self, task):
        """送到启动器时的底模分类（和启动器 model_library 的 Illustrious / NoobAI / Pony / SDXL 对应）。"""
        p = (task.get("models") or {}).get("dit") or ""
        f = family(self._item(p))
        n = os.path.basename(p).lower()
        if f == "pony":
            return "Pony"
        if f == "noobai" or (f in ("official", "vpred") and "noob" in n):
            return "NoobAI"
        if f == "illustrious" or (f == "official" and "illustrious" in n):
            return "Illustrious"
        return "SDXL"

    @staticmethod
    def _item(path):
        """主模型文件的信息：文件名 + .civitai.info 的底模 + 文件头里的 v-pred 标记。"""
        it = {"name": os.path.basename(path), "path": path}
        if path and os.path.isfile(path):
            try:
                import scanner
                import safetensors_meta as stm
                it.update(scanner.sidecar(path))
                it["vpred"] = bool(stm.classify(path).get("vpred"))
            except Exception:
                pass
        return it

    # ---------------- 参数 ----------------
    def params(self):
        f = drop(super().params(), *_ANIMA_ONLY)
        for k, v in {"learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "lr_scheduler": "cosine", "lr_warmup_steps": 0,
                     "output_name": "mylora_sdxl_v1", "network_dim": 16, "network_alpha": 16, "batch_size": 2, "epochs": 10,
                     "repeats": 10, "text_encoder_lr": 1e-5}.items():
            override(f, k, default=v)
        override(f, "optimizer_type", options=OPTIMIZERS)
        override(f, "resolution", options=[[768, "768"], [1024, "1024（SDXL 标准）"], [1280, "1280"], [1536, "1536"],
                                           ["custom", "自定义宽×高"]])
        override(f, "unet_lr", label="UNet 学习率")
        override(f, "train_te", help="SDXL 有两个文本编码器（CLIP-L + OpenCLIP-G）。一般不需要；人物 LoRA 想让触发词更牢可以开，"
                                     "学习率要比 UNet 低 5–10 倍，开启后不能缓存文本编码、显存多用约 2 GB。")
        override(f, "train_te_only", help="不训练 UNet，只训练文本编码器（特殊用途）。")
        f = insert_after(f, "network_alpha", [
            F("v_pred", "预测方式", "select", "auto", "basic",
              options=[["auto", "自动（按大模型判断）"], ["eps", "eps（普通模型）"], ["vpred", "v-pred（NoobAI v-pred 等）"]],
              help="v-pred 模型会同时开启 v-parameterization 和 zero terminal SNR。选错会练坏，一般保持自动。"),
        ])
        f = insert_after(f, "max_grad_norm", [
            F("min_snr_gamma", "Min-SNR γ", "float", 5.0, "lr", step=0.5, min=0,
              help="平衡不同噪声强度的损失，SDXL 训练普遍用 5，收敛更稳。0 = 关。"),
        ])
        f = insert_after(f, "max_timestep", [
            F("noise_offset", "噪声偏移 noise_offset", "float", 0.0, "adv", "时间步 / 噪声", step=0.01, min=0,
              help="让模型更会画很暗 / 很亮的画面，常用 0.0357。v-pred 模型不需要（zero terminal SNR 已经解决了）。0 = 关。"),
            F("noise_offset_random_strength", "噪声偏移强度随机化", "bool", False, "adv", "时间步 / 噪声"),
            F("adaptive_noise_scale", "自适应噪声偏移", "float", 0.0, "adv", "时间步 / 噪声", step=0.01, min=0,
              help="按图片平均亮度额外加偏移，需要先开噪声偏移。推荐 0.00357，0 = 关。"),
            F("multires_noise_iterations", "多分辨率噪声迭代", "int", 0, "adv", "时间步 / 噪声", min=0, max=20,
              help="另一种增强明暗 / 细节的噪声，推荐 6–10。不要和噪声偏移同时开。0 = 关。"),
            F("multires_noise_discount", "多分辨率噪声衰减", "float", 0.3, "adv", "时间步 / 噪声", step=0.05, min=0, max=1,
              help="配合上一项使用，默认 0.3。"),
            F("zero_terminal_snr", "zero terminal SNR", "bool", False, "adv", "时间步 / 噪声",
              help="v-pred 模型会自动开，普通 eps 模型不要开。"),
            F("debiased_estimation_loss", "去偏估计损失", "bool", False, "adv", "时间步 / 噪声",
              help="另一种平衡各噪声强度损失的方法，和 Min-SNR 二选一。"),
            F("scale_v_pred_loss_like_noise_pred", "v-pred 损失按 eps 缩放", "bool", False, "adv", "时间步 / 噪声",
              help="只对 v-pred 模型有效，部分教程推荐和 Min-SNR 一起开。"),
        ])
        f = insert_after(f, "cache_te_to_disk", [
            F("fp8_base", "大模型用 fp8 加载（省显存）", "bool", False, "adv", "显存 / 速度",
              help="UNet 以 fp8 放进显存，省约 2.5 GB，质量几乎不变。8 GB 以下显卡建议开。"),
            F("no_half_vae", "VAE 用 fp32", "bool", False, "adv", "显存 / 速度",
              help="混合精度选 fp16、又没换 fp16-fix VAE 时要开，否则缓存潜变量会出 NaN。bf16 不需要。"),
        ])
        f = insert_after(f, "caption_extension", [
            F("max_token_length", "标签最大长度（token）", "select", 225, "adv", "训练细节",
              options=[[75, "75（不分段）"], [150, "150"], [225, "225（标签多时用，推荐）"]],
              help="标签很长时超出部分会被截掉。D 站标签式打标一般超过 75，选 225。"),
        ])
        idx = next(i for i, x in enumerate(f) if x["key"] == "network_type")
        f[idx + 1:idx + 1] = [
            F("conv_dim", "卷积层 dim（LoCon）", "int", 0, "adv", "LoRA 结构", min=0, max=128, show_if={"network_type": "lora"},
              help="同时训练 UNet 的卷积层，对画风 / 质感学得更全，文件更大。推荐 8–16，0 = 关。"),
            F("conv_alpha", "卷积层 alpha", "float", 1, "adv", "LoRA 结构", min=0, step=1, show_if={"network_type": "lora"}),
        ]
        f += [
            F("attn_mode", "注意力实现", "select", "sdpa", "adv", "显存 / 速度",
              options=[["sdpa", "torch SDPA（默认，最稳）"], ["xformers", "xformers"], ["mem_eff_attn", "省显存注意力（最慢）"]]),
            F("sample_sampler", "测试图采样器", "select", "euler_a", "adv", "其他",
              options=[[x, x] for x in ("euler_a", "euler", "dpmsolver++", "ddim", "k_dpm_2_a", "dpmsingle", "heun", "k_lms")],
              help="训练中出测试图用的采样器。"),
        ]
        return f

    def vram_presets(self):
        return [
            {"id": "6g", "label": "6 GB", "min_gb": 5.5, "note": "fp8 大模型，768 分辨率",
             "values": {"batch_size": 1, "resolution": 768, "fp8_base": True, "gradient_checkpointing": True, "num_workers": 0,
                        "train_te": False}},
            {"id": "8g", "label": "8 GB", "min_gb": 7.5, "note": "fp8 大模型，1024 分辨率",
             "values": {"batch_size": 1, "resolution": 1024, "fp8_base": True, "gradient_checkpointing": True, "num_workers": 1,
                        "train_te": False}},
            {"id": "12g", "label": "12 GB", "min_gb": 11.5, "note": "批量 2",
             "values": {"batch_size": 2, "resolution": 1024, "fp8_base": False, "gradient_checkpointing": True, "num_workers": 2}},
            {"id": "16g", "label": "16 GB", "min_gb": 15.5, "note": "批量 4",
             "values": {"batch_size": 4, "resolution": 1024, "fp8_base": False, "gradient_checkpointing": True, "num_workers": 2}},
            {"id": "24g", "label": "24 GB", "min_gb": 23, "note": "批量 6",
             "values": {"batch_size": 6, "resolution": 1024, "fp8_base": False, "gradient_checkpointing": True, "num_workers": 2}},
            {"id": "32g", "label": "32 GB+", "min_gb": 31, "note": "批量 8",
             "values": {"batch_size": 8, "resolution": 1024, "fp8_base": False, "gradient_checkpointing": True, "num_workers": 4}},
        ]

    # ---------------- 命令 ----------------
    def is_vpred(self, task):
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        mode = p.get("v_pred") or "auto"
        if mode != "auto":
            return mode == "vpred"
        return bool(self._item((task.get("models") or {}).get("dit") or "").get("vpred"))

    def build_steps(self, task, task_dir):
        p = dict(self.defaults())
        p.update(task.get("params") or {})
        m = task.get("models") or {}
        out_dir = task["output_dir"]
        cache_te = self._can_cache_te(p)
        cache_lat = not p.get("color_aug") and not p.get("random_crop")
        ds = self._write_dataset(p, task_dir)
        vpred = self.is_vpred(task)
        notes = []
        if cache_lat and p.get("cache_latents_to_disk"):
            n = remove_legacy_latent_cache(p)
            if n:
                notes.append(f"训练集里有 {n} 个旧格式的潜变量缓存（图片名.npz，以前别的分辨率 / 别的训练器留下的），尺寸和这次分桶对不上会让训练报错，已删除，会重新缓存")

        a = ["sdxl_train_network.py",
             "--pretrained_model_name_or_path", m.get("dit", ""),
             "--dataset_config", ds, "--output_dir", out_dir, "--output_name", p.get("output_name") or "lora",
             "--save_model_as", p.get("save_model_as") or "safetensors",
             "--network_module", {"lokr": "networks.lokr", "loha": "networks.loha"}.get(p.get("network_type"), "networks.lora"),
             "--network_dim", str(int(p.get("network_dim") or 16)),
             "--network_alpha", str(p.get("network_alpha") or 1),
             "--learning_rate", str(p.get("learning_rate")),
             "--optimizer_type", optimizer_name(p),
             "--lr_scheduler", scheduler_name(p),
             "--max_grad_norm", str(p.get("max_grad_norm") if p.get("max_grad_norm") is not None else 1.0),
             "--mixed_precision", p.get("mixed_precision") or "bf16",
             "--save_precision", p.get("save_precision") or "bf16",
             "--seed", str(int(p.get("seed") or 42)),
             "--gradient_accumulation_steps", str(int(p.get("grad_accum") or 1)),
             "--max_data_loader_n_workers", str(int(p.get("num_workers") or 0)),
             "--" + {"xformers": "xformers", "mem_eff_attn": "mem_eff_attn"}.get(p.get("attn_mode"), "sdpa")]
        if m.get("vae"):
            a += ["--vae", m["vae"]]
        if vpred:
            a += ["--v_parameterization", "--zero_terminal_snr"]
            if p.get("scale_v_pred_loss_like_noise_pred"):
                a.append("--scale_v_pred_loss_like_noise_pred")
        elif p.get("zero_terminal_snr"):
            a.append("--zero_terminal_snr")
        if cache_lat:
            a.append("--cache_latents")
            if p.get("cache_latents_to_disk"):
                a.append("--cache_latents_to_disk")
        if cache_te:
            a.append("--cache_text_encoder_outputs")
            if p.get("cache_te_to_disk"):
                a.append("--cache_text_encoder_outputs_to_disk")
        if int(p.get("num_workers") or 0) > 0:
            a.append("--persistent_data_loader_workers")
        if p.get("gradient_checkpointing"):
            a.append("--gradient_checkpointing")
            if p.get("cpu_offload_checkpointing"):
                a.append("--cpu_offload_checkpointing")
        for k in ("fp8_base", "no_half_vae", "disable_mmap_load_safetensors", "lowram", "highvram", "skip_cache_check",
                  "debiased_estimation_loss", "alpha_mask"):
            if p.get(k):
                a.append("--" + k)
        if int(p.get("vae_batch_size") or 0) > 0:
            a += ["--vae_batch_size", str(int(p["vae_batch_size"]))]
        if int(p.get("max_token_length") or 75) in (150, 225):
            a += ["--max_token_length", str(int(p["max_token_length"]))]
        if float(p.get("min_snr_gamma") or 0) > 0:
            a += ["--min_snr_gamma", str(p["min_snr_gamma"])]
        if float(p.get("noise_offset") or 0) > 0 and not vpred:
            a += ["--noise_offset", str(p["noise_offset"])]
            if p.get("noise_offset_random_strength"):
                a.append("--noise_offset_random_strength")
            if float(p.get("adaptive_noise_scale") or 0) > 0:
                a += ["--adaptive_noise_scale", str(p["adaptive_noise_scale"])]
        elif int(p.get("multires_noise_iterations") or 0) > 0:
            a += ["--multires_noise_iterations", str(int(p["multires_noise_iterations"])),
                  "--multires_noise_discount", str(p.get("multires_noise_discount") or 0.3)]
        if float(p.get("ip_noise_gamma") or 0) > 0:
            a += ["--ip_noise_gamma", str(p["ip_noise_gamma"])]
            if p.get("ip_noise_gamma_random_strength"):
                a.append("--ip_noise_gamma_random_strength")
        if float(p.get("validation_split") or 0) > 0:
            for k in ("validate_every_n_epochs", "validate_every_n_steps", "max_validation_steps"):
                if int(p.get(k) or 0) > 0:
                    a += ["--" + k, str(int(p[k]))]
        if p.get("train_te"):
            a += ["--text_encoder_lr", str(p.get("text_encoder_lr") or 1e-5)]
            if p.get("train_te_only"):
                a.append("--network_train_text_encoder_only")
        else:
            a.append("--network_train_unet_only")
        # 共用参数（调度器、日志、续训……）；SDXL 没有 --split_attn / --weighting_scheme，先去掉相关设置
        a += common_args(dict(p, attn_mode="", split_attn=False, weighting_scheme="none"), task_dir)
        if (p.get("optimizer_args") or "").strip():
            a += ["--optimizer_args"] + split_args(p["optimizer_args"])
        if (p.get("loss_type") or "l2") != "l2":
            a += ["--loss_type", p["loss_type"], "--huber_c", str(p.get("huber_c") or 0.1),
                  "--huber_schedule", p.get("huber_schedule") or "snr", "--huber_scale", str(p.get("huber_scale") or 1.0)]
        v = str(p.get("unet_lr") or "").strip()
        if v:
            float(v)
            a += ["--unet_lr", v]
        if p.get("reg_data_dir"):
            a += ["--prior_loss_weight", str(p.get("prior_loss_weight") or 1.0)]
        net_args = common_network_args(p)
        if p.get("network_type") == "lokr" and int(p.get("lokr_factor") or -1) != -1:
            net_args.append(f"factor={int(p['lokr_factor'])}")
        if (p.get("network_type") or "lora") == "lora" and int(p.get("conv_dim") or 0) > 0:
            net_args += [f"conv_dim={int(p['conv_dim'])}", f"conv_alpha={p.get('conv_alpha') or 1}"]
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

        smp = self._sample_args(task, task_dir)
        if smp:
            smp += ["--sample_sampler", p.get("sample_sampler") or "euler_a"]
        a += smp
        a += split_args(p.get("extra_args") or "")
        return [{"name": "训练", "kind": "train", "cmd": a, "cwd": paths.SD_SCRIPTS, "notes": notes}]
